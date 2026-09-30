# SPDX-License-Identifier: OpenMDW-1.1
"""Read HumanGen's explicit human/robot pairs without writing into the source tree."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.domain_utils import get_domain_id

SOURCES = ("robotwin", "agibot", "robocoin", "robomind", "interna1", "oxe")
HELD_OUT = (
    "place_object_scale",
    "stamp_seal",
    "open_microwave",
    "move_stapler_pad",
    "place_bread_basket",
    "place_empty_cup",
    "stack_blocks_three",
)


def source_repositories(root):
    """Index metadata only, pruning video/data/archive trees during traversal."""
    import os

    result = defaultdict(list)
    for directory, dirs, files in os.walk(root):
        if Path(directory).name == "meta" and "info.json" in files:
            repo = Path(directory).parent
            result[repo.name].append(repo)
        dirs[:] = [d for d in dirs if d not in {"videos", "data", "latents", "meta", ".cache"}]
        # meta is inspected explicitly because it was pruned above.
        meta = Path(directory) / "meta/info.json"
        if meta.is_file():
            result[Path(directory).name].append(Path(directory))
    return result


def safe_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the original HumanGen directory")
    return output


def build_humangen_manifest(root, sources=SOURCES, seed=42, val_ratio=0.1):
    root = Path(root).resolve()
    if not 0 < val_ratio < 1:
        raise ValueError("Invalid val_ratio")
    splits = {"train": [], "val": [], "test": []}
    rejected = Counter()
    for source in sources:
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        repos = source_repositories(root / f"{source}_data")
        manifest = root / "icl_configs" / f"ICL_config_{source}.json"
        samples = json.loads(manifest.read_text())["samples"]
        episode_lookup = defaultdict(list)
        for paths in repos.values():
            for repo in paths:
                metadata = repo / "meta/episodes.jsonl"
                if metadata.is_file():
                    for line in metadata.read_text().splitlines():
                        ep = json.loads(line)
                        source_rel = ep.get("source", {}).get("video_rel_path")
                        if source_rel:
                            episode_lookup[source_rel].append((repo, ep))
        for sample in samples:
            status = sample.get("status_info") or {}
            pair_rel = status.get("video_rel_path") or sample.get("robot_video_path")
            rel = status.get("original_video_rel_path") or pair_rel
            task, video_rel = rel.split("/videos/", 1)
            matches = episode_lookup.get(pair_rel, [])
            exact = [
                (repo, ep)
                for repo, ep in matches
                if ep.get("icl", {}).get("human_video_path") == sample["human_video_path"]
            ]
            if exact:
                matches = exact
            mapped = min(matches, key=lambda item: (str(item[0]), int(item[1]["episode_index"]))) if matches else None
            if episode_lookup and mapped is None:
                rejected[f"{source}:unmatched_source_episode"] += 1
                continue
            if mapped:
                repo, ep = mapped
                episode = int(ep["episode_index"])
                info = _json(str(repo / "meta/info.json"))
                video_key = Path(video_rel).parent.name
                video = repo / info["video_path"].format(
                    episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode, video_key=video_key
                )
            else:
                candidates = repos.get(Path(task).name, []) or repos.get(task.replace("/", "__"), [])
                if len(candidates) != 1:
                    rejected[f"{source}:missing_or_ambiguous_repo"] += 1
                    continue
                repo = candidates[0]
                info = _json(str(repo / "meta/info.json"))
                match = re.fullmatch(r"episode_(\d+)(?:_\d+_\d+)?\.mp4", Path(video_rel).name)
                if match is None:
                    raise ValueError(f"Unsupported video name: {rel}")
                episode = int(match[1])
                video = repo / "videos" / video_rel
            parquet = repo / info["data_path"].format(
                episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode
            )
            human_rel = sample["human_video_path"].split("human_data/", 1)[-1]
            human_candidates = [root / "human_data" / human_rel, root / "human_data/human_data" / human_rel]
            human = next((p for p in human_candidates if p.is_file()), human_candidates[0])
            if not all(p.is_file() for p in (parquet, video, human)):
                rejected[f"{source}:missing_pair_file"] += 1
                continue
            table = _table(str(parquet), ("frame_index",))
            indexes = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            if not len(indexes):
                rejected[f"{source}:empty_episode"] += 1
                continue
            interval = status.get("interval") if status.get("interval_enable") and not mapped else None
            lower, upper = interval if interval else (int(indexes[0]), int(indexes[-1]) + 1)
            valid = np.flatnonzero((indexes >= lower) & (indexes < upper))
            if len(valid) < 5 or np.any(np.diff(indexes[valid]) != 1) or np.any(np.diff(valid) != 1):
                rejected[f"{source}:short_or_noncontiguous_interval"] += 1
                continue
            # Split whole tasks, so reused human clips and robot episodes never leak.
            held = source == "robotwin" and any(repo.name == t or repo.name.startswith(t + "-") for t in HELD_OUT)
            fraction = int(hashlib.sha256(f"{seed}:{source}:{task}".encode()).hexdigest()[:8], 16) / 2**32
            split = "test" if held else ("val" if fraction < val_ratio else "train")
            start = int(valid[0])
            pair = dict(
                source=source,
                task=task,
                episode=episode,
                start=start,
                num_frames=int(len(valid)),
                parquet=str(parquet.relative_to(root)),
                robot_video=str(video.relative_to(root)),
                human_video=str(human.relative_to(root)),
                fps=float(info["fps"]),
                caption=sample.get("robot_task_name") or status.get("fine_task") or status["coarse_task"],
                pair_id=f"{source}:{sample.get('run', '')}:{sample.get('sample', sample.get('sample_id'))}:{episode}:{start}",
            )
            try:
                pair.update(pair_sampling(root, pair, ep if mapped else episode_metadata(repo, episode)))
                from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import (
                    build_contract,
                    camera_paths,
                )

                contract = build_contract(repo, source)
                pair["robot_videos"] = camera_paths(root, pair, contract)
                pair["robot_camera_keys"] = list(contract["cameras"])
                pair["preprocessing_provenance"] = contract["provenance"]
            except (FileNotFoundError, ValueError, KeyError) as error:
                rejected[f"{source}:preprocessing:{error}"] += 1
                continue
            splits[split].append(pair)
        counts = {key: sum(p["source"] == source for p in pairs) for key, pairs in splits.items()}
        if sum(counts.values()) == 0:
            raise ValueError(f"No usable pairs for requested source {source}: {dict(rejected)}")
        print(f"HumanGen {source}: {counts}", flush=True)
    if not splits["train"] or not splits["val"]:
        raise ValueError(
            f"Empty train/val split; counts={ {k: len(v) for k, v in splits.items()} }; rejected={dict(rejected)}"
        )
    return dict(
        format="humangen_pairs_v4",
        sampling="latent_frame_ids",
        root=str(root),
        seed=seed,
        splits=splits,
        rejected=dict(rejected),
        action_contract="metadata_semantics_relative_quantile_q01_q99_clamp2; robotwin_relative_quantile; metadata_cameras",
        domain_ids={source: get_domain_id(f"humangen_{source}") for source in sources},
    )


@lru_cache(maxsize=256)
def _json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=8)
def _table(path, columns=None):
    # Each entry names one episode file. Avoid the dataset scanner and its
    # nested thread pools: DataLoader workers already parallelize episode I/O.
    try:
        with pq.ParquetFile(path, pre_buffer=False) as parquet:
            if columns is not None:
                missing = set(columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"Missing required columns: {sorted(missing)}")
            return parquet.read(columns=columns, use_threads=False)
    except Exception as error:
        raise RuntimeError(f"Failed to read HumanGen parquet {path}, columns={columns}: {error}") from error


def action_table_columns(contract):
    """Project only frame IDs and the raw columns used by action preprocessing."""
    columns = {"frame_index"}
    if contract["kind"] == "robotwin":
        columns.update(("action", "observation.state"))
    else:
        for feature in contract["features"]:
            for role in ("action", "state"):
                origin = feature[role]["origin_keys"]
                if isinstance(origin, str):
                    columns.add(origin)
                else:
                    for entry in origin:
                        columns.update(entry)
    return tuple(sorted(columns))


@lru_cache(maxsize=256)
def action_camera_contract(repo, source):
    from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import build_contract

    return build_contract(repo, source)


def is_robotwin_held_out(root, pair):
    if pair.get("source") != "robotwin":
        return False
    repo = (Path(root) / pair["parquet"]).parents[2]
    return any(repo.name == task or repo.name.startswith(task + "-") for task in HELD_OUT)


def decode_frames(path, indexes):
    """Exact frame-index decode; fail instead of silently substituting another pair."""
    import av

    wanted = list(map(int, indexes))
    wanted_set = set(wanted)
    last_index = max(wanted)
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted_set:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= last_index:
                break
    if any(index not in frames for index in wanted):
        raise ValueError(f"Missing requested frames in {path}: {wanted}")
    return torch.from_numpy(np.stack([frames[i] for i in wanted])).permute(3, 0, 1, 2).contiguous()


def decode_camera_strip(root, paths, frame_ids):
    """Decode synchronized camera views and concatenate letterboxed tiles."""
    import torch.nn.functional as F

    videos = [decode_frames(Path(root) / path, frame_ids) for path in paths]
    if len(videos) == 1:
        return videos[0]
    height, width = videos[0].shape[-2:]
    tiles = []
    for video in videos:
        h, w = video.shape[-2:]
        scale = min(height / h, width / w)
        resized_h, resized_w = max(1, round(h * scale)), max(1, round(w * scale))
        resized = F.interpolate(
            video.permute(1, 0, 2, 3).float(),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        pad_h, pad_w = height - resized_h, width - resized_w
        tile = F.pad(resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        tiles.append(tile.round().clamp(0, 255).byte().permute(1, 0, 2, 3))
    return torch.cat(tiles, dim=-1)


class HumanGenPairedDataset(Dataset):
    def __init__(
        self,
        root,
        manifest,
        injection_mode,
        split="train",
        resolution="256",
        max_action_dim=64,
        tokenizer_config=None,
        max_robot_frames=0,
        max_human_frames=0,
        robot_window_frames=0,
        robot_window_stride=1,
    ):
        from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
        from cosmos_framework.utils.lazy_config import instantiate

        self.root = Path(root).resolve()
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest.get("format") in {"humangen_pairs_v1", "humangen_pairs_v2", "humangen_pairs_v3"}:
            raise ValueError("Legacy HumanGen preprocessing manifest; regenerate with prepare_humangen_pairs")
        if self.manifest.get("format") != "humangen_pairs_v4" or Path(self.manifest["root"]).resolve() != self.root:
            raise ValueError("HumanGen manifest format/root mismatch")
        self.pairs = self.manifest["splits"][split]
        if max_robot_frames < 0 or max_human_frames < 0:
            raise ValueError("Frame limits must be nonnegative; 0 disables the limit")
        before = len(self.pairs)
        self.pairs = [
            pair
            for pair in self.pairs
            if (not max_robot_frames or len(pair["robot_frame_ids"]) <= max_robot_frames)
            and (not max_human_frames or len(pair["human_frame_ids"]) <= max_human_frames)
        ]
        if before != len(self.pairs):
            from cosmos_framework.utils import log

            log.warning(f"HumanGen {split}: frame limits kept {len(self.pairs)}/{before} pairs")
        if not self.pairs:
            raise ValueError(f"No HumanGen pairs for split {split}")
        if split != "test":
            held_out = [pair["pair_id"] for pair in self.pairs if is_robotwin_held_out(self.root, pair)]
            if held_out:
                raise ValueError(f"Robotwin held-out tasks present in {split}: {held_out[:8]}")
        if not isinstance(robot_window_frames, int) or (
            robot_window_frames != 0 and (robot_window_frames < 5 or (robot_window_frames - 1) % 4)
        ):
            raise ValueError("robot_window_frames must be 0 (full pair) or 4k+1, at least 5")
        if not isinstance(robot_window_stride, int) or robot_window_stride < 1:
            raise ValueError("robot_window_stride must be a positive integer")
        self.robot_window_frames = robot_window_frames
        self.robot_window_stride = robot_window_stride
        # Compact cumulative index, like LIBERO: no materialized copy per window.
        counts = [
            max(0, (len(pair["robot_frame_ids"]) - robot_window_frames) // robot_window_stride + 1)
            if robot_window_frames
            else 1
            for pair in self.pairs
        ]
        self._window_counts = np.asarray(counts, dtype=np.int64)
        self._window_cumulative = np.cumsum(self._window_counts)
        if not len(self):
            raise ValueError(f"No HumanGen windows for split {split}; reduce robot_window_frames")
        from cosmos_framework.utils import log

        log.info(
            f"HumanGen {split}: {len(self.pairs)} pairs, {len(self)} samples, "
            f"robot_window_frames={robot_window_frames}, stride={robot_window_stride}, "
            f"short_pairs={sum(count == 0 for count in counts)}"
        )
        self.mode, self.resolution = injection_mode, resolution
        self.max_action_dim = max_action_dim
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            max_action_dim=max_action_dim,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
        )
        self.processor = instantiate(tokenizer_config) if tokenizer_config is not None else None

    def __len__(self):
        return int(self._window_cumulative[-1])

    def get_shuffle_blocks(self):
        return [
            (int(end - count), int(count)) for end, count in zip(self._window_cumulative, self._window_counts) if count
        ]

    def _window_pair(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pair_index = int(np.searchsorted(self._window_cumulative, index, side="right"))
        base = int(self._window_cumulative[pair_index - 1]) if pair_index else 0
        offset = (index - base) * self.robot_window_stride
        pair = self.pairs[pair_index]
        if self.robot_window_frames:
            pair = dict(pair, robot_frame_ids=pair["robot_frame_ids"][offset : offset + self.robot_window_frames])
        return pair, int(offset)

    def __getitem__(self, index):
        from cosmos_framework.data.generator.action.icl_transforms import inject_human_video

        pair, window_offset = self._window_pair(index)
        repo = (self.root / pair["parquet"]).parents[2]
        contract = action_camera_contract(str(repo), pair["source"])
        table = _table(str(self.root / pair["parquet"]), action_table_columns(contract))
        start = pair["start"]
        source_frames = int(pair["num_frames"])
        frame_ids = np.asarray(pair["robot_frame_ids"], dtype=np.int64)
        original_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        first, last = np.searchsorted(original_ids, frame_ids[[0, -1]])
        if last >= len(original_ids) or original_ids[first] != frame_ids[0] or original_ids[last] != frame_ids[-1]:
            raise ValueError(f"Sampled frames outside episode: {pair['pair_id']}")
        if first < start or last >= start + source_frames:
            raise ValueError(f"Sampled frames outside pair: {pair['pair_id']}")
        # Preserve all native-rate actions between first and last sampled observation.
        frames = int(last - first + 1)
        window = table.slice(int(first), frames)
        if np.any(np.diff(window["frame_index"].to_numpy()) != 1):
            raise ValueError(f"Noncontiguous native action interval: {pair['pair_id']}")
        from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import camera_paths, process_actions

        if pair.get("preprocessing_provenance") != contract["provenance"]:
            raise ValueError(f"HumanGen preprocessing metadata changed; regenerate manifest: {pair['pair_id']}")
        if pair.get("robot_camera_keys") != list(contract["cameras"]):
            raise ValueError(f"HumanGen camera metadata changed; regenerate manifest: {pair['pair_id']}")
        action, action_normalizer = process_actions(table, int(first), int(last), contract)
        if action.ndim != 2 or action.shape[1] > self.max_action_dim or not np.isfinite(action).all():
            raise ValueError(f"Invalid action shape/values: {pair['pair_id']}")
        robot_paths = pair["robot_videos"]
        if robot_paths != camera_paths(self.root, pair, contract):
            raise ValueError(f"HumanGen camera paths changed; regenerate manifest: {pair['pair_id']}")
        robot = self.transform(
            dict(
                video=decode_camera_strip(self.root, robot_paths, frame_ids),
                action=torch.from_numpy(action.copy()),
                mode="policy",
                ai_caption=pair["caption"],
                conditioning_fps=torch.tensor(pair["robot_sample_fps"]),
                conditioning_fps_action=torch.tensor(pair["fps"]),
                viewpoint="third_person_view",
                domain_id=torch.tensor(self.manifest["domain_ids"][pair["source"]]),
            ),
            self.resolution,
            action_normalizer=action_normalizer,
        )
        human_path = self.root / pair["human_video"]
        indexes = np.asarray(pair["human_frame_ids"], dtype=np.int64)
        human = decode_frames(human_path, indexes)
        human = self.transform.video_resize(dict(video=human), self.resolution)["video"]
        robot["pair_id"] = pair["pair_id"]
        robot["robot_window_start"] = window_offset
        robot["robot_frame_ids"] = frame_ids.tolist()
        robot["sample_id"] = f"{pair['pair_id']}:robot_window:{window_offset}:{len(frame_ids)}"
        robot["robot_camera_keys"] = list(contract["cameras"])
        robot["robot_camera_count"] = len(contract["cameras"])
        robot["robot_source_frames"] = source_frames
        robot["human_source_frames"] = pair["human_source_frames"]
        robot["robot_valid_frames"] = len(frame_ids)
        robot["human_valid_frames"] = len(indexes)
        robot["action_valid_steps"] = frames - 1
        plan = robot["sequence_plan"]
        plan.condition_frame_indexes_action = []
        plan.action_start_frame_offset = 1
        robot = inject_human_video(
            robot,
            human,
            self.mode,
            self.processor,
            pair["human_sample_fps"],
            video_metadata=dict(
                fps=pair["human_source_fps"],
                total_num_frames=pair["human_source_frames"],
                frames_indices=indexes.tolist(),
            ),
        )
        positions, action_offset = sampling_temporal_positions(pair, self.mode)
        plan.vision_temporal_positions = positions
        plan.action_start_frame_offset = action_offset
        return robot


def get_humangen_training_dataset(**kwargs):
    return HumanGenPairedDataset(**kwargs)


def sampling_temporal_positions(pair, mode):
    """Exact source timestamps, expressed in sampled-video frame / 4 units for mRoPE."""
    robot_ids = np.asarray(pair["robot_frame_ids"], dtype=np.float64)
    robot_seconds = (robot_ids[::4] - robot_ids[0]) / pair["fps"]
    prefix_seconds = 0.0
    if mode == "generator":
        human_ids = np.asarray(pair["human_frame_ids"], dtype=np.float64)
        human_seconds = (human_ids[::4] - human_ids[0]) / pair["human_source_fps"]
        prefix_seconds = human_seconds[-1] + 4.0 / pair["human_sample_fps"]
        seconds = np.concatenate([human_seconds, prefix_seconds + robot_seconds])
    else:
        seconds = robot_seconds
    return (seconds * pair["robot_sample_fps"] / 4.0).tolist(), float(1.0 + prefix_seconds * pair["fps"])


def aligned_video_length(length):
    """Use the longest complete 4k+1 prefix, never synthesize tail frames."""
    if length < 5:
        raise ValueError("Video needs at least 5 frames for a video/action training sample")
    return 1 + ((length - 1) // 4) * 4


@lru_cache(maxsize=64)
def _episodes(repo):
    path = Path(repo) / "meta/episodes.jsonl"
    if not path.is_file():
        return {}
    return {int(ep["episode_index"]): ep for line in path.read_text().splitlines() if (ep := json.loads(line))}


def episode_metadata(repo, episode):
    return _episodes(str(repo)).get(episode, {})


@lru_cache(maxsize=256)
def sampling_metadata(path):
    # mmap keeps large latent tensors off the CPU heap; only metadata is consumed.
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = np.asarray(data["frame_ids"])
    if ids.ndim != 1 or len(ids) < 5 or not np.isfinite(ids).all() or not np.equal(ids, ids.astype(np.int64)).all():
        raise ValueError("invalid_latent_frame_ids")
    ids = ids.astype(np.int64)
    if np.any(np.diff(ids) <= 0) or (len(ids) - 1) % 4:
        raise ValueError("invalid_latent_frame_order_or_length")
    fps, original_fps = float(data["fps"]), float(data["ori_fps"])
    if not np.isfinite([fps, original_fps]).all() or min(fps, original_fps) <= 0:
        raise ValueError("invalid_latent_fps")
    return dict(
        frame_ids=ids.tolist(),
        fps=fps,
        ori_fps=original_fps,
        start_frame=int(data.get("start_frame", ids[0])),
        end_frame=int(data.get("end_frame", ids[-1] + 1)),
    )


def pair_sampling(root, pair, episode):
    repo = (root / pair["parquet"]).parents[2]
    camera = Path(pair["robot_video"]).parent.name
    source = episode.get("source", {})
    source_episode = int(source.get("episode_index", pair["episode"]))
    source_start = int(source.get("frame_range", [0])[0])
    candidates = sorted(
        (repo / "latents" / f"chunk-{source_episode // 1000:03d}" / camera).glob(f"episode_{source_episode:06d}_*.pth")
    )
    table = _table(str(root / pair["parquet"]), ("frame_index",))
    window = table.slice(pair["start"], pair["num_frames"])
    local_ids = np.asarray(window["frame_index"].to_numpy(), dtype=np.int64)
    lower, upper = int(local_ids[0]), int(local_ids[-1])
    # Preserve all cached sampling ranges within the pair, including split latent files.
    matches = []
    for path in candidates:
        meta = sampling_metadata(str(path))
        if not np.isclose(meta["ori_fps"], pair["fps"]):
            raise ValueError("robot_latent_source_fps_mismatch")
        ids = np.asarray(meta["frame_ids"], dtype=np.int64) - source_start
        ids = ids[(ids >= lower) & (ids <= upper)]
        if len(ids):
            matches.append((path, meta, ids))
    if not matches:
        raise FileNotFoundError("no_matching_robot_latent_sampling")
    path, meta, _ = matches[0]
    if any(not np.isclose(item[1]["fps"], meta["fps"]) for item in matches):
        raise ValueError("mixed_sampling_fps_within_pair")
    ids = np.unique(np.concatenate([item[2] for item in matches]))
    ids = ids[: aligned_video_length(len(ids))]
    parts = Path(pair["human_video"]).parts
    run = next((i for i, part in enumerate(parts) if part.startswith("run_")), None)
    if run is None:
        raise ValueError("human_latent_run_missing")
    human_path = (root / "human_latents" / pair["source"] / Path(*parts[run:])).with_suffix(".pth")
    if not human_path.is_file():
        raise FileNotFoundError("missing_human_latent_sampling")
    human = sampling_metadata(str(human_path))
    return dict(
        robot_frame_ids=ids.tolist(),
        robot_sample_fps=meta["fps"],
        robot_sampling_paths=[str(item[0].relative_to(root)) for item in matches],
        human_frame_ids=human["frame_ids"],
        human_sample_fps=human["fps"],
        human_source_fps=human["ori_fps"],
        human_source_frames=human["end_frame"],
        human_sampling_path=str(human_path.relative_to(root)),
    )
