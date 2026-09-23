# SPDX-License-Identifier: OpenMDW-1.1
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.humangen_dataset import (
    HumanGenPairedDataset,
    build_humangen_manifest,
    decode_frames,
    safe_output,
)
from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import robotwin_relative_action


def write_video(path, count=12):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), (i * 15) % 256, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def dataset_tree(tmp_path):
    root = tmp_path / "HumanGen"
    samples = []
    collection_meta = root / "agibot_data/agibot_data/meta"
    collection_meta.mkdir(parents=True)
    transform = dict(
        states=[{"observation.state.arm.position": {"origin_keys": "observation.state"}}],
        actions=[{"action.arm.position": {"origin_keys": "actions.joint.position", "absolute_value": True}}],
        images=[
            {"observation.images.camera_top": {"origin_keys": "head"}},
            {"observation.images.camera_wrist_left": {"origin_keys": "wrist_left"}},
            {"observation.images.camera_wrist_right": {"origin_keys": "wrist_right"}},
        ],
        norm_stats="action_stats.json",
    )
    (collection_meta / "action_transform.yaml").write_text(json.dumps(transform))
    stats = dict(method="abs", norm_stats={"action.arm.position": {"q01": [0.0] * 14, "q99": [20.0] * 14}})
    (collection_meta / "action_stats.json").write_text(json.dumps(stats))
    for i in range(30):
        task = f"task_{i}"
        repo = root / "agibot_data/agibot_data" / task
        (repo / "meta").mkdir(parents=True)
        info = dict(
            fps=10,
            chunks_size=1000,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        (repo / "meta/info.json").write_text(json.dumps(info))
        original = f"{task}/videos/chunk-000/head/episode_000230.mp4"
        clipped = original.replace(".mp4", "_669_681.mp4")
        (repo / "meta/episodes.jsonl").write_text(
            json.dumps(dict(episode_index=0, source=dict(video_rel_path=clipped)))
        )
        p = repo / "data/chunk-000/episode_000000.parquet"
        p.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                dict(
                    frame_index=list(range(12)),
                    **{
                        "actions.joint.position": [[float(j)] * 14 for j in range(12)],
                        "observation.state": [[0.0] * 14 for _ in range(12)],
                    },
                )
            ),
            p,
        )
        write_video(repo / "videos/chunk-000/head/episode_000000.mp4")
        write_video(repo / "videos/chunk-000/wrist_left/episode_000000.mp4")
        write_video(repo / "videos/chunk-000/wrist_right/episode_000000.mp4")
        human = f"agibot/run_test/samples/{i}/human.mp4"
        write_video(root / "human_data/human_data" / human)
        latent = repo / "latents/chunk-000/head/episode_000000_0_12.pth"
        latent.parent.mkdir(parents=True)
        meta = dict(frame_ids=list(range(9)), fps=10, ori_fps=10, start_frame=0, end_frame=12)
        torch.save(meta, latent)
        human_latent = root / "human_latents/agibot/run_test/samples" / str(i) / "human.pth"
        human_latent.parent.mkdir(parents=True)
        torch.save(meta, human_latent)
        samples.append(
            dict(
                sample_id=i,
                human_video_path=f"icl_raw_data/human_data/{human}",
                robot_task_name=task,
                status_info=dict(
                    video_rel_path=clipped, original_video_rel_path=original, interval_enable=True, interval=[669, 681]
                ),
            )
        )
    (root / "icl_configs").mkdir()
    (root / "icl_configs/ICL_config_agibot.json").write_text(json.dumps(dict(samples=samples)))
    return root


def test_action_column_projection(tmp_path):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import _table, action_table_columns

    contract = dict(
        kind="metadata",
        features=[
            dict(
                action={"origin_keys": [{"raw_action": {"start": 0, "end": 2}}]},
                state={"origin_keys": "raw_state"},
            )
        ],
    )
    columns = action_table_columns(contract)
    assert columns == ("frame_index", "raw_action", "raw_state")
    assert action_table_columns({"kind": "robotwin"}) == ("action", "frame_index", "observation.state")
    table = pa.table(
        {
            "frame_index": [0, 1],
            "raw_action": [[1.0, 2.0], [3.0, 4.0]],
            "raw_state": [[0.0, 0.0], [1.0, 1.0]],
            "subtask_annotation": [[1, 2, 3], [4]],
        }
    )
    path = tmp_path / "episode.parquet"
    pq.write_table(table, path)
    assert _table(str(path), columns).equals(table.select(columns))
    with pytest.raises(RuntimeError, match="Missing required columns"):
        _table(str(path), ("missing_action",))


def test_episode_reader_preserves_fixed_size_vectors(tmp_path):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import _table

    expected = pa.table(
        {
            "frame_index": [0, 1, 2],
            "action": pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type=pa.list_(pa.float32(), 3)),
            "observation.state": pa.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], type=pa.list_(pa.float32(), 2)),
        }
    )
    path = tmp_path / "episode.parquet"
    pq.write_table(expected, path, row_group_size=1)
    assert _table(str(path)).equals(expected)


def test_episode_read_failure_identifies_file(tmp_path):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import _table

    path = tmp_path / "invalid_episode.parquet"
    path.write_bytes(b"invalid parquet")
    with pytest.raises(RuntimeError, match="invalid_episode.parquet") as caught:
        _table(str(path))
    assert caught.value.__cause__ is not None


def test_frame_limits_filter_whole_pairs(dataset_tree, tmp_path):
    manifest = build_humangen_manifest(dataset_tree, ["agibot"])
    pairs = manifest["splits"]["train"]
    pairs[0]["robot_frame_ids"] = list(range(101))
    pairs[1]["human_frame_ids"] = list(range(101))
    output = tmp_path / "pairs.json"
    output.write_text(json.dumps(manifest))
    unlimited = HumanGenPairedDataset(dataset_tree, output, "generator")
    limited = HumanGenPairedDataset(dataset_tree, output, "generator", max_robot_frames=9, max_human_frames=9)
    assert len(limited) == len(unlimited) - 2
    assert limited.pairs == pairs[2:]
    with pytest.raises(ValueError, match="nonnegative"):
        HumanGenPairedDataset(dataset_tree, output, "generator", max_robot_frames=-1)
    with pytest.raises(ValueError, match="No HumanGen pairs"):
        HumanGenPairedDataset(dataset_tree, output, "generator", max_robot_frames=1)


def test_reindexed_pair_and_split(dataset_tree, tmp_path):
    manifest = build_humangen_manifest(dataset_tree, ["agibot"])
    assert not manifest["rejected"]
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    assert train and val
    assert {p["task"] for p in train}.isdisjoint({p["task"] for p in val})
    assert {p["human_video"] for p in train}.isdisjoint({p["human_video"] for p in val})
    assert all(p["episode"] == 0 and p["start"] == 0 and p["num_frames"] == 12 for p in train + val)
    output = tmp_path / "pairs.json"
    output.write_text(json.dumps(manifest))
    dataset = HumanGenPairedDataset(dataset_tree, output, "generator")
    sample = dataset[0]
    assert sample["video"].shape[1] == 18
    assert sample["raw_action_dim"] == 14
    assert sample["action_raw"].shape == (8, 14)
    assert sample["action"].shape == (8, 64)
    torch.testing.assert_close(sample["action"][:, 0], (torch.arange(8, dtype=torch.float32) - 10.0) / 10.0)
    restored = sample["action_processing_record"].action_normalizer.denormalize_action(sample["action"][:, :14])
    torch.testing.assert_close(restored, sample["action_raw"])
    assert sample["robot_camera_count"] == 3
    assert sample["robot_camera_keys"] == ["head", "wrist_left", "wrist_right"]
    assert sample["sequence_plan"].condition_frame_indexes_vision == [0, 1, 2, 3]
    assert sample["sequence_plan"].action_start_frame_offset == pytest.approx(13)
    assert torch.isfinite(sample["action"]).all()


def test_robotwin_held_out_guard(tmp_path):
    root = tmp_path / "HumanGen"
    manifest = tmp_path / "held.json"
    pair = dict(
        source="robotwin",
        pair_id="held",
        parquet="robotwin_data/place_object_scale/data/chunk-000/episode_000000.parquet",
    )
    manifest.write_text(
        json.dumps(
            dict(
                format="humangen_pairs_v4",
                root=str(root.resolve()),
                splits={"train": [pair], "val": [pair], "test": []},
            )
        )
    )
    with pytest.raises(ValueError, match="held-out"):
        HumanGenPairedDataset(root, manifest, "generator", split="train")


def test_relative_robotwin():
    state = np.zeros((8, 16), dtype=np.float32)
    state[:, [6, 14]] = 1
    action = state.copy()
    action[:, 0] = 0.25
    action[:, 7] = 0.8
    result = robotwin_relative_action(action, state)
    np.testing.assert_allclose(result[:, 0], 0.25)
    np.testing.assert_allclose(result[:, 7], 0.8)
    np.testing.assert_allclose(result[:, [6, 14]], 1)


def test_source_write_guard(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        safe_output(tmp_path, tmp_path / "new/pairs.json")
    link = tmp_path.parent / (tmp_path.name + "_link")
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="outside"):
            safe_output(tmp_path, link / "pairs.json")
    finally:
        link.unlink()


def test_decode_fails_without_pair_substitution(tmp_path):
    video = tmp_path / "video.mp4"
    write_video(video)
    assert decode_frames(video, [0, 4, 8]).shape == (3, 3, 32, 32)
    with pytest.raises(ValueError, match="Missing requested"):
        decode_frames(video, [100])


@pytest.mark.parametrize("width", [7, 14, 16, 36])
def test_sampler_preserves_native_action_width(width):
    from types import SimpleNamespace

    from cosmos_framework.inference.icl_policy import sample_icl_policy

    model = SimpleNamespace(generate_samples_from_batch=lambda *a, **kw: {"action": [torch.zeros(8, width)]})
    sample = dict(video=torch.zeros(3, 9, 4, 4), action=torch.zeros(8, 64), action_raw=torch.zeros(8, width))
    action, video = sample_icl_policy(model, sample, decode_video=False)
    assert action.shape == (8, width) and video is None
    model.generate_samples_from_batch = lambda *a, **kw: {"action": [torch.zeros(8, width + 1)]}
    with pytest.raises(ValueError, match="Invalid predicted actions"):
        sample_icl_policy(model, sample, decode_video=False)


@pytest.mark.parametrize("length", [5, 9, 12, 17, 34, 101])
def test_video_alignment_truncates_only_incomplete_tail(length):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import aligned_video_length

    aligned = aligned_video_length(length)
    assert (aligned - 1) % 4 == 0
    assert 0 <= length - aligned <= 3
    assert aligned == 1 + (length - 1) // 4 * 4


@pytest.mark.parametrize("length", [0, 1, 2, 4])
def test_short_video_is_not_padded(length):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import aligned_video_length

    with pytest.raises(ValueError, match="at least 5"):
        aligned_video_length(length)


def test_legacy_manifest_requires_regeneration(dataset_tree, tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(dict(format="humangen_pairs_v1", root=str(dataset_tree))))
    with pytest.raises(ValueError, match="regenerate"):
        HumanGenPairedDataset(dataset_tree, path, "generator")


def test_reasoner_keeps_all_human_frames():
    from types import SimpleNamespace

    from cosmos_framework.data.generator.action.icl_transforms import inject_human_video

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["video"].shape[0] == 34
            return "instruction"

        def __call__(self, **kwargs):
            assert kwargs["videos"][0].shape[0] == 34
            assert kwargs["videos_kwargs"]["video_metadata"][0]["frames_indices"] == list(range(34))
            return dict(
                input_ids=torch.zeros(1, 2, dtype=torch.long),
                pixel_values_videos=torch.zeros(34, 1),
                video_grid_thw=torch.tensor([[34, 1, 1]]),
            )

    sample = dict(sequence_plan=SimpleNamespace(), ai_caption="task")
    inject_human_video(sample, torch.zeros(3, 34, 2, 2), "reasoner", SimpleNamespace(processor=Processor()), 10)


@pytest.mark.parametrize("length", [5, 34])
def test_variable_full_pair_keeps_action_alignment(dataset_tree, tmp_path, length):
    repo = dataset_tree / "agibot_data/agibot_data/task_0"
    pq.write_table(
        pa.table(
            dict(
                frame_index=list(range(length)),
                **{
                    "actions.joint.position": [[float(j)] * 14 for j in range(length)],
                    "observation.state": [[0.0] * 14 for _ in range(length)],
                },
            )
        ),
        repo / "data/chunk-000/episode_000000.parquet",
    )
    write_video(repo / "videos/chunk-000/head/episode_000000.mp4", count=length)
    write_video(repo / "videos/chunk-000/wrist_left/episode_000000.mp4", count=length)
    write_video(repo / "videos/chunk-000/wrist_right/episode_000000.mp4", count=length)
    torch.save(
        dict(frame_ids=list(range(1 + (length - 1) // 4 * 4)), fps=10, ori_fps=10, start_frame=0, end_frame=length),
        repo / "latents/chunk-000/head/episode_000000_0_12.pth",
    )
    manifest = build_humangen_manifest(dataset_tree, ["agibot"])
    path = tmp_path / "full.json"
    path.write_text(json.dumps(manifest))
    split = next(k for k, pairs in manifest["splits"].items() if any(p["task"] == "task_0" for p in pairs))
    dataset = (
        HumanGenPairedDataset(dataset_tree, path, "generator")
        if split == "train"
        else HumanGenPairedDataset(dataset_tree, path, "generator", split=split)
    )
    index = next(i for i, p in enumerate(dataset.pairs) if p["task"] == "task_0")
    sample = dataset[index]
    aligned = 1 + (length - 1) // 4 * 4
    assert sample["robot_source_frames"] == length
    assert sample["robot_valid_frames"] == aligned
    assert sample["human_source_frames"] == 12
    assert sample["human_valid_frames"] == 9
    assert sample["action_raw"].shape == (aligned - 1, 14)
    torch.testing.assert_close(sample["action_raw"][:, 0], torch.arange(aligned - 1, dtype=torch.float32))
    assert sample["sequence_plan"].condition_frame_indexes_action == []
    assert sample["sequence_plan"].action_start_frame_offset == pytest.approx(13)
    assert sample["video_segment_frames"] == [9, aligned]


def test_unmapped_interval_uses_all_and_only_paired_frames(dataset_tree):
    config = dataset_tree / "icl_configs/ICL_config_agibot.json"
    samples = json.loads(config.read_text())
    for sample in samples["samples"]:
        repo = dataset_tree / "agibot_data/agibot_data" / sample["robot_task_name"]
        (repo / "meta/episodes.jsonl").unlink()
        status = sample["status_info"]
        status["original_video_rel_path"] = status["original_video_rel_path"].replace("000230", "000000")
        status["interval"] = [2, 12]
    config.write_text(json.dumps(samples))
    manifest = build_humangen_manifest(dataset_tree, ["agibot"])
    pairs = sum(manifest["splits"].values(), [])
    assert len(pairs) == 30
    assert all(p["start"] == 2 and p["num_frames"] == 10 for p in pairs)


def test_policy_decode_returns_requested_video_length():
    from types import SimpleNamespace

    from cosmos_framework.inference.icl_policy import sample_icl_policy

    seen = []

    def decode(latent):
        seen.append(latent.shape[2])
        return torch.zeros(1, 3, 13, 2, 2)

    model = SimpleNamespace(
        generate_samples_from_batch=lambda *a, **kw: {
            "action": [torch.zeros(11, 14)],
            "vision": [torch.zeros(1, 48, 8, 2, 2)],
        },
        decode=decode,
    )
    sample = dict(
        video=torch.zeros(3, 26, 2, 2),
        action=torch.zeros(11, 64),
        action_raw=torch.zeros(11, 14),
        video_segment_frames=[13, 13],
        human_demo_frames=13,
        robot_valid_frames=12,
    )
    actions, video = sample_icl_policy(model, sample)
    assert seen == [4]
    assert actions.shape == (11, 14)
    assert video.shape == (1, 3, 12, 2, 2)


@pytest.mark.parametrize("mode", ["reasoner", "generator"])
def test_sampling_temporal_positions_match_native_action_clock(mode):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import sampling_temporal_positions

    pair = dict(
        robot_frame_ids=[100, 103, 106, 110, 113, 116, 120, 123, 126],
        fps=30,
        robot_sample_fps=10,
        human_frame_ids=list(range(0, 25, 3)),
        human_source_fps=30,
        human_sample_fps=10,
    )
    positions, offset = sampling_temporal_positions(pair, mode)
    robot_start = 1.2 if mode == "generator" else 0
    # Convert the packer's sampled-frame/4 units back to seconds.
    robot_positions = np.asarray(positions[-3:]) * 4 / 10
    np.testing.assert_allclose(robot_positions, robot_start + np.array([0, 13, 26]) / 30)
    assert (offset - 1) / 30 == pytest.approx(robot_start)


def test_subsampled_video_keeps_native_actions(dataset_tree, tmp_path):
    manifest = build_humangen_manifest(dataset_tree, ["agibot"])
    pair = manifest["splits"]["train"][0]
    pair["robot_frame_ids"] = [0, 2, 5, 7, 10]
    pair["robot_sample_fps"] = 4
    path = tmp_path / "sampled.json"
    path.write_text(json.dumps(manifest))
    sample = HumanGenPairedDataset(dataset_tree, path, "generator")[0]
    assert sample["robot_valid_frames"] == 5
    assert sample["action_valid_steps"] == 10
    assert sample["action_raw"].shape == (10, 14)
    torch.testing.assert_close(sample["action_raw"][:, 0], torch.arange(10, dtype=torch.float32))
    assert sample["conditioning_fps"] == 4
    assert sample["conditioning_fps_action"] == 10
    assert sample["sequence_plan"].condition_frame_indexes_action == []


@pytest.mark.parametrize("mode", ["generator", "reasoner"])
def test_packer_preserves_irregular_video_and_native_action_timestamps(mode):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import sampling_temporal_positions
    from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    pair = dict(
        robot_frame_ids=[100, 103, 106, 110, 113, 116, 120, 123, 126],
        fps=30,
        robot_sample_fps=10,
        human_frame_ids=list(range(0, 25, 3)),
        human_source_fps=30,
        human_sample_fps=10,
    )
    positions, offset = sampling_temporal_positions(pair, mode)
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0],
        action_start_frame_offset=offset,
        vision_temporal_positions=positions,
    )
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 48, len(positions), 4, 4)],
        temporal_positions_vision=[torch.tensor(positions, dtype=torch.float32)],
        x0_tokens_action=[torch.zeros(26, 64)],
        action_domain_id=[torch.tensor(0)],
        fps_vision=torch.tensor([10.0]),
        fps_action=torch.tensor([30.0]),
    )
    packed = pack_input_sequence(
        [plan],
        [[1, 2]],
        data,
        torch.tensor([500.0]),
        special_tokens={"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
        latent_patch_size=2,
        enable_fps_modulation=True,
    )
    video_t = packed.position_ids[0, packed.vision.sequence_indexes].reshape(len(positions), -1)[:, 0]
    action_t = packed.position_ids[0, packed.action.sequence_indexes]
    origin = video_t[0]
    prefix = 7.2 if mode == "generator" else 0
    torch.testing.assert_close(video_t[-3:] - origin, torch.tensor([prefix, prefix + 2.6, prefix + 5.2]))
    torch.testing.assert_close(action_t - origin, prefix + torch.arange(1, 27) * 0.2)


def test_multiple_latent_ranges_do_not_shorten_pair(dataset_tree):
    from cosmos_framework.data.generator.action.datasets.humangen_dataset import episode_metadata, pair_sampling

    repo = dataset_tree / "agibot_data/agibot_data/task_0"
    parquet = repo / "data/chunk-000/episode_000000.parquet"
    pq.write_table(
        pa.table(dict(frame_index=list(range(33)), **{"actions.joint.position": [[float(j)] * 14 for j in range(33)]})),
        parquet,
    )
    torch.save(
        dict(frame_ids=list(range(8, 33, 3)), fps=10, ori_fps=10, start_frame=8, end_frame=33),
        repo / "latents/chunk-000/head/episode_000000_8_33.pth",
    )
    pair = dict(
        source="agibot",
        episode=0,
        start=0,
        num_frames=33,
        fps=10,
        parquet=str(parquet.relative_to(dataset_tree)),
        robot_video=str((repo / "videos/chunk-000/head/episode_000000.mp4").relative_to(dataset_tree)),
        human_video="human_data/human_data/agibot/run_test/samples/0/human.mp4",
    )
    result = pair_sampling(dataset_tree, pair, episode_metadata(repo, 0))
    assert result["robot_frame_ids"] == list(range(9)) + list(range(11, 33, 3))
    assert len(result["robot_sampling_paths"]) == 2


def test_video_metric_matches_vae_content_crop():
    from cosmos_framework.inference.icl_policy import policy_video_mse

    raw = torch.arange(3 * 18 * 32 * 48).reshape(3, 18, 32, 48).remainder(256).byte()
    sample = dict(video=raw, human_demo_frames=9, image_size=torch.tensor([32, 48, 30, 45]))
    decoded = (raw[:, 9:, :16, :32].float() / 127.5 - 1).unsqueeze(0)
    assert policy_video_mse(decoded, sample, 9) == 0
    with pytest.raises(ValueError, match="lengths"):
        policy_video_mse(decoded[:, :, :8], sample, 9)
    with pytest.raises(ValueError, match="exceeds"):
        policy_video_mse(torch.zeros(1, 3, 9, 32, 48), sample, 9)


def test_action_stats_falls_back_to_collection_metadata(dataset_tree):
    from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import build_contract

    repo = dataset_tree / "agibot_data/agibot_data/task_0"
    collection_transform = repo.parent / "meta/action_transform.yaml"
    config = json.loads(collection_transform.read_text())
    config["norm_stats"] = "missing_legacy_reference.json"
    (repo / "meta/action_transform.yaml").write_text(json.dumps(config))
    contract = build_contract(repo, "agibot")
    assert contract["q99"] == [20.0] * 14
    assert str(repo.parent / "meta/action_stats.json") in contract["provenance"]
