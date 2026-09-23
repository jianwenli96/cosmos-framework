# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen HumanGen action semantics, quantiles and camera ordering for each repository."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization

ROBOTWIN_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
FEATURES = ("hand.position", "arm.position", "effector.position")


def build_contract(repo, source):
    repo = Path(repo)
    if source == "robotwin":
        path = repo.parent / "meta/action_stats.json"
        stats = json.loads(path.read_text())["norm_stats"]
        hand, grip = stats["action.hand.position"], stats["action.effector.position"]
        q = {name: hand[name][:7] + grip[name][:1] + hand[name][7:14] + grip[name][1:2] for name in ("q01", "q99")}
        contract = dict(kind="robotwin", cameras=ROBOTWIN_CAMERAS, **q)
        files = [path]
    else:
        transform = next(
            (
                p
                for p in (repo / "meta/action_transform.yaml", repo.parent / "meta/action_transform.yaml")
                if p.is_file()
            ),
            None,
        )
        if transform is None:
            raise ValueError("missing_action_transform")
        config = yaml.safe_load(transform.read_text())
        # Match Zero-WAM: prefer the explicit reference, then task/collection fallbacks.
        candidates = []
        if config.get("norm_stats"):
            candidate = Path(config["norm_stats"])
            candidates.append(candidate if candidate.is_absolute() else transform.parent / candidate)
        candidates.extend((repo / "meta/action_stats.json", repo.parent / "meta/action_stats.json"))
        stats_path = next((path for path in candidates if path.is_file()), candidates[0])
        stats = json.loads(stats_path.read_text())
        if stats.get("method") != "abs":
            raise ValueError("unsupported_action_stats_method")
        states = {k: v for entry in config["states"] for k, v in entry.items()}
        actions = {k: v for entry in config["actions"] for k, v in entry.items()}
        features = []
        q01, q99 = [], []
        for name in FEATURES:
            ak, sk = "action." + name, "observation.state." + name
            if ak not in actions or sk not in states:
                continue
            norm = stats["norm_stats"][ak]
            features.append(dict(name=ak, action=actions[ak], state=states[sk]))
            q01.extend(norm["q01"])
            q99.extend(norm["q99"])
        cameras = []
        for entry in config["images"]:
            for spec in entry.values():
                origin = spec["origin_keys"]
                cameras.extend([origin] if isinstance(origin, str) else [next(iter(x)) for x in origin])
        contract = dict(kind="metadata", features=features, cameras=cameras, q01=q01, q99=q99)
        files = [transform, stats_path]
    q01, q99 = np.asarray(contract["q01"]), np.asarray(contract["q99"])
    if not len(q01) or q01.shape != q99.shape or not np.isfinite([q01, q99]).all() or np.any(q99 < q01):
        raise ValueError("invalid_action_quantiles")
    if not contract["cameras"] or len(set(contract["cameras"])) != len(contract["cameras"]):
        raise ValueError("invalid_camera_keys")
    contract["provenance"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return contract


def origin_value(table, spec):
    if isinstance(spec, str):
        value = np.asarray(table[spec].to_pylist(), dtype=np.float32)
        return value.reshape(len(table), -1)
    return np.concatenate(
        [origin_value(table, k)[:, int(v["start"]) : int(v["end"])] for entry in spec for k, v in entry.items()], axis=1
    )


def relative_pose(state, action, fmt, local=False):
    def rotation(x):
        if fmt == "xyzq":
            return Rotation.from_quat(x[:, 3:7])
        if fmt == "xyze":
            return Rotation.from_euler("xyz", x[:, 3:6])
        if fmt == "xyza":
            return Rotation.from_rotvec(x[:, 3:6])
        raise ValueError(f"Unsupported pose format: {fmt}")

    reference = rotation(state)
    xyz = action[:, :3] - state[:, :3]
    if local:
        xyz = reference.inv().apply(xyz)
    quat = (reference.inv() * rotation(action)).as_quat()
    quat *= np.where(quat[:, -1:] > 0, 1.0, -1.0)
    return np.concatenate([xyz, quat], axis=1)


def relative_action(state, action, spec):
    fmt = spec.get("format", "joint")
    if fmt == "joint":
        return action - state
    dual = {"xyzqxyzq": ("xyzq", 7), "xyzexyze": ("xyze", 6), "xyzaxyza": ("xyza", 6)}
    if fmt in dual:
        fmt, width = dual[fmt]
        return np.concatenate(
            [
                relative_pose(
                    state[:, i : i + width], action[:, i : i + width], fmt, spec.get("use_local_frame", False)
                )
                for i in (0, width)
            ],
            axis=1,
        )
    return relative_pose(state, action, fmt, spec.get("use_local_frame", False))


def robotwin_relative_action(action, state):
    """Convert Robotwin dual-arm poses to deltas from the window initial state."""
    if action.shape[1] != 16 or state.shape[1] != 16:
        raise ValueError("Robotwin requires 16D dual-arm xyz/xyzw/gripper")
    result = action.copy()
    for offset in (0, 8):
        result[:, offset : offset + 3] -= state[0, offset : offset + 3]
        quat = (
            Rotation.from_quat(state[0, offset + 3 : offset + 7]).inv()
            * Rotation.from_quat(action[:, offset + 3 : offset + 7])
        ).as_quat()
        quat *= np.where(quat[:, -1:] > 0, 1, -1)
        result[:, offset + 3 : offset + 7] = quat
    return result


def process_actions(table, first, last, contract):
    """Preserve native-rate [first,last) actions; state shifts precede interval selection."""
    if contract["kind"] == "robotwin":
        action = robotwin_relative_action(
            origin_value(table, "action")[first:last], origin_value(table, "observation.state")[first:last]
        )
    else:
        parts = []
        for feature in contract["features"]:
            spec = feature["action"]
            state_spec = feature["state"]
            action = origin_value(table, spec["origin_keys"])[first:last]
            state = origin_value(table, state_spec["origin_keys"])
            shift = int(state_spec.get("shift", 0))
            if shift < 0:
                raise ValueError("Negative state shifts are unsupported")
            if shift:
                shifted = state.copy()
                shifted[shift:] = state[:-shift]
                state = shifted
            if spec.get("absolute_value", True) and not spec.get("use_absolute", False):
                action = relative_action(np.repeat(state[first : first + 1], len(action), axis=0), action, spec)
            parts.append(action)
        action = np.concatenate(parts, axis=1)
    q01 = torch.tensor(contract["q01"], dtype=torch.float32)
    q99 = torch.tensor(contract["q99"], dtype=torch.float32)
    if action.shape[1] != len(q01) or not np.isfinite(action).all():
        raise ValueError("Action schema/quantile dimension mismatch")
    scale = (q99 - q01 + 1e-6) / 2
    return action.astype(np.float32), ActionAffineNormalization(
        offset=q01 + scale, scale=scale, forward_clamp=(-2.0, 2.0)
    )


def camera_paths(root, pair, contract):
    repo = (Path(root) / pair["parquet"]).parents[2]
    info = json.loads((repo / "meta/info.json").read_text())
    episode = pair["episode"]
    paths = [
        repo
        / info["video_path"].format(
            episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode, video_key=k
        )
        for k in contract["cameras"]
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing_robot_camera:{path}")
    return [str(p.relative_to(root)) for p in paths]
