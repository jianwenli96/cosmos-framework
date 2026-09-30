# SPDX-License-Identifier: OpenMDW-1.1
"""RoboTwin transport and window-relative end-effector action conversion."""

import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERAS = ("head_camera", "left_camera", "right_camera")


def encode_image(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value):
    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as image:
        return np.array(image.convert("RGB"))


def observation_pose(observation):
    pose = observation["endpose"]
    return np.concatenate(
        [
            np.asarray(pose["left_endpose"]),
            [pose["left_gripper"]],
            np.asarray(pose["right_endpose"]),
            [pose["right_gripper"]],
        ]
    )


def absolute_actions(relative, reference):
    """Invert HumanGen robotwin_relative_action using this window's initial pose.

    Both inputs use xyz, quaternion xyzw, absolute gripper for each arm.
    Translation is world-frame; rotation composes reference * relative.
    """
    actions = np.asarray(relative, dtype=np.float64).copy()
    reference = np.asarray(reference, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16 or reference.shape != (16,):
        raise ValueError("Expected actions [T,16] and reference [16]")
    if not len(actions) or not np.isfinite(actions).all() or not np.isfinite(reference).all():
        raise ValueError("Empty or nonfinite RoboTwin action/pose")
    for offset in (0, 8):
        q = actions[:, offset + 3 : offset + 7]
        if np.any(np.linalg.norm(q, axis=1) < 1e-8):
            raise ValueError("Predicted quaternion has zero norm")
        actions[:, offset : offset + 3] += reference[offset : offset + 3]
        actions[:, offset + 3 : offset + 7] = (
            Rotation.from_quat(reference[offset + 3 : offset + 7]) * Rotation.from_quat(q)
        ).as_quat()
    return actions


class RobotwinClient:
    def __init__(self, url, timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.instruction = None
        # Match batch health checks: simulation traffic goes directly to the model host.
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Policy HTTP {error.code} at {request.full_url}: {body}") from error
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def predict(self, observation, seed):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Set the current episode instruction before predicting")
        result = self.request(
            "/predict",
            {
                "images": {key: encode_image(observation["observation"][key]["rgb"]) for key in CAMERAS},
                "seed": seed,
                "instruction": self.instruction,
            },
        )
        return absolute_actions(result["action"], observation_pose(observation))


def rollout(env, client, execute_steps, seed, on_observation=None, video_stride=5):
    """Re-observe after each executed chunk; stop immediately on success/limit."""
    if video_stride < 1:
        raise ValueError("video_stride must be positive")
    if on_observation is not None:
        on_observation(env.get_obs())
    if execute_steps < 1:
        raise ValueError("execute_steps must be positive")
    requests = 0
    while env.take_action_cnt < env.step_lim and not env.eval_success:
        actions = client.predict(env.get_obs(), seed + requests)
        requests += 1
        if len(actions) < execute_steps:
            raise ValueError("execute_steps exceeds the predicted window's native action horizon")
        before = env.take_action_cnt
        for action in actions[:execute_steps]:
            env.take_action(action, action_type="ee")
            if on_observation is not None and (
                env.take_action_cnt % video_stride == 0 or env.eval_success or env.take_action_cnt >= env.step_lim
            ):
                on_observation(env.get_obs())
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
        if env.take_action_cnt <= before:
            raise RuntimeError("RoboTwin take_action did not advance the step counter")
    return {"success": bool(env.eval_success), "steps": int(env.take_action_cnt), "requests": requests}


def live_sample(template, images, resize, resolution):
    """Replace the conditioning image; never expose offline robot targets."""
    from copy import deepcopy

    import torch
    import torch.nn.functional as F

    videos = [torch.from_numpy(decode_image(images[key])).permute(2, 0, 1)[None] for key in CAMERAS]
    height, width = videos[0].shape[-2:]
    tiles = []
    for video in videos:
        h, w = video.shape[-2:]
        scale = min(height / h, width / w)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        tile = F.interpolate(video.float(), size=(rh, rw), mode="bilinear", align_corners=False)
        ph, pw = height - rh, width - rw
        tiles.append(F.pad(tile, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)).round().byte())
    video = torch.cat(tiles, dim=-1).permute(1, 0, 2, 3)
    resized = resize({"video": video}, resolution)
    sample = deepcopy(template)
    prefix = int(sample.get("human_demo_frames", 0))
    if resized["video"].shape[-2:] != sample["video"].shape[-2:]:
        raise ValueError("Live camera aspect ratio differs from the training camera strip")
    sample["video"][:, prefix:] = 0
    sample["video"][:, prefix : prefix + 1] = resized["video"]
    sample["image_size"] = resized["image_size"]
    sample["action"].zero_()
    sample["action_raw"].zero_()
    return sample


def validate_demo_task(pair, task):
    """Do not let a manually supplied task name relabel another task's demonstration."""
    repo = Path(pair["parquet"]).parents[2].name
    if pair.get("source") != "robotwin" or not (repo == task or repo.startswith(task + "-")):
        raise ValueError(f"Demonstration repository {repo!r} does not match RoboTwin task {task!r}")


def instruction_template(dataset, index, pair_index, instruction):
    """Rebuild both text and multimodal tokens using the original preprocessing path.

    A shallow dataset copy keeps the source manifest unchanged. The HTTP server
    caches only the latest instruction to bound demo/token tensor memory.
    """
    from copy import copy

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty episode instruction is required")
    conditioned = copy(dataset)
    conditioned.pairs = list(dataset.pairs)
    conditioned.pairs[pair_index] = dict(dataset.pairs[pair_index], caption=instruction)
    sample = conditioned[index]
    prefix = int(sample.get("human_demo_frames", 0))
    sample["video"][:, prefix:] = 0
    sample["action"].zero_()
    sample["action_raw"].zero_()
    return sample
