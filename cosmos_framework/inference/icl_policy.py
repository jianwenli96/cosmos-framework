# SPDX-License-Identifier: OpenMDW-1.1
"""Offline HumanGen ICL policy sampling helpers."""

from copy import deepcopy

import numpy as np
import torch


def policy_batch(sample: dict) -> dict:
    """Build an inference batch while hiding future robot pixels and actions."""
    sample = deepcopy(sample)
    prefix = int(sample.get("human_demo_frames", 0))
    sample["video"][:, prefix + 1 :] = 0
    sample["action"].zero_()
    sample["action_raw"].zero_()
    return pack_policy_sample(sample)


def pack_policy_sample(sample: dict) -> dict:
    """Pack one sample in the nested layout used by PackingDataLoader."""
    multi_value_keys = {"video", "text_token_ids", "action", "action_raw"}
    batch = {}
    for key, value in sample.items():
        if key in multi_value_keys:
            batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            batch[key] = [value.unsqueeze(0)]
        else:
            batch[key] = [value]
    return batch


@torch.no_grad()
def sample_icl_policy(
    model,
    sample: dict,
    seed: int = 42,
    num_steps: int = 20,
    decode_video: bool = True,
):
    """Sample external-space actions and the future robot video."""
    generated = model.generate_samples_from_batch(
        policy_batch(sample),
        seed=[seed],
        guidance=1.0,
        num_steps=num_steps,
        shift=3.0,
    )
    actions = generated["action"][0].float().cpu().numpy()
    expected_shape = tuple(sample["action_raw"].shape)
    if actions.shape != expected_shape or not np.isfinite(actions).all():
        raise ValueError(f"Invalid predicted actions: got {actions.shape}, expected {expected_shape}")

    if not decode_video:
        return actions, None

    latent = generated["vision"][0]
    if "video_segment_frames" in sample:
        human_frames = int(sample["video_segment_frames"][0])
        human_latents = (human_frames - 1) // 4 + 1
        latent = latent[:, :, human_latents:]

    # Robot latents are decoded independently from the demonstration prefix.
    video = model.decode(latent).float().cpu()
    if "robot_valid_frames" in sample:
        video = video[:, :, : int(sample["robot_valid_frames"])]
    return actions, video


def policy_video_mse(video, sample, robot_frames: int):
    """Compare decoded future robot content, excluding spatial padding."""
    from cosmos_framework.data.generator.action.transforms import remove_reflection_padding

    prefix = int(sample.get("human_demo_frames", 0))
    reference = sample["video"][:, prefix : prefix + robot_frames].float().cpu() / 127.5 - 1
    reference = remove_reflection_padding(reference, sample.get("image_size"))
    if video.shape[2] != robot_frames or reference.shape[1] != robot_frames:
        raise ValueError("Decoded and reference video lengths must match")
    height, width = video.shape[-2:]
    if height > reference.shape[-2] or width > reference.shape[-1]:
        raise ValueError("Decoded video exceeds reference content dimensions")
    reference = reference[..., :height, :width]
    return (video[0, :, 1:] - reference[:, 1:]).square().mean()
