# SPDX-License-Identifier: OpenMDW-1.1
"""Evaluate a HumanGen ICL checkpoint on real paired episodes."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import yaml

if os.environ.get("COSMOS_DEVICE", "cuda") == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

from cosmos_framework.data.generator.action.datasets.humangen_dataset import (
    HumanGenPairedDataset,
    safe_output,
)
from cosmos_framework.inference.icl_policy import (
    pack_policy_sample,
    policy_video_mse,
    sample_icl_policy,
)
from cosmos_framework.utils import distributed
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint


def _run_dir_from_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint).resolve()
    if path.name == "model" and path.parent.name.startswith("iter_"):
        return path.parents[2]
    raise ValueError(
        "Expected checkpoint path ending in checkpoints/iter_XXXXXXXXX/model so its training config can be recovered"
    )


def _model_overrides_from_training_config(checkpoint: str, mode: str) -> list[str]:
    config_path = _run_dir_from_checkpoint(checkpoint) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing trainer config next to checkpoint: {config_path}")
    saved = yaml.safe_load(config_path.read_text())
    datasets = saved["dataloader_train"]["dataloader"]["datasets"]
    if len(datasets) != 1:
        raise ValueError("HumanGen ICL checkpoint config must contain exactly one training dataset")
    dataset_config = next(iter(datasets.values()))["dataset"]
    saved_mode = dataset_config.get("injection_mode")
    if saved_mode != mode:
        raise ValueError(f"Checkpoint was trained with injection_mode={saved_mode!r}, expected {mode!r}")

    model = saved["model"]["config"]
    structural_keys = ("fsdp_mixed_precision", "precision", "resolution", "max_action_dim")
    overrides = []
    for key in structural_keys:
        if key in model:
            value = json.dumps(model[key], separators=(",", ":"))
            overrides.append(f"model.config.{key}={value}")
    return overrides


def _window_options_from_training_config(checkpoint: str) -> dict:
    saved = yaml.safe_load((_run_dir_from_checkpoint(checkpoint) / "config.yaml").read_text())
    datasets = saved["dataloader_train"]["dataloader"]["datasets"]
    dataset = next(iter(datasets.values()))["dataset"]
    return {
        "robot_window_frames": dataset.get("robot_window_frames", 0),
        "robot_window_stride": dataset.get("robot_window_stride", 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reasoner", "generator"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default=os.environ.get("HUMAN_GEN_ROOT"))
    parser.add_argument("--manifest", default=os.environ.get("ICL_PAIR_MANIFEST"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-ablation", action="store_true")
    parser.add_argument(
        "--robot-window-frames",
        type=int,
        default=None,
        help="Override checkpoint window length; 0 evaluates full pairs",
    )
    parser.add_argument("--robot-window-stride", type=int, default=None)
    args = parser.parse_args()
    if not args.root or not args.manifest:
        parser.error("--root/--manifest or HUMAN_GEN_ROOT/ICL_PAIR_MANIFEST are required")

    experiment_opts = _model_overrides_from_training_config(args.checkpoint, args.mode)
    window_options = _window_options_from_training_config(args.checkpoint)
    for key in window_options:
        value = getattr(args, key)
        if value is not None:
            window_options[key] = value
    experiment_opts.extend(
        [
            f"model.config.tokenizer.vae_path={os.environ['WAN_VAE_PATH']}",
            f"model.config.vlm_config.tokenizer.tokenizer_type={os.environ['COSMOS3_EDGE_PROCESSOR_PATH']}",
        ]
    )

    distributed.init()
    model, config = load_model_from_checkpoint(
        experiment_name=f"human_video_icl_{args.mode}_edge",
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        parallelism_config={
            "data_parallel_shard_degree": torch.distributed.get_world_size(),
            "enable_inference_mode": True,
        },
        compile_config={"enabled": False},
        experiment_opts=experiment_opts,
        keys_to_skip_loading=["net_ema."],
    )
    model.eval()

    dataset = HumanGenPairedDataset(
        root=args.root,
        manifest=args.manifest,
        injection_mode=args.mode,
        split=args.split,
        resolution=config.model.config.resolution,
        max_action_dim=config.model.config.max_action_dim,
        tokenizer_config=config.model.config.vlm_config.tokenizer,
        **window_options,
    )
    output = safe_output(dataset.manifest["root"], args.output)
    output.mkdir(parents=True, exist_ok=True)

    count = min(args.samples, len(dataset))
    if count < 1:
        raise ValueError("--samples must be positive")
    rng = np.random.default_rng(args.seed)
    metrics = []

    for index in range(count):
        start = index * len(dataset) // count
        stop = (index + 1) * len(dataset) // count
        dataset_index = int(rng.integers(start, stop))
        sample = dataset[dataset_index]
        target = sample["action_raw"].numpy().copy()

        torch.manual_seed(args.seed + index)
        with torch.no_grad():
            loss_outputs, validation_loss = model.training_step(
                pack_policy_sample(deepcopy(sample)),
                0,
            )
        action, video = sample_icl_policy(
            model,
            sample,
            seed=args.seed + index,
            num_steps=args.steps,
        )
        robot_frames = int(sample["robot_valid_frames"])
        row = {
            "future_robot_video_mse": float(policy_video_mse(video, sample, robot_frames)),
            "validation_loss": float(validation_loss),
            "validation_action_loss": float(loss_outputs["flow_matching_loss_action"]),
            "validation_vision_loss": float(loss_outputs["flow_matching_loss_vision"]),
            "index": index,
            "dataset_index": dataset_index,
            "pair_id": sample["pair_id"],
            "sample_id": sample["sample_id"],
            "robot_window_start": sample["robot_window_start"],
            "robot_frame_ids": sample["robot_frame_ids"],
            "instruction": sample["ai_caption"],
            "action_mse": float(np.mean((action - target) ** 2)),
            "zero_action_mse": float(np.mean(target**2)),
            "action_max_abs": float(np.abs(action).max()),
            "video_std": float(video.std()),
            "finite": bool(np.isfinite(action).all() and torch.isfinite(video).all()),
        }

        if args.condition_ablation:
            ablated = deepcopy(sample)
            if args.mode == "reasoner":
                ablated["sequence_plan"].reasoner_video_input["pixel_values_videos"].zero_()
            else:
                prefix = int(ablated["human_demo_frames"])
                ablated["video"][:, :prefix].zero_()
            ablated_action, _ = sample_icl_policy(
                model,
                ablated,
                seed=args.seed + index,
                num_steps=args.steps,
            )
            row["human_ablation_action_rms"] = float(np.sqrt(np.mean((action - ablated_action) ** 2)))

        if torch.distributed.get_rank() == 0:
            np.save(output / f"action_{index:03d}.npy", action)
            np.save(output / f"target_{index:03d}.npy", target)
            pixels = ((video[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
            iio.imwrite(
                output / f"robot_{index:03d}.mp4",
                pixels,
                fps=float(sample["conditioning_fps"]),
            )
            print(row, flush=True)
        metrics.append(row)

    if torch.distributed.get_rank() == 0:
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
