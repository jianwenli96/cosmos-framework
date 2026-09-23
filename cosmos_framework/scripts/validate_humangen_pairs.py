# SPDX-License-Identifier: OpenMDW-1.1
"""Decode representative real HumanGen pairs and validate the policy input contract."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.data.generator.action.datasets.humangen_dataset import (
    HumanGenPairedDataset,
    safe_output,
)
from cosmos_framework.inference.icl_policy import policy_batch


def _schema_key(pair: dict) -> tuple:
    provenance = tuple(sorted(pair.get("preprocessing_provenance", {}).items()))
    cameras = tuple(pair.get("robot_camera_keys", ()))
    return pair["source"], provenance, cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--processor", help="Local Cosmos3-Edge processor; required for reasoner mode")
    parser.add_argument("--mode", choices=["generator", "reasoner"], default="generator")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output = safe_output(manifest["root"], args.output)
    tokenizer_config = None
    if args.processor:
        from cosmos_framework.data.generator.processors import build_processor_lazy
        from cosmos_framework.utils.lazy_config import LazyCall as L

        tokenizer_config = L(build_processor_lazy)(tokenizer_type=args.processor)
    if args.mode == "reasoner" and tokenizer_config is None:
        parser.error("--processor is required for reasoner mode")

    rows = []
    for split, pairs in manifest["splits"].items():
        if not pairs:
            continue
        dataset = HumanGenPairedDataset(
            manifest["root"],
            args.manifest,
            args.mode,
            split=split,
            tokenizer_config=tokenizer_config,
        )
        groups = {}
        for index, pair in enumerate(pairs):
            groups.setdefault(_schema_key(pair), []).append(index)

        for indexes in groups.values():
            for index in sorted({indexes[0], indexes[-1]}):
                sample = dataset[index]
                pair = pairs[index]
                assert sample["action_raw"].shape[0] == sample["action_valid_steps"]
                assert sample["robot_valid_frames"] == len(pair["robot_frame_ids"])
                assert torch.isfinite(sample["action"]).all()
                raw_dim = int(sample["raw_action_dim"])
                normalized = sample["action"][:, :raw_dim]
                assert normalized.abs().max() <= 2.0 + 1e-5
                normalizer = sample["action_processing_record"].action_normalizer
                expected = normalizer.normalize_action(sample["action_raw"])
                torch.testing.assert_close(normalized, expected, rtol=2e-4, atol=2e-4)
                # Forward clipping is lossy: only unsaturated channels round-trip.
                restored = normalizer.denormalize_action(normalized)
                unsaturated = normalized.abs() < 2.0
                torch.testing.assert_close(
                    restored[unsaturated], sample["action_raw"][unsaturated], rtol=2e-4, atol=2e-4
                )

                batch = policy_batch(sample)
                assert batch["action"][0][0].count_nonzero() == 0
                assert batch["action_raw"][0][0].count_nonzero() == 0
                prefix = int(sample.get("human_demo_frames", 0))
                assert batch["video"][0][0][:, prefix + 1 :].count_nonzero() == 0

                row = {
                    "split": split,
                    "source": pair["source"],
                    "pair_id": sample["pair_id"],
                    "video_shape": list(sample["video"].shape),
                    "action_shape": list(sample["action_raw"].shape),
                    "normalized_action_abs_max": float(normalized.abs().max()),
                    "robot_cameras": sample["robot_camera_count"],
                    "domain_id": int(sample["domain_id"]),
                    "video_fps": float(sample["conditioning_fps"]),
                    "action_fps": float(sample["conditioning_fps_action"]),
                    "action_steps": sample["action_valid_steps"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mode": args.mode, "samples": rows}, indent=2))


if __name__ == "__main__":
    main()
