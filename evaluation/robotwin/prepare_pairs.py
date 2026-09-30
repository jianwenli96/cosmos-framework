#!/usr/bin/env python
# SPDX-License-Identifier: OpenMDW-1.1
"""Resolve the seven Zero-WAM held-out tasks to eligible HumanGen test pairs."""

import argparse
import json
from pathlib import Path

import yaml

# Preferred human samples from Zero-WAM; tasks.txt is the sole ordering authority.
TASKS = Path(__file__).with_name("tasks.txt").read_text().splitlines()
TASK_SAMPLES = {
    "stack_blocks_three": "519_robotwin_",
    "place_object_scale": "086_robotwin_",
    "stamp_seal": "206_robotwin_",
    "open_microwave": "242_robotwin_",
    "move_stapler_pad": "074_robotwin_",
    "place_bread_basket": "007_robotwin_",
    "place_empty_cup": "273_robotwin_",
}


def select_pairs(manifest, window_frames, overrides):
    selected = []
    unknown = overrides.keys() - TASK_SAMPLES.keys()
    if unknown:
        raise ValueError(f"Unknown tasks in pair map: {sorted(unknown)}")
    for task in TASKS:
        preferred = TASK_SAMPLES[task]
        candidates = []
        for pair in manifest["splits"]["test"]:
            repo = Path(pair["parquet"]).parents[2].name
            if (
                pair["source"] == "robotwin"
                and (repo == task or repo.startswith(task + "-"))
                and len(pair["robot_frame_ids"]) >= window_frames
            ):
                candidates.append(pair)
        if task in overrides:
            candidates = [pair for pair in candidates if pair["pair_id"] == overrides[task]]
        if not candidates:
            raise ValueError(f"No eligible test pair for {task}; check manifest, window size and PAIR_MAP_JSON")
        candidates.sort(
            key=lambda pair: (
                not any(part.startswith(preferred) for part in Path(pair["human_video"]).parts),
                pair["pair_id"],
            )
        )
        pair = candidates[0]
        if any(char in pair["pair_id"] for char in "\t\r\n"):
            raise ValueError("Pair IDs cannot contain tabs/newlines")
        selected.append(
            dict(task=task, pair_id=pair["pair_id"], human_video=pair["human_video"], caption=pair["caption"])
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-map")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.name != "model" or not checkpoint.parent.name.startswith("iter_"):
        parser.error("Expected checkpoints/iter_XXXXXXXXX/model")
    config = yaml.safe_load((checkpoint.parents[2] / "config.yaml").read_text())
    datasets = config["dataloader_train"]["dataloader"]["datasets"]
    if len(datasets) != 1:
        parser.error("Expected exactly one HumanGen training dataset")
    window = next(iter(datasets.values()))["dataset"].get("robot_window_frames", 0)
    if window < 5 or (window - 1) % 4:
        parser.error("Expected a window-trained checkpoint with robot_window_frames=4k+1 >=5")
    manifest = json.loads(Path(args.manifest).read_text())
    overrides = json.loads(Path(args.pair_map).read_text()) if args.pair_map else {}
    rows = select_pairs(manifest, window, overrides)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (output / "pairs.tsv").write_text("".join(f"{row['task']}\t{row['pair_id']}\n" for row in rows))


if __name__ == "__main__":
    main()
