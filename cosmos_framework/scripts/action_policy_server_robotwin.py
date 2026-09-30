# SPDX-License-Identifier: OpenMDW-1.1
"""Serve a window-trained HumanGen ICL checkpoint for RoboTwin over HTTP."""

import argparse
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

if os.environ.get("COSMOS_DEVICE", "cuda") == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

from cosmos_framework.data.generator.action.datasets.humangen_dataset import HumanGenPairedDataset
from cosmos_framework.evaluation.robotwin.policy import CAMERAS, instruction_template, live_sample, validate_demo_task
from cosmos_framework.inference.icl_policy import sample_icl_policy
from cosmos_framework.scripts.infer_icl_policy import (
    _model_overrides_from_training_config,
    _window_options_from_training_config,
)
from cosmos_framework.utils import distributed
from cosmos_framework.utils.generator.model_loader import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("reasoner", "generator"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair-id", required=True, help="Fixed demonstration and task caption from the manifest")
    parser.add_argument("--task", required=True, help="RoboTwin environment task name matching the selected pair")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("HTTP serving currently requires a single model process/device")
    windows = _window_options_from_training_config(args.checkpoint)
    if windows["robot_window_frames"] < 5:
        parser.error("Closed-loop serving requires a window-trained checkpoint")
    overrides = _model_overrides_from_training_config(args.checkpoint, args.mode)
    overrides += [
        f"model.config.tokenizer.vae_path={os.environ['WAN_VAE_PATH']}",
        f"model.config.vlm_config.tokenizer.tokenizer_type={os.environ['COSMOS3_EDGE_PROCESSOR_PATH']}",
    ]
    distributed.init()
    try:
        model, config = load_model_from_checkpoint(
            experiment_name=f"human_video_icl_{args.mode}_edge",
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            parallelism_config={"data_parallel_shard_degree": 1, "enable_inference_mode": True},
            compile_config={"enabled": False},
            experiment_opts=overrides,
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
            **windows,
        )
        matches = [i for i, pair in enumerate(dataset.pairs) if pair["pair_id"] == args.pair_id]
        if len(matches) != 1:
            raise ValueError("pair-id must identify exactly one pair in the requested split")
        pair_index = matches[0]
        pair = dataset.pairs[pair_index]
        validate_demo_task(pair, args.task)
        if pair["source"] != "robotwin" or dataset._window_counts[pair_index] < 1:
            raise ValueError("Selected pair must be RoboTwin and contain a complete window")
        index = int(dataset._window_cumulative[pair_index] - dataset._window_counts[pair_index])

        @lru_cache(maxsize=1)
        def get_template(instruction):
            return instruction_template(dataset, index, pair_index, instruction)

        template = get_template(pair["caption"])
        info = dict(
            task=args.task,
            mode=args.mode,
            checkpoint=args.checkpoint,
            pair_id=args.pair_id,
            instruction=pair["caption"],
            supports_episode_instruction=True,
            action_horizon=int(template["action_valid_steps"]),
            action_space="window_relative_xyz_xyzw_gripper",
            cameras=list(CAMERAS),
            **windows,
        )

        del template  # Retain only the bounded instruction cache, not a second demo tensor.

        class Handler(BaseHTTPRequestHandler):
            def respond(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self.respond(200 if self.path in ("/", "/info") else 404, info)

            def do_POST(self):  # noqa: N802
                if self.path != "/predict":
                    self.respond(404, {"error": "Not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("Invalid request length (maximum 32 MiB)")
                    request = json.loads(self.rfile.read(length))
                    instruction = request.get("instruction")
                    if not isinstance(instruction, str) or not instruction.strip():
                        raise ValueError("Request must contain a nonempty episode instruction")
                    sample = live_sample(
                        get_template(instruction), request["images"], dataset.transform.video_resize, dataset.resolution
                    )
                    action, _ = sample_icl_policy(
                        model,
                        sample,
                        seed=int(request.get("seed", args.seed)),
                        num_steps=args.steps,
                        decode_video=False,
                    )
                    if action.shape[1] != 16 or not np.isfinite(action).all():
                        raise ValueError("Expected finite 16D external-space RoboTwin actions")
                    self.respond(200, {"action": action[: info["action_horizon"]].tolist()})
                except (ValueError, KeyError, TypeError) as error:
                    self.respond(400, {"error": str(error)})
                except Exception as error:
                    import traceback

                    traceback.print_exc()
                    self.respond(500, {"error": str(error)})

        print(json.dumps(info, indent=2), flush=True)
        # Serial requests protect model state and device memory.
        with HTTPServer((args.host, args.port), Handler) as server:
            server.serve_forever()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
