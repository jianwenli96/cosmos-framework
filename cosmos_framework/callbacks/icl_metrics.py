# SPDX-License-Identifier: OpenMDW-1.1
"""Persist globally averaged HumanGen ICL training losses."""

import json
from pathlib import Path

import torch

from cosmos_framework.utils.callback import Callback


class ICLMetrics(Callback):
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        names = ["loss", "vision_loss", "action_loss"]
        values = torch.stack(
            [
                loss.detach().float(),
                output_batch["flow_matching_loss_vision"].detach().float(),
                output_batch["flow_matching_loss_action"].detach().float(),
            ]
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Non-finite ICL loss at iteration {iteration}")

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(zip(names, values.cpu().tolist(), strict=True))
            row["iteration"] = iteration
            row["parameter_dtype"] = str(next(model.net.parameters()).dtype)
            with self.output_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
