# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos3-Edge HumanGen ICL policy recipes.

Both modes train the same robot video/action targets. In reasoner mode the
human demonstration enters the causal text/reasoner stream; in generator mode
it enters the generation-stream vision context.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.icl_metrics import ICLMetrics
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.humangen_dataset import get_humangen_training_dataset
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L


def _human_video_icl_recipe(mode: str):
    recipe = copy.deepcopy(action_policy_droid_nano)
    name = f"human_video_icl_{mode}_edge"
    recipe["job"].update(project="cosmos3_action", group="humangen", name=name, wandb_mode="offline")

    model = copy.deepcopy(EDGE_MODEL_CONFIG)
    model["action_gen"] = True
    model["resolution"] = "256"
    model["max_num_tokens_after_packing"] = -1
    model["compile"]["enabled"] = False
    model["ema"]["enabled"] = False
    model["parallelism"]["data_parallel_shard_degree"] = -1
    model["fsdp_mixed_precision"] = True
    model["tokenizer"]["encode_exact_durations"] = None
    model["vlm_config"]["tokenizer"] = L(build_processor_lazy)(tokenizer_type="${oc.env:COSMOS3_EDGE_PROCESSOR_PATH}")
    recipe["model"]["config"] = model

    recipe["defaults"] = [
        {"override /optimizer": "adamw"} if item == {"override /optimizer": "fusedadamw"} else item
        for item in recipe["defaults"]
    ]
    recipe["optimizer"].update(optimizer_type="AdamW", fused=True, lr=1.0e-4)
    recipe["optimizer"]["keys_to_select"].append("k_norm_und_for_gen")
    recipe["optimizer"]["lr_multipliers"] = {
        "action2llm": 5.0,
        "llm2action": 5.0,
        "action_modality_embed": 5.0,
    }

    recipe["checkpoint"].update(keys_to_skip_loading=["net_ema."], save_iter=100, strict_resume=False)
    recipe["scheduler"].update(
        cycle_lengths=[1000],
        warm_up_steps=[10],
        f_max=[1.0],
        f_min=[0.1],
        f_start=[0.1],
    )
    recipe["trainer"].update(max_iter=200, logging_iter=1, grad_accum_iter=1)
    recipe["trainer"]["callbacks"]["compile_tokenizer"] = dict(enabled=False)
    recipe["trainer"]["callbacks"]["icl_metrics"] = L(ICLMetrics)(
        output_path="${oc.env:IMAGINAIRE_OUTPUT_ROOT}/${job.project}/${job.group}/${job.name}/icl_metrics.jsonl"
    )

    loader = recipe["dataloader_train"]
    loader["dataset_name"] = "humangen"
    loader["max_samples_per_batch"] = 1
    loader["dataloader"].update(
        batch_size=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
        in_order=True,
        distributed_shuffle=True,
        shuffle_seed="${trainer.seed}",
    )
    loader["dataloader"]["datasets"] = {
        "humangen": {
            "ratio": 1,
            "dataset": L(get_humangen_training_dataset)(
                root="${oc.env:HUMAN_GEN_ROOT}",
                manifest="${oc.env:ICL_PAIR_MANIFEST}",
                injection_mode=mode,
                split="train",
                max_robot_frames=0,
                max_human_frames=0,
                resolution="${model.config.resolution}",
                max_action_dim="${model.config.max_action_dim}",
                tokenizer_config="${model.config.vlm_config.tokenizer}",
            ),
        }
    }
    return name, recipe


for _mode in ("reasoner", "generator"):
    _name, _recipe = _human_video_icl_recipe(_mode)
    ConfigStore.instance().store(group="experiment", package="_global_", name=_name, node=_recipe)
