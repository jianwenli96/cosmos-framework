# SPDX-License-Identifier: OpenMDW-1.1
"""Focused invariants for HumanGen ICL integration."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
    pack_input_sequence,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def test_generator_injection_conditions_demo_and_robot_observation():
    from cosmos_framework.data.generator.action.icl_transforms import inject_human_video

    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0],
        action_start_frame_offset=1,
    )
    robot = {
        "video": torch.zeros(3, 9, 8, 8, dtype=torch.uint8),
        "sequence_plan": plan,
    }
    human = torch.ones(3, 13, 8, 8, dtype=torch.uint8)

    result = inject_human_video(robot, human, "generator", None, 20.0)

    assert result["video"].shape[1] == 22
    assert result["video_segment_frames"] == [13, 9]
    assert result["human_demo_frames"] == 13
    assert result["video"][:, :13].all()
    assert result["video"][:, 13:].count_nonzero() == 0
    assert plan.condition_frame_indexes_vision == [0, 1, 2, 3, 4]
    assert plan.reasoner_video_input is None


def test_policy_batch_hides_future_targets_without_mutating_sample():
    from cosmos_framework.inference.icl_policy import policy_batch

    video = torch.arange(3 * 22 * 4 * 4).reshape(3, 22, 4, 4)
    sample = {
        "video": video.clone(),
        "human_demo_frames": 13,
        "action": torch.ones(8, 64),
        "action_raw": torch.ones(8, 7),
        "text_token_ids": torch.tensor([3, 4]),
    }
    result = policy_batch(sample)

    assert torch.equal(result["video"][0][0][:, :14], video[:, :14])
    assert result["video"][0][0][:, 14:].count_nonzero() == 0
    assert result["action"][0][0].count_nonzero() == 0
    assert result["action_raw"][0][0].count_nonzero() == 0
    assert torch.equal(sample["video"], video)
    assert sample["action"].sum() == 8 * 64


def test_reasoner_video_replaces_only_video_placeholders(monkeypatch, tmp_path):
    from cosmos_framework.model.generator.reasoner import icl_conditioning as icl

    holder = SimpleNamespace(config=SimpleNamespace(video_token_id=9))
    monkeypatch.setenv("COSMOS3_EDGE_PROCESSOR_PATH", str(tmp_path))
    monkeypatch.setattr(icl, "frozen_edge_vision", lambda *args: holder)

    def features_and_positions(holder, media, ids, dtype):
        value = float(media["pixel_values_videos"][0])
        features = torch.full((1, 4), value, dtype=dtype)
        positions = torch.zeros(3, 1, ids.shape[1], dtype=torch.long)
        return features, positions

    monkeypatch.setattr(icl, "_video_features_and_positions", features_and_positions)

    packed = PackedSequence(
        sequence_length=8,
        sample_lens=[4, 4],
        split_lens=[3, 1, 3, 1],
        attn_modes=["causal", "full", "causal", "full"],
        text_ids=torch.tensor([1, 9, 2, 3, 9, 4]),
        text_indexes=torch.tensor([0, 1, 2, 4, 5, 6]),
        position_ids=torch.ones(3, 8),
        reasoner_video_inputs=[
            {"pixel_values_videos": torch.tensor([7.0]), "video_grid_thw": torch.tensor([[1, 2, 2]])},
            {"pixel_values_videos": torch.tensor([8.0]), "video_grid_thw": torch.tensor([[1, 2, 2]])},
        ],
    )
    embeddings = torch.ones(6, 4, requires_grad=True)
    output = icl.inject_reasoner_video(packed, embeddings)

    assert torch.equal(output[:, 0], torch.tensor([1.0, 7.0, 1.0, 1.0, 8.0, 1.0]))
    output.sum().backward()
    assert embeddings.grad[1].count_nonzero() == 0
    assert embeddings.grad[4].count_nonzero() == 0
    assert embeddings.grad[0].sum() == 4
    assert torch.all(packed.position_ids[:, [3, 7]] == 1)


@pytest.mark.parametrize("mixed", [False, True])
def test_fp32_master_policy_reaches_each_fsdp_decoder(monkeypatch, mixed):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from cosmos_framework.model.generator.mot import parallelize_unified_mot as parallel

    policies = []
    monkeypatch.setattr(
        parallel,
        "fully_shard",
        lambda block, **kwargs: policies.append(kwargs.get("mp_policy")),
    )
    monkeypatch.setattr(parallel, "register_fsdp_forward_method", lambda *args: None)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]))
    )
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if mixed else None
    parallel.apply_fsdp(model, SimpleNamespace(dp_mesh=None), policy)
    assert policies == [policy, policy]


def test_generator_segments_are_vae_encoded_independently():
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    calls = []

    def encode(value):
        calls.append(value.shape[-3])
        return value[:, :, ::4]

    model = SimpleNamespace(encode=encode)
    video = torch.zeros(1, 3, 22, 8, 8)
    latent = OmniMoTModel._encode_vision_item(
        model,
        video,
        num_views=2,
        frames_per_view=[13, 9],
    )

    assert calls == [13, 9]
    assert latent.shape[-3] == 7


def test_sample_temporal_positions_pack_with_actions():
    plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=[0],
        action_start_frame_offset=2.5,
    )
    temporal_positions = torch.tensor([0.0, 1.5, 3.0])
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 48, 3, 4, 4)],
        temporal_positions_vision=[temporal_positions],
        x0_tokens_action=[torch.randn(8, 64)],
        action_domain_id=[torch.tensor(0)],
        fps_vision=torch.tensor([20.0]),
        fps_action=torch.tensor([20.0]),
    )
    packed = pack_input_sequence(
        [plan],
        [[1, 2]],
        data,
        torch.tensor([500.0]),
        special_tokens={
            "eos_token_id": 3,
            "start_of_generation": 4,
            "end_of_generation": 5,
        },
        latent_patch_size=2,
        enable_fps_modulation=True,
    )

    assert packed.vision is not None
    assert packed.action is not None
    assert packed.action.noisy_frame_indexes[0].tolist() == list(range(8))


def test_action_only_sampling_does_not_decode():
    from cosmos_framework.inference.icl_policy import sample_icl_policy

    def decode(_):
        raise AssertionError("Action-only sampling must not decode video")

    model = SimpleNamespace(
        generate_samples_from_batch=lambda *args, **kwargs: {"action": [torch.zeros(8, 7)]},
        decode=decode,
    )
    sample = {
        "video": torch.zeros(3, 9, 4, 4),
        "action": torch.zeros(8, 64),
        "action_raw": torch.zeros(8, 7),
    }
    action, video = sample_icl_policy(model, sample, decode_video=False)
    assert action.shape == (8, 7)
    assert video is None


def test_inference_recovers_structural_model_options_from_trainer_config(tmp_path):
    import yaml

    from cosmos_framework.scripts.infer_icl_policy import (
        _model_overrides_from_training_config,
        _window_options_from_training_config,
    )

    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "iter_000000002" / "model"
    checkpoint.mkdir(parents=True)
    config = {
        "dataloader_train": {"dataloader": {"datasets": {"humangen": {"dataset": {"injection_mode": "generator"}}}}},
        "model": {
            "config": {
                "fsdp_mixed_precision": True,
                "precision": "float32",
                "resolution": "480",
                "max_action_dim": 32,
            }
        },
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))

    assert _window_options_from_training_config(str(checkpoint)) == {
        "robot_window_frames": 0,
        "robot_window_stride": 1,
    }
    config["dataloader_train"]["dataloader"]["datasets"]["humangen"]["dataset"].update(
        robot_window_frames=17,
        robot_window_stride=2,
    )
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    assert _window_options_from_training_config(str(checkpoint)) == {
        "robot_window_frames": 17,
        "robot_window_stride": 2,
    }
    overrides = _model_overrides_from_training_config(str(checkpoint), "generator")
    assert "model.config.fsdp_mixed_precision=true" in overrides
    assert 'model.config.precision="float32"' in overrides
    assert 'model.config.resolution="480"' in overrides
    assert "model.config.max_action_dim=32" in overrides

    with pytest.raises(ValueError, match="expected"):
        _model_overrides_from_training_config(str(checkpoint), "reasoner")
