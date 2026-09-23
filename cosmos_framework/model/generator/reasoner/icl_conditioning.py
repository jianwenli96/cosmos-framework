# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen Cosmos3-Edge visual conditioning for packed Reasoner ICL."""

import os
from functools import lru_cache
from types import SimpleNamespace

import torch


@lru_cache(maxsize=4)
def frozen_edge_vision(checkpoint_dir: str, device: str):
    """Load the checkpoint-local Edge vision tower once per device."""
    from cosmos_framework.model.generator.mot.unified_mot import Nemotron3DenseVLTextForCausalLM

    holder = SimpleNamespace(
        _local_checkpoint_dir=checkpoint_dir,
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.empty(0, device=device, dtype=torch.bfloat16))),
        config=SimpleNamespace(),
    )
    Nemotron3DenseVLTextForCausalLM._ensure_vision_tower(holder)
    holder.visual.requires_grad_(False).eval()
    return holder


def _video_features_and_positions(holder, media, input_ids, dtype):
    """Encode one video prompt using the same primitives as Edge reasoner prefill."""
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import get_rope_index

    with torch.no_grad():
        grid = media["video_grid_thw"].to(input_ids.device)
        features = holder.visual.get_image_features(
            media["pixel_values_videos"].to(input_ids.device),
            grid,
        ).to(dtype)

        # Packed MoT text adds one generator delimiter after the reasoner prompt.
        # Compute the reasoner mRoPE grid on the prompt, then place that delimiter
        # immediately after the largest multimodal position.
        positions, _ = get_rope_index(
            holder,
            input_ids=input_ids[:, :-1].cpu(),
            video_grid_thw=grid.cpu(),
        )
        boundary = positions.new_full((3, 1, 1), int(positions.max()) + 1)
        positions = torch.cat([positions, boundary], dim=-1).to(input_ids.device)
    return features, positions


def inject_reasoner_video(packed_seq, text_embeddings):
    """Replace packed video placeholders with frozen Edge visual features."""
    media_inputs = packed_seq.reasoner_video_inputs
    if media_inputs is None:
        return text_embeddings

    checkpoint_dir = os.environ.get("COSMOS3_EDGE_PROCESSOR_PATH")
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        raise ValueError("Reasoner ICL requires local COSMOS3_EDGE_PROCESSOR_PATH including vision_encoder/.")
    holder = frozen_edge_vision(checkpoint_dir, str(text_embeddings.device))
    causal_lengths = [
        length for length, mode in zip(packed_seq.split_lens, packed_seq.attn_modes, strict=True) if mode == "causal"
    ]
    if len(media_inputs) != len(causal_lengths):
        raise ValueError("Reasoner media must align 1:1 with packed samples")

    output = text_embeddings.clone()
    text_offset = 0
    for media, length in zip(media_inputs, causal_lengths, strict=True):
        input_ids = packed_seq.text_ids[text_offset : text_offset + length].unsqueeze(0)
        if media is not None:
            features, positions = _video_features_and_positions(
                holder,
                media,
                input_ids,
                output.dtype,
            )
            mask = input_ids[0] == holder.config.video_token_id
            if int(mask.sum()) != features.shape[0]:
                raise ValueError("Video placeholder count does not match Edge visual features")
            output[text_offset : text_offset + length][mask] = features
            text_indexes = packed_seq.text_indexes[text_offset : text_offset + length]
            packed_seq.position_ids[:, text_indexes] = positions[:, 0].to(packed_seq.position_ids.dtype)
        text_offset += length
    return output
