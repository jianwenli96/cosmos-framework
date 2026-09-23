# SPDX-License-Identifier: OpenMDW-1.1
"""Route a HumanGen demonstration into the selected ICL pathway."""

import torch


def inject_human_video(
    robot: dict,
    human: torch.Tensor,
    injection_mode: str,
    processor,
    fps: float,
    video_metadata: dict | None = None,
) -> dict:
    """Inject one human demonstration without changing robot supervision."""
    if injection_mode not in {"reasoner", "generator"}:
        raise ValueError(f"Unknown ICL mode: {injection_mode}")
    if human.ndim != 4 or human.shape[0] != 3 or human.shape[1] < 1:
        raise ValueError("Human clip must be nonempty RGB CTHW")

    plan = robot["sequence_plan"]
    num_frames = int(human.shape[1])
    if injection_mode == "generator":
        if num_frames < 5 or (num_frames - 1) % 4:
            raise ValueError("Generator demonstration must contain 4k+1 frames")

        # The causal VAE must never encode across the human/robot boundary.
        from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target

        robot_frames = int(robot["video"].shape[1])
        height, width = robot["video"].shape[-2:]
        human = reflection_pad_to_target({"video": human}, ["video"], True, width, height)["video"]
        robot["video"] = torch.cat([human, robot["video"]], dim=1)
        robot["video_segment_frames"] = [num_frames, robot_frames]
        robot["human_demo_frames"] = num_frames

        human_latent_frames = (num_frames - 1) // 4 + 1
        plan.condition_frame_indexes_vision = list(range(human_latent_frames + 1))
        return robot

    if processor is None:
        raise ValueError("Reasoner ICL requires the Cosmos3-Edge multimodal processor")

    frames = human.permute(1, 0, 2, 3)
    metadata = video_metadata or {
        "fps": fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": robot["ai_caption"]},
            ],
        }
    ]
    edge_processor = processor.processor
    text = edge_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    processed = edge_processor(
        text=[text],
        videos=[frames],
        return_tensors="pt",
        videos_kwargs={
            "do_sample_frames": False,
            "size": {"shortest_edge": 65536, "longest_edge": 65536},
            "video_metadata": [metadata],
        },
    )
    robot["text_token_ids"] = processed["input_ids"][0]
    plan.reasoner_video_input = {key: processed[key] for key in ("pixel_values_videos", "video_grid_thw")}
    return robot
