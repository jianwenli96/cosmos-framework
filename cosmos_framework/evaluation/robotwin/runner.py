# SPDX-License-Identifier: OpenMDW-1.1
"""RoboTwin environment setup, episode evaluation and metric persistence."""

import importlib
import json
import os
import sys
from pathlib import Path

import yaml

from cosmos_framework.evaluation.robotwin.policy import RobotwinClient, rollout
from cosmos_framework.evaluation.robotwin.simulation import EpisodeVideo, episode_instruction, initialize_renderer


class EvaluationConfigError(ValueError):
    """Invalid evaluation arguments or an incompatible policy service."""


def load_task_config(root, task, name, configs_path):
    def read(path):
        return yaml.safe_load(Path(path).read_text())

    args = read(root / "task_config" / f"{name}.yml")
    embodiments = read(Path(configs_path) / "_embodiment_config.yml")
    cameras = read(Path(configs_path) / "_camera_config.yml")
    types = args["embodiment"]
    if len(types) not in (1, 3):
        raise ValueError("RoboTwin embodiment must have one or three entries")
    left = embodiments[types[0]]["file_path"]
    right = embodiments[types[0] if len(types) == 1 else types[1]]["file_path"]
    camera = cameras[args["camera"]["head_camera_type"]]
    args.update(
        task_name=task,
        task_config=name,
        eval_mode=True,
        eval_video_log=False,
        head_camera_h=camera["h"],
        head_camera_w=camera["w"],
        left_robot_file=left,
        right_robot_file=right,
        dual_arm_embodied=len(types) == 1,
        left_embodiment_config=read(Path(left) / "config.yml"),
        right_embodiment_config=read(Path(right) / "config.yml"),
    )
    if len(types) == 3:
        args["embodiment_dis"] = types[2]
    return args


def run_evaluation(args):
    """Evaluate the requested episodes using the parsed CLI configuration."""
    if args.episode_offset < 0 or args.video_fps <= 0 or args.video_stride < 1:
        raise EvaluationConfigError("Invalid episode offset or video settings")
    if min(args.episodes, args.execute_steps, args.max_attempts) < 1:
        raise EvaluationConfigError("episodes, execute-steps and max-attempts must be positive")
    root = Path(args.robotwin_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = RobotwinClient(args.server, args.timeout)
    info = client.request("/info")
    if not info.get("supports_episode_instruction"):
        raise EvaluationConfigError("Server must support episode instructions; upgrade the policy server")
    if info["task"] != args.task:
        raise EvaluationConfigError("Requested task does not match the server demonstration task")
    if args.execute_steps > info["action_horizon"]:
        raise EvaluationConfigError("execute-steps exceeds server action_horizon")
    # RoboTwin loads assets and task configs relative to its own checkout.
    sys.path.insert(0, str(root))
    os.chdir(root)
    from description.utils.generate_episode_instructions import generate_episode_descriptions
    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError

    renderer_context = None if args.skip_render_check else initialize_renderer()
    config = load_task_config(root, args.task, args.task_config, CONFIGS_PATH)
    config.update(policy_name="cosmos_icl", ckpt_setting="0", save_root=str(output))
    if args.save_video:
        (output / "videos").mkdir(exist_ok=True)
    env = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
    env.suc = 0
    env.test_num = 0
    initial_seed = args.start_seed if args.start_seed is not None else 100000 * (1 + args.seed)
    rows, rejected = [], []

    def save():
        result = dict(
            task=args.task,
            task_config=args.task_config,
            policy=info,
            seed=args.seed,
            start_seed=initial_seed,
            execute_steps=args.execute_steps,
            instruction_type=args.instruction_type,
            episode_offset=args.episode_offset,
            renderer="robotwin_default" if args.skip_render_check else "zerowam_rt",
            episodes=rows,
            rejected=rejected,
            successes=sum(row["success"] for row in rows),
            completed=len(rows),
            success_rate=sum(row["success"] for row in rows) / len(rows) if rows else None,
        )
        temp = output / "metrics.json.tmp"
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(output / "metrics.json")

    for attempt in range(args.max_attempts):
        if len(rows) >= args.episodes:
            break
        seed = initial_seed + attempt
        episode = args.episode_offset + len(rows)
        # Check solvability, then reset exactly the same seed for the learned policy.
        try:
            env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **dict(config, render_freq=0))
            episode_info = env.play_once()
            valid = bool(env.plan_success and env.check_success())
        except UnStableError:
            valid = False
        finally:
            env.close_env()
        if not valid:
            rejected.append(seed)
            save()
            continue
        video = None
        try:
            env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **config)
            instruction = episode_instruction(
                generate_episode_descriptions,
                args.task,
                episode_info,
                args.episodes,
                args.seed + seed,
                args.instruction_type,
            )
            env.set_instruction(instruction=instruction)
            client.instruction = instruction
            if args.save_video:
                video = EpisodeVideo(output / "videos" / f"episode_{episode:04d}_seed_{seed}.mp4", args.video_fps)
            row = rollout(
                env,
                client,
                args.execute_steps,
                args.seed + seed,
                on_observation=video.append if video else None,
                video_stride=args.video_stride,
            )
            rows.append(dict(episode=episode, seed=seed, instruction=instruction, **row))
            env.suc += int(row["success"])
            env.test_num += 1
        finally:
            try:
                if video is not None:
                    video.close()
            finally:
                try:
                    env.close_env(clear_cache=(episode + 1) % max(1, config.get("clear_cache_freq", 1)) == 0)
                finally:
                    save()
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) < args.episodes:
        raise RuntimeError(f"Only {len(rows)}/{args.episodes} valid episodes after {args.max_attempts} seeds")
