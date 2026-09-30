# SPDX-License-Identifier: OpenMDW-1.1
"""Optional RoboTwin runtime helpers; no model/torch dependencies."""

import numpy as np
from PIL import Image, ImageOps

from cosmos_framework.evaluation.robotwin.policy import CAMERAS


def initialize_renderer():
    """Match Zero-WAM's Sapien_TEST settings and fail before episode execution."""
    import sapien.core as sapien
    from sapien import render

    render.set_global_config(max_num_materials=50000, max_num_textures=50000)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    render.set_camera_shader_dir("rt")
    render.set_ray_tracing_samples_per_pixel(32)
    render.set_ray_tracing_path_depth(8)
    render.set_ray_tracing_denoiser("oidn")
    scene = engine.create_scene(sapien.SceneConfig())
    # Keep the initialization objects alive for the evaluation lifetime.
    return engine, renderer, scene


def episode_instruction(generate, task, episode_info, episodes, seed, instruction_type="seen"):
    """Generate language from this expert scene, with a separate reproducible RNG."""
    descriptions = generate(task, [episode_info["info"]], episodes)
    choices = descriptions[0][instruction_type]
    if (
        not isinstance(choices, (list, tuple))
        or not choices
        or not all(isinstance(text, str) and text.strip() for text in choices)
    ):
        raise ValueError(f"No valid {instruction_type} instructions for {task}")
    return choices[int(np.random.default_rng(seed).integers(len(choices)))]


class EpisodeVideo:
    """Stream actual three-camera observations without buffering entire episodes."""

    def __init__(self, path, fps):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(str(path), fps=fps)
        self.tile_size = None

    def append(self, observation):
        frames = [
            Image.fromarray(np.asarray(observation["observation"][key]["rgb"], dtype=np.uint8)) for key in CAMERAS
        ]
        if self.tile_size is None:
            self.tile_size = frames[0].size
        tiles = [np.asarray(ImageOps.pad(frame, self.tile_size)) for frame in frames]
        self.writer.append_data(np.concatenate(tiles, axis=1))

    def close(self):
        self.writer.close()
