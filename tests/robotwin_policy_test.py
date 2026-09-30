# SPDX-License-Identifier: OpenMDW-1.1
"""RoboTwin action convention and receding-horizon regression tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cosmos_framework.evaluation.robotwin.policy import absolute_actions, decode_image, encode_image, rollout


def test_window_relative_pose_roundtrip():
    from cosmos_framework.data.generator.action.datasets.humangen_preprocessing import robotwin_relative_action

    rng = np.random.default_rng(4)
    action = rng.normal(size=(7, 16))
    reference = rng.normal(size=(1, 16))
    for offset in (0, 8):
        action[:, offset + 3 : offset + 7] = Rotation.random(7, random_state=rng).as_quat()
        reference[:, offset + 3 : offset + 7] = Rotation.random(1, random_state=rng).as_quat()
    restored = absolute_actions(robotwin_relative_action(action, reference), reference[0])
    for offset in (0, 8):
        np.testing.assert_allclose(restored[:, offset : offset + 3], action[:, offset : offset + 3])
        np.testing.assert_allclose(restored[:, offset + 7], action[:, offset + 7])
        np.testing.assert_allclose(
            Rotation.from_quat(restored[:, offset + 3 : offset + 7]).as_matrix(),
            Rotation.from_quat(action[:, offset + 3 : offset + 7]).as_matrix(),
            atol=1e-12,
        )


def test_invalid_quaternion_rejected():
    with pytest.raises(ValueError, match="zero norm"):
        absolute_actions(np.zeros((2, 16)), np.zeros(16))


def test_rgb_transport():
    image = np.random.default_rng(1).integers(0, 256, (12, 17, 3), dtype=np.uint8)
    np.testing.assert_array_equal(decode_image(encode_image(image)), image)


@pytest.mark.parametrize("success_at,expected", [(5, 5), (100, 7)])
def test_rollout_reobserves_and_stops(success_at, expected):
    class Env:
        take_action_cnt = 0
        step_lim = 7
        eval_success = False

        def get_obs(self):
            return self.take_action_cnt

        def take_action(self, action, action_type):
            assert action_type == "ee"
            self.take_action_cnt += 1
            self.eval_success = self.take_action_cnt == success_at

    class Client:
        observations = []

        def predict(self, obs, seed):
            self.observations.append(obs)
            return np.zeros((8, 16))

    client = Client()
    recorded = []
    result = rollout(Env(), client, 3, 42, on_observation=recorded.append, video_stride=2)
    assert recorded == ([0, 2, 4, 5] if success_at == 5 else [0, 2, 4, 6, 7])
    assert result["steps"] == expected
    assert result["success"] == (success_at == 5)
    assert client.observations == ([0, 3] if success_at == 5 else [0, 3, 6])


@pytest.mark.parametrize("prefix", [0, 5])
def test_live_condition_preserves_demo_and_removes_targets(prefix):
    import torch

    from cosmos_framework.evaluation.robotwin.policy import CAMERAS, live_sample

    template = {
        "video": torch.full((3, prefix + 5, 4, 18), 99, dtype=torch.uint8),
        "human_demo_frames": prefix,
        "action": torch.ones(8, 64),
        "action_raw": torch.ones(8, 16),
    }
    images = {key: encode_image(np.full((4, 6, 3), i + 1, dtype=np.uint8)) for i, key in enumerate(CAMERAS)}

    def resize(data, resolution):
        return dict(data, image_size=torch.tensor([4, 18, 4, 18]))

    sample = live_sample(template, images, resize, "256")
    assert torch.all(sample["video"][:, :prefix] == 99)
    for i in range(3):
        assert torch.all(sample["video"][:, prefix, :, i * 6 : (i + 1) * 6] == i + 1)
    assert not sample["video"][:, prefix + 1 :].any()
    assert not sample["action"].any()
    assert not sample["action_raw"].any()
    assert torch.all(template["video"] == 99)
    assert torch.all(template["action"] == 1)


def test_demo_cannot_be_relabelled_as_another_task():
    from cosmos_framework.evaluation.robotwin.policy import validate_demo_task

    pair = {"source": "robotwin", "parquet": "robotwin_data/stamp_seal-demo_clean/data/chunk-000/episode.parquet"}
    validate_demo_task(pair, "stamp_seal")
    with pytest.raises(ValueError, match="does not match"):
        validate_demo_task(pair, "open_microwave")


def test_policy_http_bypasses_proxy_and_reports_server_error(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cosmos_framework.evaluation.robotwin.policy import RobotwinClient

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "invalid predicted quaternion"}')

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RobotwinClient(f"http://127.0.0.1:{server.server_port}")
            with pytest.raises(RuntimeError, match="invalid predicted quaternion"):
                client.request("/info")
        finally:
            server.shutdown()
            thread.join()


def test_episode_instruction_uses_expert_scene_and_is_reproducible():
    from cosmos_framework.evaluation.robotwin.simulation import episode_instruction

    def generate(task, infos, episodes):
        assert task == "stamp_seal" and episodes == 100
        assert infos == [{"target_color": "red"}]
        return [{"seen": ["Stamp the red pad", "Use the red pad"]}]

    result = episode_instruction(generate, "stamp_seal", {"info": {"target_color": "red"}}, 100, 42)
    assert result == episode_instruction(generate, "stamp_seal", {"info": {"target_color": "red"}}, 100, 42)
    assert "red" in result


def test_renderer_matches_zero_wam_configuration(monkeypatch):
    import sys
    from types import SimpleNamespace

    from cosmos_framework.evaluation.robotwin.simulation import initialize_renderer

    calls = {}

    class Engine:
        def set_renderer(self, renderer):
            calls["renderer"] = renderer

        def create_scene(self, config):
            return "scene"

    render = SimpleNamespace(
        **{
            name: (lambda *args, _name=name, **kwargs: calls.update({_name: (args, kwargs)}))
            for name in (
                "set_global_config",
                "set_camera_shader_dir",
                "set_ray_tracing_samples_per_pixel",
                "set_ray_tracing_path_depth",
                "set_ray_tracing_denoiser",
            )
        }
    )
    core = SimpleNamespace(Engine=Engine, SapienRenderer=lambda: "renderer", SceneConfig=lambda: None)
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace(core=core, render=render))
    monkeypatch.setitem(sys.modules, "sapien.core", core)
    result = initialize_renderer()
    assert result[2] == "scene"
    assert calls["set_global_config"][1] == {"max_num_materials": 50000, "max_num_textures": 50000}
    assert calls["set_camera_shader_dir"][0] == ("rt",)
    assert calls["set_ray_tracing_samples_per_pixel"][0] == (32,)
    assert calls["set_ray_tracing_path_depth"][0] == (8,)
    assert calls["set_ray_tracing_denoiser"][0] == ("oidn",)


def test_client_sends_current_episode_instruction():
    from cosmos_framework.evaluation.robotwin.policy import RobotwinClient

    client = RobotwinClient("http://unused")
    requests = []
    action = np.zeros((1, 16))
    action[0, [6, 14]] = 1

    def request(path, payload):
        requests.append(payload)
        return {"action": action.tolist()}

    client.request = request
    observation = {
        "observation": {
            key: {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)} for key in ("head_camera", "left_camera", "right_camera")
        },
        "endpose": {
            "left_endpose": [0, 0, 0, 0, 0, 0, 1],
            "left_gripper": 0,
            "right_endpose": [0, 0, 0, 0, 0, 0, 1],
            "right_gripper": 0,
        },
    }
    for instruction in ("red pad", "blue pad"):
        client.instruction = instruction
        client.predict(observation, 42)
        assert requests[-1]["instruction"] == instruction


def test_client_episode_passes_scene_instruction_and_runtime_config(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from cosmos_framework.evaluation.robotwin import runner
    from cosmos_framework.scripts import eval_robotwin

    class Env:
        take_action_cnt = 0
        step_lim = 3
        eval_success = False
        plan_success = True

        def setup_demo(self, **config):
            assert config["policy_name"] == "cosmos_icl"
            assert config["save_root"] == str(tmp_path / "output")
            self.take_action_cnt = 0
            self.eval_success = False

        def play_once(self):
            return {"info": {"color": "green"}}

        def check_success(self):
            return True

        def close_env(self, **kwargs):
            pass

        def set_instruction(self, instruction):
            assert instruction == "Stamp green"

        def get_obs(self):
            return {}

        def take_action(self, action, action_type):
            self.take_action_cnt += 1
            self.eval_success = True

    class Client:
        def __init__(self, *args):
            pass

        def request(self, path):
            return {"task": "stamp_seal", "supports_episode_instruction": True, "action_horizon": 16}

        def predict(self, observation, seed):
            assert self.instruction == "Stamp green"
            return np.zeros((16, 16))

    modules = {}
    for name in (
        "envs",
        "envs.utils",
        "envs.utils.create_actor",
        "envs.stamp_seal",
        "description",
        "description.utils",
        "description.utils.generate_episode_instructions",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["envs"].CONFIGS_PATH = str(tmp_path)
    modules["envs.utils.create_actor"].UnStableError = type("UnStableError", (Exception,), {})
    modules["envs.stamp_seal"].stamp_seal = Env

    def generate(task, infos, count):
        assert infos == [{"color": "green"}]
        return [{"seen": ["Stamp green"]}]

    modules["description.utils.generate_episode_instructions"].generate_episode_descriptions = generate
    monkeypatch.setattr(runner, "RobotwinClient", Client)
    monkeypatch.setattr(runner, "load_task_config", lambda *args: {"render_freq": 0})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--robotwin-root",
            str(tmp_path),
            "--task",
            "stamp_seal",
            "--output",
            str(tmp_path / "output"),
            "--episodes",
            "1",
            "--skip-render-check",
            "--no-save-video",
        ],
    )
    eval_robotwin.main()
    import json

    result = json.loads((tmp_path / "output/metrics.json").read_text())
    assert result["episodes"][0]["instruction"] == "Stamp green"
    assert result["successes"] == 1
