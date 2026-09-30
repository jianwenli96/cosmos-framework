# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise seven-task launch, routing, result collection and process cleanup."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_pairs", ROOT / "evaluation/robotwin/prepare_pairs.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def make_manifest():
    return {
        "splits": {
            "test": [
                dict(
                    source="robotwin",
                    parquet=f"{task}/data/chunk-000/file.parquet",
                    robot_frame_ids=list(range(5)),
                    pair_id=f"pair-{task}",
                    caption=task,
                    human_video=f"human/{sample}demo/video.mp4",
                )
                for task, sample in prepare.TASK_SAMPLES.items()
            ]
        }
    }


def test_pair_selection_fails_before_launch_for_missing_task():
    manifest = make_manifest()
    manifest["splits"]["test"].pop()
    with pytest.raises(ValueError, match="place_empty_cup"):
        prepare.select_pairs(manifest, 5, {})


def test_pair_override_is_validated():
    with pytest.raises(ValueError, match="stack_blocks_three"):
        prepare.select_pairs(make_manifest(), 5, {"stack_blocks_three": "wrong-task"})


@pytest.mark.parametrize(
    "devices,fail_task,split",
    [
        ("0", "", False),
        ("0,1,2", "", False),
        ("0,1,2", "stamp_seal", False),
        ("0,1,2,3,4,5,6", "", True),
        ("0,1,2,3,4,5,6", "stamp_seal", True),
    ],
)
def test_seven_task_orchestration(tmp_path, devices, fail_task, split):
    # Stand-in Python executable: keep the real pair resolver, replace only model/simulator.
    fake = tmp_path / "fake_python"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen
if sys.argv[1:2] != ["-m"]:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
args = sys.argv[3:]
def get(key): return args[args.index(key) + 1]
if sys.argv[2].endswith("action_policy_server_robotwin"):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"task": get("--task"), "action_horizon": 16, "supports_episode_instruction": True}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
    HTTPServer(("127.0.0.1", int(get("--port"))), Handler).serve_forever()
else:
    with urlopen(get("--server") + "/info") as response:
        assert json.load(response)["task"] == get("--task")
    if os.environ.get("FAIL_TASK") == get("--task"):
        sys.exit(3)
    output = Path(get("--output"))
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps({"completed": 1, "successes": 1, "success_rate": 1.0}))
"""
    )
    fake.chmod(0o755)
    checkpoint = tmp_path / "run/checkpoints/iter_000000001/model"
    checkpoint.mkdir(parents=True)
    (checkpoint.parents[2] / "config.yaml").write_text(
        "dataloader_train:\n  dataloader:\n    datasets:\n      humangen:\n        dataset:\n          robot_window_frames: 5\n"
    )
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps(make_manifest()))
    # Reserve a contiguous block during selection, then release for the launcher.
    for port in range(24000, 25000, 3):
        sockets = []
        try:
            for offset in range(7):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port + offset))
            break
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    else:
        pytest.fail("No free test ports")
    output = tmp_path / "results"
    env = dict(
        os.environ,
        CHECKPOINT=str(checkpoint),
        HUMAN_GEN_ROOT=str(tmp_path),
        ICL_PAIR_MANIFEST=str(manifest),
        ROBOTWIN_ROOT=str(tmp_path),
        SERVER_PYTHON=str(fake),
        CLIENT_PYTHON=str(fake),
        DEVICES=devices,
        SIM_DEVICES=devices,
        SAVE_ROOT=str(output),
        START_PORT=str(port),
        SERVER_START_TIMEOUT="30",
        TEST_NUM="1",
        HOST="127.0.0.1",
        SERVER_BIND="127.0.0.1",
    )
    env["FAIL_TASK"] = fail_task
    env.pop("PAIR_MAP_JSON", None)
    if split:
        env["LOG_ROOT"] = str(tmp_path / "server_logs")
        server_log = tmp_path / "server_launcher.log"
        with server_log.open("w") as log:
            server = subprocess.Popen(
                ["bash", str(ROOT / "evaluation/robotwin/launch_server_multigpus.sh")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                client_env = dict(env, SERVER_IP="127.0.0.1")
                for key in (
                    "CHECKPOINT",
                    "HUMAN_GEN_ROOT",
                    "ICL_PAIR_MANIFEST",
                    "SERVER_PYTHON",
                    "WAN_VAE_PATH",
                    "COSMOS3_EDGE_PROCESSOR_PATH",
                ):
                    client_env.pop(key, None)
                result = subprocess.run(
                    ["bash", str(ROOT / "evaluation/robotwin/launch_client_multigpus.sh")],
                    env=client_env,
                    timeout=60,
                    capture_output=True,
                    text=True,
                )
                assert server.poll() is None, server_log.read_text()
                # A client finishing/failing must not terminate the remote services.
                for offset in range(7):
                    with socket.socket() as sock:
                        assert sock.connect_ex(("127.0.0.1", port + offset)) == 0
            finally:
                server.terminate()
                server.wait(timeout=20)
    else:
        result = subprocess.run(
            ["bash", str(ROOT / "evaluation/robotwin/run_icl_eval.sh")],
            env=env,
            check=False,
            timeout=60,
            capture_output=True,
            text=True,
        )
    assert result.returncode == (1 if fail_task else 0), result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert [row["task"] for row in summary] == list(prepare.TASK_SAMPLES)
    for row in summary:
        if row["task"] == fail_task:
            assert row["completed"] == 0 and row["success_rate"] is None
        else:
            assert row["completed"] == 1 and row["success_rate"] == 1
    for offset in range(len(devices.split(","))):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port + offset)) != 0
