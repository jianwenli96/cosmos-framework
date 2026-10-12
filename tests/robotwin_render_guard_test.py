# SPDX-License-Identifier: OpenMDW-1.1
"""CPU regression tests for shader selection and version-safe patching."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.evaluation.robotwin import render_guard as guard


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    source = tmp_path / "sapien/vulkan_shader/rt"
    source.mkdir(parents=True)
    original = b"void main() {\n      traceRayEXT(tlas, rest);\n}\n"
    (source / "camera.rgen").write_bytes(original)
    (source / "camera.rchit").write_text("unchanged hit shader")
    calls = []
    render = SimpleNamespace(set_camera_shader_dir=calls.append)
    core = SimpleNamespace(render=render)
    monkeypatch.setitem(
        sys.modules, "sapien", SimpleNamespace(__file__=str(source.parents[1] / "__init__.py"), core=core)
    )
    monkeypatch.setitem(sys.modules, "sapien.core", core)
    monkeypatch.setattr(guard, "RENDER_RUNTIME", {})
    monkeypatch.setenv("COSMOS_RT_RAY_GUARD", "auto")
    # Synthetic shaders exercise control flow; production hashes remain pinned.
    monkeypatch.setattr(guard, "SOURCE_SHA256", hashlib.sha256(original).hexdigest())
    expected = original.replace(
        b"      traceRayEXT(tlas,",
        b"      // An absorbing material has no next ray. Do not trace a zero/invalid direction.\n"
        b"      if (dot(ray.direction, ray.direction) == 0.0 ||\n"
        b"          any(isnan(ray.direction)) || any(isinf(ray.direction))) {\n"
        b"        break;\n"
        b"      }\n\n"
        b"      traceRayEXT(tlas,",
    )
    monkeypatch.setattr(guard, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(guard.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "private"))
    return source, original, expected, render, calls


def test_private_patch_survives_scene_reset_and_is_idempotent(runtime):
    source, original, expected, render, calls = runtime
    metadata = guard.prepare_rt_ray_guard()
    setter = render.set_camera_shader_dir
    assert metadata["rt_ray_guard"] is True
    assert guard.prepare_rt_ray_guard() == metadata
    assert render.set_camera_shader_dir is setter
    # Both initial scene setup and later resets select the same private shaders.
    render.set_camera_shader_dir("rt")
    render.set_camera_shader_dir("rt")
    render.set_camera_shader_dir("default")
    assert calls[0] == calls[1] and calls[2] == "default"
    private = Path(calls[0])
    assert private != source
    assert (private / "camera.rgen").read_bytes() == expected
    assert (private / "camera.rchit").read_bytes() == (source / "camera.rchit").read_bytes()
    assert (source / "camera.rgen").read_bytes() == original
    metadata["rt_ray_guard"] = False
    assert guard.prepare_rt_ray_guard()["rt_ray_guard"] is True


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_shader_auto_skips_but_forced_mode_fails(runtime, monkeypatch, missing):
    source, _, _, render, _ = runtime
    if missing:
        (source / "camera.rgen").unlink()
    else:
        (source / "camera.rgen").write_bytes(b"a different shader version")
    setter = render.set_camera_shader_dir
    assert guard.prepare_rt_ray_guard()["status"] == "unrecognized_shader"
    assert render.set_camera_shader_dir is setter
    guard.RENDER_RUNTIME.clear()
    monkeypatch.setenv("COSMOS_RT_RAY_GUARD", "1")
    with pytest.raises(RuntimeError, match="Unrecognized SAPIEN RT shader"):
        guard.prepare_rt_ray_guard()
    assert not guard.RENDER_RUNTIME
    assert render.set_camera_shader_dir is setter


def test_disabled_mode_preserves_original_shader(runtime, monkeypatch):
    _, _, _, render, calls = runtime
    monkeypatch.setenv("COSMOS_RT_RAY_GUARD", "0")
    assert guard.prepare_rt_ray_guard()["status"] == "disabled"
    render.set_camera_shader_dir("rt")
    assert calls == ["rt"]


def test_patch_checksum_failure_does_not_install(runtime, monkeypatch):
    _, _, _, render, _ = runtime
    setter = render.set_camera_shader_dir
    monkeypatch.setattr(guard, "PATCHED_SHA256", "wrong")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        guard.prepare_rt_ray_guard()
    assert not guard.RENDER_RUNTIME
    assert render.set_camera_shader_dir is setter


def test_legacy_environment_variable_supported(runtime, monkeypatch):
    monkeypatch.delenv("COSMOS_RT_RAY_GUARD")
    monkeypatch.setenv("LINGBOT_RT_RAY_GUARD", "1")
    assert guard.prepare_rt_ray_guard()["rt_ray_guard_mode"] == "1"


def test_invalid_mode_fails(runtime, monkeypatch):
    monkeypatch.setenv("COSMOS_RT_RAY_GUARD", "invalid")
    with pytest.raises(ValueError, match="must be auto, 0 or 1"):
        guard.prepare_rt_ray_guard()
