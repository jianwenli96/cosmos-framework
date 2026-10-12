# SPDX-License-Identifier: OpenMDW-1.1
"""Verified SAPIEN RT workaround ported from lingbot-va-icl's RoboTwin client.

Keep the shader bytes identical to the validated fix. SAPIEN is imported lazily
so model servers and CPU tests do not require the simulator installation.
"""

import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

SOURCE_SHA256 = "c5d4f1e21e1da1b90bcdd2f2124e87ec72fd5191f4434620abbaaf718f2fe75e"
PATCHED_SHA256 = "499e4dd8dfcb45209dbb9587b70b353398db0ef7b9f7b8e7e162388a3ec7e5c3"
RENDER_RUNTIME = {}


def prepare_rt_ray_guard():
    """Guard zero-direction secondary rays in the verified SAPIEN 3.0.0b1 shader.

    Use a private shader copy; never edit the installed SAPIEN package. All
    RoboTwin scene resets selecting 'rt' must continue to use this same copy.
    """
    if RENDER_RUNTIME:
        return dict(RENDER_RUNTIME)
    import sapien as sapien_package

    mode = os.environ.get("COSMOS_RT_RAY_GUARD", os.environ.get("LINGBOT_RT_RAY_GUARD", "auto"))
    if mode not in ("auto", "0", "1"):
        raise ValueError("COSMOS_RT_RAY_GUARD must be auto, 0 or 1")
    source = Path(sapien_package.__file__).resolve().parent / "vulkan_shader" / "rt"
    shader = source / "camera.rgen"
    original = shader.read_bytes() if shader.exists() else b""
    digest = hashlib.sha256(original).hexdigest() if original else None
    known_digest = SOURCE_SHA256
    metadata = dict(rt_ray_guard=False, rt_source_sha256=digest, rt_ray_guard_mode=mode)
    if mode == "0" or digest != known_digest:
        if mode == "1":
            raise RuntimeError(f"Unrecognized SAPIEN RT shader: {shader} ({digest})")
        metadata["status"] = "disabled" if mode == "0" else "unrecognized_shader"
        RENDER_RUNTIME.update(metadata)
        print(f"SAPIEN RT runtime: {RENDER_RUNTIME}", flush=True)
        return dict(RENDER_RUNTIME)

    needle = b"      traceRayEXT(tlas,"
    guard = (
        b"      // An absorbing material has no next ray. Do not trace a zero/invalid direction.\n"
        b"      if (dot(ray.direction, ray.direction) == 0.0 ||\n"
        b"          any(isnan(ray.direction)) || any(isinf(ray.direction))) {\n"
        b"        break;\n"
        b"      }\n\n"
    )
    if original.count(needle) != 1:
        raise RuntimeError("Unexpected SAPIEN RT ray-generation shader layout")
    patched = original.replace(needle, guard + needle)
    patched_digest = hashlib.sha256(patched).hexdigest()
    if patched_digest != PATCHED_SHA256:
        raise RuntimeError("SAPIEN RT ray guard checksum mismatch")
    private = Path(tempfile.mkdtemp(prefix="cosmos-rt-ray-guard-"))
    atexit.register(shutil.rmtree, private, ignore_errors=True)
    shutil.copytree(source, private, dirs_exist_ok=True)
    (private / "camera.rgen").write_bytes(patched)
    import sapien.core as sapien

    original_set_shader = sapien.render.set_camera_shader_dir

    def set_camera_shader_dir(shader_dir):
        return original_set_shader(str(private) if str(shader_dir) == "rt" else shader_dir)

    sapien.render.set_camera_shader_dir = set_camera_shader_dir
    metadata.update(rt_ray_guard=True, rt_patched_sha256=patched_digest, status="zero_direction_guard")
    RENDER_RUNTIME.update(metadata)
    print(f"SAPIEN RT runtime: {RENDER_RUNTIME}", flush=True)
    return dict(RENDER_RUNTIME)
