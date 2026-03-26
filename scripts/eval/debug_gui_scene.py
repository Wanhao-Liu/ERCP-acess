#!/usr/bin/env python3
"""
Phase 1 deep diagnostic: inspect Viewport internals to find why
USD relationship fix doesn't make VNC show the camera view.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, "/data/ERCP/ercp_access")
sys.stdout = sys.stderr

import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCylinder
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from pxr import UsdLux, UsdGeom, Gf, Sdf

world = World(physics_dt=0.002, rendering_dt=0.02, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Lights
for path, typ, intensity in [
    ("/World/key_light",  "DistantLight", 3000.0),
    ("/World/dome_light", "DomeLight",    1000.0),
]:
    p = stage.DefinePrim(path, typ)
    if typ == "DistantLight":
        UsdLux.DistantLight(p).CreateIntensityAttr(intensity)
        UsdGeom.Xformable(p).AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
    else:
        UsdLux.DomeLight(p).CreateIntensityAttr(intensity)

# Large papilla
FixedCylinder(
    prim_path="/World/papilla", name="papilla",
    position=np.array([0.30, 0.0, 0.0]),
    radius=0.05, height=0.05,
    color=np.array([0.9, 0.3, 0.3]),
)

# Scope / camera
prim_utils.create_prim("/World/scope_camera", "Xform")
scope_prim = XFormPrim("/World/scope_camera", name="scope_camera")
cam_prim_path = "/World/scope_camera/camera_sensor"
camera = Camera(prim_path=cam_prim_path, name="camera_sensor", resolution=(512, 512))
world.reset()
camera.initialize()
scope_prim.set_world_pose(
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
)

# Step before any viewport manipulation
for _ in range(5):
    world.step(render=True)
    app.update()

# ── Deep Viewport inspection ──────────────────────────────────────────────────
print("\n=== [DIAG] Viewport API deep inspection ===")
try:
    from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name
    import omni.kit.viewport.utility as vp_util

    vp = get_active_viewport()
    print(f"  viewport object : {vp}")
    print(f"  type            : {type(vp)}")

    # Check all available methods/attrs on viewport
    methods = [m for m in dir(vp) if not m.startswith('__')]
    print(f"  available attrs : {methods}")

    # Try get_active_camera
    if hasattr(vp, 'get_active_camera'):
        print(f"  get_active_camera() = {vp.get_active_camera()}")

    # Try camera_path attribute
    if hasattr(vp, 'camera_path'):
        print(f"  camera_path = {vp.camera_path}")

    # Try viewport_api
    if hasattr(vp, 'viewport_api'):
        api = vp.viewport_api
        print(f"  viewport_api = {api}")
        api_methods = [m for m in dir(api) if not m.startswith('__')]
        print(f"  api attrs = {api_methods}")

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

# ── Try every method to set camera ───────────────────────────────────────────
print("\n=== [DIAG] Attempting all camera-set approaches ===")

# Approach A: set_active_camera (API)
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    vp.set_active_camera(cam_prim_path)
    print(f"  A: set_active_camera({cam_prim_path}) -> OK")
    if hasattr(vp, 'get_active_camera'):
        print(f"     verify get_active_camera() = {vp.get_active_camera()}")
    if hasattr(vp, 'camera_path'):
        print(f"     verify camera_path = {vp.camera_path}")
except Exception as e:
    print(f"  A: ERROR {e}")

# Approach B: USD relationship
try:
    vp_rp_path = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
    vp_rp = stage.GetPrimAtPath(vp_rp_path)
    if vp_rp.IsValid():
        rel = vp_rp.GetRelationship("camera")
        if rel:
            old = rel.GetTargets()
            rel.SetTargets([Sdf.Path(cam_prim_path)])
            new = rel.GetTargets()
            print(f"  B: USD rel camera: {old} -> {new}")
        else:
            print("  B: no 'camera' relationship on RenderProduct")
    else:
        print(f"  B: RenderProduct prim not found")
except Exception as e:
    print(f"  B: ERROR {e}")

# Approach C: omni.kit.commands SetActiveCameraCommand
try:
    import omni.kit.commands
    omni.kit.commands.execute("SetActiveCameraCommand",
                              camera_path=Sdf.Path(cam_prim_path))
    print(f"  C: SetActiveCameraCommand({cam_prim_path}) -> OK")
except Exception as e:
    print(f"  C: ERROR {e}")

# Approach D: viewport_api.camera_path setter
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if hasattr(vp, 'viewport_api') and hasattr(vp.viewport_api, 'camera_path'):
        vp.viewport_api.camera_path = Sdf.Path(cam_prim_path)
        print(f"  D: viewport_api.camera_path = {cam_prim_path} -> OK")
        print(f"     verify: {vp.viewport_api.camera_path}")
    else:
        print("  D: viewport_api.camera_path not available")
except Exception as e:
    print(f"  D: ERROR {e}")

# ── Step and check result ─────────────────────────────────────────────────────
print("\n=== [DIAG] Stepping 30 frames after all camera-set attempts ===")
for _ in range(30):
    world.step(render=True)
    app.update()

# Replicator annotator
rp = rep.create.render_product(cam_prim_path, resolution=(512, 512))
rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_ann.attach([rp])
for _ in range(10):
    world.step(render=True)
    app.update()
frame = rgb_ann.get_data()
print(f"  annotator frame: max={frame.max()} mean={frame.mean():.1f}")

# Final viewport camera state
print("\n=== [DIAG] Final viewport camera state ===")
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    for attr in ['camera_path', 'get_active_camera']:
        if hasattr(vp, attr):
            val = getattr(vp, attr)
            print(f"  {attr} = {val() if callable(val) else val}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== PAUSING 90s — CHECK VNC ===")
sys.stderr.flush()
for i in range(90):
    world.step(render=True)
    app.update()
    if i % 20 == 0:
        f = rgb_ann.get_data()
        print(f"  t={i}s frame max={f.max()} mean={f.mean():.1f}")
        sys.stderr.flush()

app.close()
