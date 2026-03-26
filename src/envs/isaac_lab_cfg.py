"""
isaac_lab_cfg.py — Scene configuration dataclasses for IsaacAccessEnv.

Uses omni.isaac.core (Isaac Sim 4.2 / omni.isaac.core-3.19.5) API directly,
NOT the IsaacLab ManagerBasedRLEnv, because we need a simple single-env
interface matching AccessEnvBase.

Available at runtime inside isaaclab.sh -p after SimulationApp is started:
  - omni.isaac.core.World
  - omni.isaac.core.objects.FixedCylinder
  - omni.isaac.core.prims.XFormPrim
  - omni.isaac.sensor.Camera
  - omni.replicator.core

NOTE: This file only defines config dataclasses.
      Isaac imports happen lazily inside IsaacAccessEnv.__init__()
      so that this module can be imported in plain Python for type checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PapillaObjectCfg:
    """
    Configuration for the papilla target cylinder in Isaac Sim.

    The cylinder is created as a FixedCylinder (kinematic rigid body,
    no physics simulation, stays at its initial pose).
    """
    prim_path: str = "/World/papilla"
    name: str = "papilla"
    # Position in world frame [x, y, z] metres.
    # Camera looks along +X; papilla is ahead of scope start position.
    position: Tuple[float, float, float] = (0.10, 0.0, 0.0)
    radius: float = 0.005   # 5 mm
    height: float = 0.003   # 3 mm — thin disc representing ostium
    # RGB colour of the papilla (pink/red so it's visible to camera)
    color: Tuple[float, float, float] = (0.9, 0.3, 0.3)


@dataclass
class ScopeCameraCfg:
    """
    Configuration for the scope-tip camera.

    Camera is mounted at the scope tip prim and looks along +X.
    It is repositioned each step kinematically via XFormPrim.set_world_pose().
    """
    prim_path: str = "/World/scope_camera"
    name: str = "scope_camera"
    resolution: Tuple[int, int] = (512, 512)   # (width, height)
    # Horizontal field of view in degrees — controls perspective projection.
    # NOTE: omni.isaac.sensor.Camera does not accept fov directly;
    #       we set it via focal_length after construction.
    fov_deg: float = 60.0
    # Annotator type used by replicator to read RGB frames.
    # "rgb" returns RGBA uint8 (H, W, 4).
    annotator_type: str = "rgb"


@dataclass
class IsaacAccessSceneCfg:
    """
    Top-level scene configuration for IsaacAccessEnv.

    Groups all sub-configs. Passed into IsaacAccessEnv.__init__().
    """
    papilla: PapillaObjectCfg = field(default_factory=PapillaObjectCfg)
    camera: ScopeCameraCfg = field(default_factory=ScopeCameraCfg)

    # Physics settings
    physics_dt: float = 0.002       # sim_dt (1/500 s)
    rendering_dt: float = 0.02      # control dt (decimation * sim_dt)

    # Lighting prim path (DistantLight for uniform scene illumination)
    light_prim_path: str = "/World/distant_light"
    light_intensity: float = 5000.0

    # Ground plane (keeps objects from falling if gravity is on)
    ground_prim_path: str = "/World/ground"
    add_ground: bool = False  # Not needed; papilla is kinematic, scope has no physics


# ── Default singleton ─────────────────────────────────────────────────────────

DEFAULT_SCENE_CFG = IsaacAccessSceneCfg()
