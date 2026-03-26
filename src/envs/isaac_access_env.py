"""
IsaacAccessEnv — biliary access environment backed by Isaac Sim.

Design decisions:
  - Does NOT use ManagerBasedRLEnv (vector-parallel, incompatible interface).
  - Uses omni.isaac.core.World + FixedCylinder + XFormPrim (kinematic control).
  - Scope is controlled kinematically: position/orientation set directly each step.
  - Camera (omni.isaac.sensor.Camera) is mounted at scope tip prim.
  - Oracle state mode (use_oracle=True): calls compute_oracle_gt_info() to bypass
    perception; pure-sim validation of FSM/gating/policy.

Coordinate convention (aligns with oracle_state.py):
  Camera looks along world +X axis.
  scope_pos[0] → X (insertion depth)
  scope_pos[1] → Y (lateral, e_x error axis)
  scope_pos[2] → Z (vertical, e_y error axis)
  papilla_pos  = [0.10, 0.0, 0.0]

Usage (inside isaaclab.sh -p):
    import sys
    sys.path.insert(0, "/data/ERCP/ercp_access")
    from src.envs.isaac_access_env import IsaacAccessEnv
    env = IsaacAccessEnv(use_oracle=True, headless=True)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""
from __future__ import annotations

import math
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.envs.base_env import AccessEnvBase
from src.envs.isaac_lab_cfg import IsaacAccessSceneCfg, DEFAULT_SCENE_CFG
from src.perception.oracle_state import compute_oracle_gt_info
from src.perception.state_builder import StateBuilder
from src.gating.phase_manager import PhaseManager
from src.gating.insertion_gate import InsertionGate, SafetyRules


class IsaacAccessEnv(AccessEnvBase):
    """
    Isaac Sim backed biliary access environment.

    The SimulationApp is created lazily on first __init__ call.
    Only one IsaacAccessEnv instance should exist per process.
    """

    # ── Class-level SimulationApp handle (singleton) ──────────────────────────
    _sim_app = None

    def __init__(
        self,
        use_oracle: bool = True,
        headless: bool = True,
        max_steps: int = 1500,
        camera_width: int = 512,
        camera_height: int = 512,
        init_dist_range: Tuple[float, float] = (0.05, 0.10),
        init_angle_range_deg: Tuple[float, float] = (10.0, 15.0),
        insertion_scale: float = 0.0025,
        bend_scale: float = 0.05,
        ref_distance: float = 0.05,
        ref_scale: float = 0.15,
        success_insert_depth: float = 0.02,
        scene_cfg: Optional[IsaacAccessSceneCfg] = None,
        seed: int = 42,
    ):
        self._use_oracle = use_oracle
        self._headless = headless
        self._max_steps = max_steps
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._init_dist_range = init_dist_range
        self._init_angle_range_deg = init_angle_range_deg
        self._insertion_scale = insertion_scale
        self._bend_scale = bend_scale
        self._ref_distance = ref_distance
        self._ref_scale = ref_scale
        self._success_insert_depth = success_insert_depth
        self._scene_cfg = scene_cfg or DEFAULT_SCENE_CFG

        self._rng = np.random.default_rng(seed)

        # ── Control components (same as ToyAccessEnv) ────────────────────────
        self._state_builder = StateBuilder(history_len=8, use_oracle=use_oracle)
        self._insertion_gate = InsertionGate()
        self._safety_rules = SafetyRules()
        self._phase_manager = PhaseManager(
            conf_thresh=0.5,
            recovery_steps=8,   # matches ToyAccessEnv reference value
            insertion_gate=self._insertion_gate,
        )

        # ── Internal state ────────────────────────────────────────────────────
        # Scope tip position in world frame [x, y, z] metres
        self._scope_pos = np.zeros(3, dtype=np.float64)
        # Scope orientation as Euler angles [pitch_rad, yaw_rad] (roll=0)
        self._scope_euler = np.zeros(2, dtype=np.float64)  # [pitch, yaw]
        # Papilla position (fixed)
        self._papilla_pos = np.array(
            self._scene_cfg.papilla.position, dtype=np.float64
        )
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._phase = 0
        self._insert_depth = 0.0
        self._episode_id = ""

        # ── Isaac Sim objects (populated in _init_isaac) ──────────────────────
        self._world = None
        self._papilla_obj = None       # FixedCylinder
        self._scope_prim = None        # XFormPrim (scope tip / camera mount)
        self._camera = None            # omni.isaac.sensor.Camera
        self._rgb_annotator = None     # replicator annotator

        # ── Start Isaac Sim ───────────────────────────────────────────────────
        self._init_isaac()

    # ── AccessEnvBase properties ──────────────────────────────────────────────

    @property
    def dt(self) -> float:
        return self._scene_cfg.rendering_dt

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    # ── Isaac Sim initialisation ──────────────────────────────────────────────

    def _init_isaac(self):
        """Start SimulationApp (if not started) and build the scene."""
        # --- 1. Launch SimulationApp (singleton) ---
        if IsaacAccessEnv._sim_app is None:
            from isaacsim import SimulationApp
            IsaacAccessEnv._sim_app = SimulationApp({"headless": self._headless})

        # --- 2. Import Isaac modules (only after app is running) ---
        from omni.isaac.core import World
        from omni.isaac.core.objects import FixedCylinder
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.sensor import Camera
        import omni.replicator.core as rep
        import omni.isaac.core.utils.prims as prim_utils

        # Store module references for use in step/reset
        self._World = World
        self._XFormPrim = XFormPrim
        self._Camera = Camera
        self._rep = rep

        # --- 3. Create World ---
        self._world = World(
            physics_dt=self._scene_cfg.physics_dt,
            rendering_dt=self._scene_cfg.rendering_dt,
            stage_units_in_meters=1.0,
        )

        # --- 4. Add lighting ---
        self._add_lighting(prim_utils)

        # --- 5. Add papilla (static cylinder) ---
        # In GUI mode scale up 10× so it's visible in the Viewport world-view.
        # Oracle state uses scope_pos/papilla_pos directly, so visual scale
        # does not affect e_x/e_y/scale computation.
        _vis_scale = 10.0 if not self._headless else 1.0
        self._papilla_obj = FixedCylinder(
            prim_path=self._scene_cfg.papilla.prim_path,
            name=self._scene_cfg.papilla.name,
            position=np.array(self._scene_cfg.papilla.position),
            radius=self._scene_cfg.papilla.radius * _vis_scale,
            height=self._scene_cfg.papilla.height * _vis_scale,
            color=np.array(self._scene_cfg.papilla.color),
        )

        # --- 6. Add scope tip XFormPrim (camera mount point) ---
        prim_utils.create_prim(
            prim_path=self._scene_cfg.camera.prim_path,
            prim_type="Xform",
        )
        self._scope_prim = XFormPrim(
            prim_path=self._scene_cfg.camera.prim_path,
            name=self._scene_cfg.camera.name,
        )

        # --- 7. Add Camera attached to scope prim ---
        self._camera = Camera(
            prim_path=self._scene_cfg.camera.prim_path + "/camera_sensor",
            name="camera_sensor",
            resolution=(self._camera_width, self._camera_height),
        )

        # --- 8. Reset world to initialize physics ---
        self._world.reset()

        # --- 9. Initialize camera and set up replicator annotator ---
        self._camera.initialize()

        # Set camera focal length for desired FOV
        # For a horizontal FoV θ, focal_length = sensor_width / (2 * tan(θ/2))
        # Isaac sensor_width default ≈ 20.955mm with aperture 20.955
        # We use the set_focal_length API:
        fov_rad = math.radians(self._scene_cfg.camera.fov_deg)
        # Sensor horizontal aperture is typically 20.955mm in Isaac
        sensor_aperture = 20.955  # mm (default horizontal aperture)
        focal_length_mm = sensor_aperture / (2.0 * math.tan(fov_rad / 2.0))
        self._camera.set_focal_length(focal_length_mm)

        # Set up replicator render product + RGB annotator
        cam_prim_path = self._scene_cfg.camera.prim_path + "/camera_sensor"
        render_product = rep.create.render_product(
            cam_prim_path,
            resolution=(self._camera_width, self._camera_height),
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([render_product])

        # --- 10. In GUI mode: position the default Persp camera to see the scene ---
        # The Viewport shows the default /OmniverseKit_Persp camera (world view).
        # scope_camera renders to annotator only (for RL obs).
        # We move Persp to a position where papilla + scope are clearly visible.
        if not self._headless:
            try:
                import omni.usd
                from pxr import UsdGeom, Gf
                _stage = omni.usd.get_context().get_stage()
                # Move Persp camera to look at scene from slightly above and behind
                persp = _stage.GetPrimAtPath("/OmniverseKit_Persp")
                if persp.IsValid():
                    xf = UsdGeom.Xformable(persp)
                    xf.ClearXformOpOrder()
                    # Position: behind origin (scope start), elevated, looking +X
                    xf.AddTranslateOp().Set(Gf.Vec3d(-0.15, -0.10, 0.08))
                    xf.AddRotateXYZOp().Set(Gf.Vec3f(-20.0, -30.0, 0.0))
            except Exception:
                pass  # non-critical

    def _add_lighting(self, prim_utils):
        """Add lighting to the scene.

        DistantLight: directional key light pointing along +X (toward papilla).
        DomeLight: ambient fill so the scene is never fully black in any viewport.
        """
        import omni.usd
        from pxr import UsdLux, UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        # Key light (DistantLight)
        light_path = self._scene_cfg.light_prim_path
        if not stage.GetPrimAtPath(light_path).IsValid():
            light_prim = stage.DefinePrim(light_path, "DistantLight")
            light = UsdLux.DistantLight(light_prim)
            light.CreateIntensityAttr(self._scene_cfg.light_intensity)
            xform = UsdGeom.Xformable(light_prim)
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))

        # Ambient fill (DomeLight) — illuminates scene from all directions
        dome_path = self._scene_cfg.light_prim_path + "_dome"
        if not stage.GetPrimAtPath(dome_path).IsValid():
            dome_prim = stage.DefinePrim(dome_path, "DomeLight")
            UsdLux.DomeLight(dome_prim).CreateIntensityAttr(800.0)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        case_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # --- 1. Randomise initial scope position ---
        dist = self._rng.uniform(*self._init_dist_range)
        angle_xy = self._rng.uniform(0.0, 2.0 * math.pi)
        # Scope starts behind papilla along -X, with random lateral offset
        lateral = dist * 0.5  # half the dist as lateral offset
        self._scope_pos = np.array([
            self._papilla_pos[0] - dist,          # behind papilla along X
            lateral * math.cos(angle_xy),          # Y lateral
            lateral * math.sin(angle_xy),          # Z lateral
        ], dtype=np.float64)

        # --- 2. Randomise initial orientation (pitch/yaw towards papilla) ---
        max_angle_rad = math.radians(
            self._rng.uniform(*self._init_angle_range_deg)
        )
        self._scope_euler = np.array([
            self._rng.uniform(-max_angle_rad, max_angle_rad),  # pitch
            self._rng.uniform(-max_angle_rad, max_angle_rad),  # yaw
        ], dtype=np.float64)

        # --- 3. Reset internal counters ---
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._phase = 0
        self._insert_depth = 0.0
        self._state_builder.reset()
        self._phase_manager.reset()
        self._safety_rules.reset()

        # --- 4. Apply pose to Isaac Sim ---
        self._apply_scope_pose()

        # --- 5. Step sim a few times to stabilise ---
        for _ in range(5):
            self._world.step(render=True)

        # --- 6. Read camera + compute state ---
        rgb = self._read_camera_rgb()
        access_state = self._compute_oracle_state()

        # --- 7. Build obs ---
        self._episode_id = case_id or str(uuid.uuid4())[:8]
        obs = self._make_obs_dict(
            state=access_state.to_array(),
            phase=self._phase,
            prev_action=self._prev_action,
            valid=True,
            rgb=rgb,
        )
        info = {
            "episode_id": self._episode_id,
            "case_id": self._episode_id,
            "difficulty": difficulty or "medium",
            "init_pose": self._scope_pos.copy(),
        }
        return obs, info

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # 1. Validate
        action = self.validate_action(action)

        # 2. Apply action as kinematic delta
        #    action[0] = delta_insert → scope_pos[0] (X, insertion axis)
        #    action[1] = delta_pitch  → scope_euler[0] (pitch) and scope_pos[2] (Z)
        #    action[2] = delta_yaw    → scope_euler[1] (yaw) and scope_pos[1] (Y)
        self._scope_pos[0] += float(action[0]) * self._insertion_scale
        # Lateral movement derived from bend angles
        self._scope_euler[0] += float(action[1]) * self._bend_scale
        self._scope_euler[1] += float(action[2]) * self._bend_scale
        # Clamp euler angles to ±90 deg
        self._scope_euler = np.clip(
            self._scope_euler, -math.pi / 2, math.pi / 2
        )
        # Pose-position coupling: bending the scope tip also shifts the camera viewpoint
        # laterally by a small fraction. Factor 0.1 = bend_scale(0.05) × coupling(0.1) per
        # unit action, resulting in 0.005m/unit lateral drift — much smaller than the main
        # lateral control in ToyAccessEnv (bend_scale=0.01m/unit direct). This keeps the
        # oracle state responsive to bend commands while remaining numerically stable.
        self._scope_pos[2] += float(action[1]) * self._bend_scale * 0.1  # pitch → Z drift (pose-position coupling)
        self._scope_pos[1] += float(action[2]) * self._bend_scale * 0.1  # yaw → Y drift (pose-position coupling)

        self._step_count += 1
        self._prev_action = action.copy()

        # 3. Push pose to Isaac Sim and step physics
        self._apply_scope_pose()
        for _ in range(10):  # decimation = 10 sim steps per control step
            self._world.step(render=True)

        # 4. Read camera + oracle state
        rgb = self._read_camera_rgb()
        access_state = self._compute_oracle_state()

        # 5. FSM update (pre-safety event flags)
        event_flags_pre = {
            "target_loss": False, "off_axis": False,
            "no_progress": False, "recovery_triggered": False,
            "unsafe_insert_attempt": False,
        }
        self._phase = self._phase_manager.update(access_state, event_flags_pre)

        # 6. Insertion execution check (only in phase 2)
        insert_executed = False
        if self._phase == 2 and float(action[0]) > 0.1:
            self._insert_depth += float(action[0]) * self._insertion_scale
            insert_executed = True

        # 7. Success condition (aligns with ToyAccessEnv)
        success = (
            access_state.is_aligned(e_thresh=0.08)
            and access_state.scale >= 0.3
            and self._insert_depth >= self._success_insert_depth
        )

        # 8. Reward
        reward = self._compute_reward(access_state, insert_executed, success)

        # 9. Termination
        terminated = success
        truncated = self._step_count >= self._max_steps

        # 10. Safety event flags
        event_flags = self._safety_rules.check(
            access_state, self._phase, insert_executed
        )

        # 11. Build info
        info = {
            "phase": self._phase,
            "gating": self._insertion_gate.get_gating(access_state, self._phase),
            "alignment_error": access_state.alignment_error,
            "conf": access_state.conf,
            "readiness": access_state.readiness,
            "target_visible": True,
            "insert_executed": insert_executed,
            "event_flags": event_flags,
            "success": success,
        }

        # 12. Build obs
        obs = self._make_obs_dict(
            state=access_state.to_array(),
            phase=self._phase,
            prev_action=self._prev_action,
            valid=True,
            rgb=rgb,
        )
        return obs, reward, terminated, truncated, info

    # ── close ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down Isaac Sim. After calling this, do not use the env."""
        if IsaacAccessEnv._sim_app is not None:
            IsaacAccessEnv._sim_app.close()
            IsaacAccessEnv._sim_app = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_scope_pose(self):
        """
        Set scope tip XFormPrim world pose from self._scope_pos / _scope_euler.

        Isaac XFormPrim.set_world_pose() accepts:
          position: np.ndarray [3] in metres
          orientation: np.ndarray [4] as quaternion [w, x, y, z]
        """
        pos = self._scope_pos.astype(np.float32)
        quat = self._euler_to_quat(self._scope_euler[0], self._scope_euler[1])
        self._scope_prim.set_world_pose(position=pos, orientation=quat)

    @staticmethod
    def _euler_to_quat(pitch: float, yaw: float) -> np.ndarray:
        """
        Convert pitch (rotation around Y) and yaw (rotation around Z) to
        quaternion [w, x, y, z].

        Camera looks along +X by default (zero rotation).
        Pitch = rotation around world Y axis.
        Yaw = rotation around world Z axis.
        Order: yaw first, then pitch (extrinsic ZY).
        """
        # Half angles
        cp = math.cos(pitch / 2.0)
        sp = math.sin(pitch / 2.0)
        cy = math.cos(yaw / 2.0)
        sy = math.sin(yaw / 2.0)

        # Quaternion for pitch (Y-axis rotation)
        qp = np.array([cp, 0.0, sp, 0.0])  # [w, x, y, z]

        # Quaternion for yaw (Z-axis rotation)
        qy = np.array([cy, 0.0, 0.0, sy])  # [w, x, y, z]

        # Combined: q = qy * qp (apply pitch first, then yaw)
        w = qy[0] * qp[0] - qy[1] * qp[1] - qy[2] * qp[2] - qy[3] * qp[3]
        x = qy[0] * qp[1] + qy[1] * qp[0] + qy[2] * qp[3] - qy[3] * qp[2]
        y = qy[0] * qp[2] - qy[1] * qp[3] + qy[2] * qp[0] + qy[3] * qp[1]
        z = qy[0] * qp[3] + qy[1] * qp[2] - qy[2] * qp[1] + qy[3] * qp[0]
        return np.array([w, x, y, z], dtype=np.float32)

    def _compute_oracle_state(self):
        """
        Compute AccessState using oracle ground truth (bypasses perception).

        Calls compute_oracle_gt_info() with current scope_pos as camera_pos,
        then feeds gt_info into StateBuilder.update().
        """
        gt_info = compute_oracle_gt_info(
            scope_tip_pos=self._scope_pos,
            papilla_pos=self._papilla_pos,
            camera_pos=self._scope_pos,   # camera at scope tip
            image_width=self._camera_width,
            image_height=self._camera_height,
            ref_distance=self._ref_distance,
            ref_scale=self._ref_scale,
        )
        return self._state_builder.update(rgb=None, gt_info=gt_info)

    def _read_camera_rgb(self) -> np.ndarray:
        """
        Read current RGB frame from Isaac replicator annotator.

        Returns float32 [3, H, W] in [0, 1].
        Falls back to zeros if annotator returns None.
        """
        try:
            rgba = self._rgb_annotator.get_data()  # uint8 (H, W, 4) or None
            if rgba is None or rgba.size == 0:
                return np.zeros(self.RGB_SHAPE, dtype=np.float32)
            # Drop alpha, convert to float, normalise, transpose to CHW
            rgb = rgba[:, :, :3].astype(np.float32) / 255.0  # (H, W, 3)
            rgb = rgb.transpose(2, 0, 1)                     # (3, H, W)
            # Resize if needed (should match camera resolution already)
            if rgb.shape[1] != self._camera_height or rgb.shape[2] != self._camera_width:
                # Fallback: just return zeros rather than import cv2
                return np.zeros(self.RGB_SHAPE, dtype=np.float32)
            return rgb.astype(np.float32)
        except Exception:
            return np.zeros(self.RGB_SHAPE, dtype=np.float32)

    def _compute_reward(self, state, insert_executed: bool, success: bool) -> float:
        """Reward function aligned with ToyAccessEnv."""
        r = 0.0
        r -= state.alignment_error * 1.0          # alignment penalty
        r += (state.scale - 0.1) * 2.0            # approach reward
        if insert_executed and not state.is_aligned():
            r -= 5.0                               # unsafe insert penalty
        if success:
            r += 50.0
        return float(r)
