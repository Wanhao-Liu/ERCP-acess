# TASK_INTERFACE.md — ERCP Access Task Interface (Frozen)

> **Status: FROZEN** — This document defines the canonical interface for all modules in the ERCP autonomous biliary access project.
> Do not change obs/action shapes, field names, or metrics keys without a version bump and team review.
>
> Last updated: 2026-03-24

---

## 1. Task Definition

| Field | Value |
|-------|-------|
| Task name | `near_papilla_biliary_access` |
| Robot | ERCP daughter scope (3-DOF cable-driven) |
| Start condition | Daughter scope placed near papilla; papilla visible in endoscope FoV; pose not yet insertion-ready |
| Goal | Close-loop alignment → approach → guarded insertion → biliary access established |
| Out of scope | Long-range papilla search; post-insertion bile duct navigation; full ERCP procedure |
| Sole sensor | Monocular RGB endoscope camera (512×512, 30 fps) |

---

## 2. Observation Dictionary

```python
obs: Dict[str, Any] = {
    "rgb":         np.ndarray,   # float32, shape [3, 512, 512], range [0, 1]
    "state":       np.ndarray,   # float32, shape [6]
    "phase":       np.int64,     # scalar: 0=alignment, 1=approach, 2=insertion, 3=recovery
    "prev_action": np.ndarray,   # float32, shape [3]
    "valid":       np.bool_,     # True if perception is reliable
}
```

### `state` vector — 6D AccessState

| Index | Name | Range | Semantics |
|-------|------|-------|-----------|
| 0 | `e_x` | [-1, 1] | Horizontal offset of papilla centroid from image center, normalized by `ref_scale=0.15` |
| 1 | `e_y` | [-1, 1] | Vertical offset, same normalization |
| 2 | `scale` | [0, 1] | Apparent target area relative to image area (papilla size proxy) |
| 3 | `conf` | [0, 1] | Perception confidence (detector score or oracle 1.0) |
| 4 | `stability` | [0, 1] | Temporal stability of [e_x, e_y] over 8-frame window: `exp(-var × 10)` |
| 5 | `readiness` | [0, 1] | Composite insertion readiness: `min(alignment_ok, scale_ok, stability_ok, conf_ok)` |

**Axis mapping**: `e_x` (horizontal error) → `yaw` correction → `action[2]`; `e_y` (vertical error) → `pitch` correction → `action[1]`

**In Phase 0 (oracle/debug)**: `conf=1.0`, `stability=1.0`, `readiness` computed from oracle geometry.

### `rgb` tensor

- Shape: `[3, 512, 512]` (C, H, W), float32, range `[0.0, 1.0]`
- Source: endoscope monocular RGB, resized from 1920×1080 via INTER_AREA
- Channel order: RGB (not BGR)

---

## 3. Action

```python
action: np.ndarray  # float32, shape [3], range [-1.0, 1.0]
# action = [delta_insert, delta_pitch, delta_yaw]
```

| Index | Component | Physical mapping | Scale |
|-------|-----------|-----------------|-------|
| 0 | `delta_insert` | Insertion / retraction depth | × `insertion_scale = 0.0025 m` |
| 1 | `delta_pitch` | Pitch (up/down bending, M3) | × `bend_scale = 0.05 rad` |
| 2 | `delta_yaw` | Yaw (left/right bending, M4) | × `bend_scale = 0.05 rad` |

Positive `delta_insert` → advance toward papilla.
Positive `delta_pitch` → pitch up; positive `delta_yaw` → yaw right.

**After postprocessing** (see §6), the executed action may differ from the raw policy output.

---

## 4. Environment API

### `reset()`

```python
obs_dict, reset_info_dict = env.reset(
    seed: Optional[int] = None,
    case_id: Optional[str] = None,
    difficulty: Optional[str] = None,   # "easy" | "medium" | "hard"
)
```

`reset_info_dict` fields:

```python
{
    "episode_id": str,
    "case_id":    str,
    "difficulty": str,
    "init_pose":  Optional[np.ndarray],   # float32, [3]: initial motor state
}
```

### `step()`

```python
next_obs_dict, reward, terminated, truncated, info_dict = env.step(
    action: np.ndarray   # float32, [3], pre-validated
)
```

`terminated=True` on task success or hard abort.
`truncated=True` on `max_episode_steps` timeout.

### Environment constants

```python
env.dt: float             # control timestep in seconds (ToyAccessEnv: 0.033s ≈ 30Hz)
env.max_episode_steps: int  # episode length limit (default: 200)
```

---

## 5. Info Dictionary (per step)

```python
info_dict: Dict[str, Any] = {
    "phase":           int,    # 0=alignment, 1=approach, 2=insertion, 3=recovery
    "gating":          int,    # 0=hold, 1=approach, 2=insert, 3=recover
    "alignment_error": float,  # sqrt(e_x² + e_y²), Euclidean distance from center
    "conf":            float,  # perception confidence
    "readiness":       float,  # composite insertion readiness
    "target_visible":  bool,   # True if papilla is within FoV
    "insert_executed": bool,   # True if insertion delta was applied this step
    "event_flags": {
        "target_loss":           bool,   # conf < 0.2 for ≥5 consecutive frames
        "off_axis":              bool,   # |e_x| > 0.4 or |e_y| > 0.4
        "no_progress":           bool,   # scale ΔRange < 0.01 over 20 steps (phase 1/2)
        "recovery_triggered":    bool,   # True when phase == 3
        "unsafe_insert_attempt": bool,   # insert executed while not aligned
    },
    "success": bool,   # True if biliary access established (only True at termination)
}
```

---

## 6. Action Postprocessing Pipeline

Policies output **raw actions**. The `ActionPostprocessor` transforms them before `env.step()`:

```
raw_action  (policy output)
  ↓ clip([-1, 1])
  ↓ rate_limit(max_delta_per_step=0.4)   # |Δ per step| ≤ 0.4
  ↓ phase_mask
       phase 0: action[0] = 0.0            # no insertion during alignment
       phase 3: action[1:] = 0.0           # no pitch/yaw during recovery
               action[0] = -0.8            # forced retreat
  ↓ gating
       gating 0: action = [0, 0, 0]        # hold
       gating 3: action[0] = -0.8, action[1:] = 0.0
  ↓ clip([-1, 1])
  ↓ exec_action  (sent to env.step)
```

`ActionPostprocessor.process(raw_action, phase, gating) -> exec_action`

B2 (PPO-State) does **not** use a postprocessor — raw policy output goes directly to `env.step()`.
B0, B1, and all FSM-based methods use the postprocessor.

---

## 7. Phase FSM

```
Phase 0: ALIGNMENT
  → Phase 1 (approach):   state.is_aligned() AND conf ≥ 0.5

Phase 1: APPROACH
  → Phase 0 (alignment):  NOT state.is_aligned()           (lost alignment)
  → Phase 2 (insertion):  InsertionGate.check(state)        (all 5 conditions met)

Phase 2: INSERTION
  → Phase 3 (recovery):   off_axis OR no_progress OR target_loss
  → done (success):        access established

Phase 3: RECOVERY
  → Phase 0 (alignment):  after recovery_steps=10 steps
```

`state.is_aligned(e_thresh=0.1)` → `|e_x| < 0.1 AND |e_y| < 0.1`

---

## 8. InsertionGate Conditions

All five must be satisfied simultaneously:

| Condition | Default threshold | Field |
|-----------|------------------|-------|
| Horizontal alignment | `\|e_x\| < 0.08` | `e_x` |
| Vertical alignment | `\|e_y\| < 0.08` | `e_y` |
| Scale (proximity) | `scale ≥ 0.25` | `scale` |
| Confidence | `conf ≥ 0.60` | `conf` |
| Stability | `stability ≥ 0.60` | `stability` |
| Readiness | `readiness ≥ 0.70` | `readiness` |

---

## 9. SafetyRules Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| `target_loss` | `conf < 0.2` for ≥ 5 consecutive frames | → recovery (phase 3) |
| `off_axis` | `\|e_x\| > 0.4` or `\|e_y\| > 0.4` | → recovery (phase 3) |
| `no_progress` | scale range < 0.01 over 20 steps (phases 1/2) | → abort (truncated) |
| `unsafe_insert` | insert executed without alignment | logged, no abort |

---

## 10. Run-Level Metrics

`Evaluator.evaluate()` returns `RunMetrics` with these keys (also saved as JSON):

```python
{
    "n_episodes":             int,
    "access_success_rate":    float,   # ↑ primary metric
    "mean_steps_to_access":   float,   # ↑ efficiency (successful eps only)
    "mean_insert_attempts":   float,   # ↓ insertion efficiency
    "target_loss_rate":       float,   # ↓ proportion of eps with any target_loss
    "off_axis_rate":          float,   # ↓ proportion of eps with any off_axis
    "recovery_rate":          float,   # proportion of eps with ≥1 recovery
    "abort_rate":             float,   # ↓ proportion of eps that timed out
    "mean_alignment_error":   float,   # ↓ mean √(e_x²+e_y²) across all steps
}
```

Eval convention: `n_episodes=100`, cycling `["easy","medium","hard"]` difficulty.

---

## 11. Evaluator Usage

```python
from src.eval.evaluator import Evaluator
from src.controllers.postprocessor import ActionPostprocessor

ev = Evaluator(output_dir="outputs/b0_scripted", save_video=False)
pp = ActionPostprocessor()

metrics = ev.evaluate(
    env=env,
    policy=policy.act,       # callable: obs_dict -> np.ndarray[3]
    n_episodes=100,
    postprocessor=pp.process,
    split="val",
)
# Saves: outputs/b0_scripted/val/val_run_metrics.json
#        outputs/b0_scripted/val/val_episodes.csv
```

---

## 12. Per-Episode CSV Schema

`{output_dir}/{split}/{split}_episodes.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | str | Unique episode identifier |
| `success` | int | 1=success, 0=failure |
| `steps` | int | Total steps taken |
| `mean_alignment_error` | float | Mean √(e_x²+e_y²) |
| `num_insert_attempts` | int | Count of insert_executed=True |
| `target_loss_count` | int | Steps with target_loss |
| `off_axis_count` | int | Steps with off_axis |
| `recovery_count` | int | Recovery episodes entered |
| `mean_conf` | float | Mean confidence |
| `mean_readiness` | float | Mean readiness |

---

## 13. Baseline Compliance

| Baseline | Uses FSM | Uses Postprocessor | Obs used |
|----------|----------|-------------------|----------|
| B0 Scripted | ✅ | ✅ | `state` only |
| B1 Seg+PID | ✅ | ✅ | `state` only |
| B2 PPO-State | ✗ | ✗ | `state` + `prev_action` (9D) |
| B3 w/o Gate | ✅ | ✅ | `state` only |
| Ours V2 | ✅ | ✅ | `rgb` + `state` |

---

## 14. File Locations

| Component | Path |
|-----------|------|
| Interface ABC | `src/envs/base_env.py` |
| ToyAccessEnv | `src/envs/toy_access_env.py` |
| IsaacAccessEnv | `src/envs/isaac_access_env.py` *(Week 4)* |
| AccessState dataclass | `src/perception/access_state.py` |
| PhaseManager FSM | `src/gating/phase_manager.py` |
| InsertionGate | `src/gating/insertion_gate.py` |
| ActionPostprocessor | `src/controllers/postprocessor.py` |
| Evaluator | `src/eval/evaluator.py` |
| StepLogger / EpisodeLogger | `src/logging/logger.py` |
| Scripted policy (B0) | `src/policies/scripted_policy.py` |
| PID policy (B1) | `src/policies/pid_policy.py` |
| PPO policy (B2) | `src/policies/ppo_state_policy.py` |
