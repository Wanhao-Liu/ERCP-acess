# Dataset Card: ERCP Phantom Biliary Access (ercp_phantom_v1)

## Overview

| Field | Value |
|-------|-------|
| Dataset name | `ercp_phantom_v1` |
| Version | v2.1 (LeRobot-compatible) |
| Task | Near-papilla autonomous biliary access via ERCP daughter scope |
| Robot type | `ercp_daughter_scope` (3-DOF cable-driven endoscope) |
| Data source | Physical phantom model, human expert demonstrations |
| Total episodes | 71 |
| Total frames | 36,297 |
| Sampling rate | 30 fps |
| Format | LeRobot dataset v2.1 (parquet + MP4) |
| Open-H status | Prepared for Open-H-Embodiment contribution (proposal pending) |

---

## Task Description

**Near-papilla autonomous biliary access** is the task of guiding an ERCP daughter scope from an initial position near the duodenal papilla to successful cannulation of the common bile duct.

The operator starts with the papilla visible in the endoscope field of view. The task requires:
1. **Alignment** — centering the scope on the papilla ostium
2. **Approach** — advancing while maintaining alignment
3. **Insertion** — controlled insertion into the bile duct when conditions are met
4. **Recovery** (if needed) — retreating and re-aligning after off-axis displacement

**Sole sensor**: monocular RGB endoscope camera. No depth sensor, no force/torque sensor, no external camera.

---

## Data Collection Protocol

- **Platform**: Physical biliary phantom model (ex-vivo compatible geometry)
- **Control**: Motor-controlled daughter scope with 3 active DOFs
- **Operator**: Expert endoscopist performing manual demonstrations
- **Recording**: Synchronized AVI video (1920×1080, 30 fps) + motor encoder CSV
- **Scope DOFs**:

| DOF | Motor | Active episodes |
|-----|-------|-----------------|
| Insertion / retraction | M1 | All (ep 1–71) |
| Pitch (up/down bending) | M3 | ep 11–71 |
| Yaw (left/right bending) | M4 | ep 1–13 |

---

## Dataset Statistics

### Episodes

| Split | Episodes | Frames |
|-------|----------|--------|
| Train | 57 | ~29,000 |
| Val | 14 | ~7,300 |
| **Total** | **71** | **36,297** |

Episode split: 80/20 by episode index, seed=42.

### Episode Duration

| Metric | Value |
|--------|-------|
| Mean duration | 17.0 s (511 frames) |
| Min duration | 7.3 s (219 frames) |
| Max duration | 78.7 s (2,360 frames) |

### Action Statistics (`[delta_insert, delta_pitch, delta_yaw]`)

| Dimension | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| delta_insert | 0.037 | 0.169 | −1.0 | 1.0 |
| delta_pitch  | −0.015 | 0.202 | −1.0 | 1.0 |
| delta_yaw    | 0.011 | 0.103 | −1.0 | 1.0 |

### Cumulative Motor State Statistics (`[cum_insert_deg, cum_pitch_deg, cum_yaw_deg]`)

| Dimension | Mean (°) | Std (°) | Min (°) | Max (°) |
|-----------|----------|---------|---------|---------|
| cum_insert_deg | 61.7 | 45.1 | −19.3 | 229.4 |
| cum_pitch_deg  | 11.9 | 128.6 | −284.9 | 285.5 |
| cum_yaw_deg    | 16.5 | 59.5 | −66.8 | 280.3 |

---

## Data Format

This dataset uses **LeRobot dataset format v2.1**.

### Directory Structure

```
ercp_phantom_v1/
├── meta/
│   ├── info.json          # Dataset metadata and feature schema
│   ├── episodes.jsonl     # Per-episode metadata (71 entries)
│   ├── tasks.jsonl        # Task description (1 entry)
│   └── stats.json         # Per-feature statistics
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...            # 71 parquet files total
└── videos/
    └── chunk-000/
        └── observation.images.endoscope/
            ├── episode_000000.mp4
            └── ...        # 71 MP4 files (512×512, 30fps)
```

### Parquet Schema

Each parquet file contains one row per frame:

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `observation.state` | float32 list | [3] | Cumulative motor angles relative to episode start (degrees) |
| `action` | float32 list | [3] | Normalized motor delta commands, clipped to [−1, 1] |
| `timestamp` | float64 | scalar | Time in seconds from episode start |
| `frame_index` | int64 | scalar | Frame index within episode (0-based) |
| `episode_index` | int64 | scalar | Episode index in dataset (0-based) |
| `index` | int64 | scalar | Global frame index across all episodes |
| `task_index` | int64 | scalar | Task index (always 0) |
| `next.done` | bool | scalar | True for the last frame of each episode |

### Video Format

- Resolution: 512 × 512 px (resized from 1920 × 1080 using INTER_AREA)
- Codec: mp4v
- Frame rate: 30 fps
- Color space: yuv420p

---

## Features

### `observation.state` — Cumulative Motor Proprioception

```
[cum_insert_deg, cum_pitch_deg, cum_yaw_deg]
```

Cumulative motor rotation (in degrees) relative to the start of each episode:
- `cum_insert_deg`: cumulative M1 encoder rotation → insertion depth proxy
- `cum_pitch_deg`: cumulative M3 encoder rotation → pitch angle proxy (0 for ep 1–10)
- `cum_yaw_deg`: cumulative M4 encoder rotation → yaw angle proxy (0 for ep 14–71)

> **Note**: This version does not include visual perception fields (e_x, e_y, scale, conf, stability, readiness). These will be added in v2 after the papilla detector (YOLOv8-nano) is trained on annotated data.

### `observation.images.endoscope` — Endoscope RGB Video

Single monocular endoscope view, 512×512, RGB, 30 fps. Stored as MP4 per episode (not embedded in parquet).

### `action` — Normalized Motor Delta Commands

```
[delta_insert, delta_pitch, delta_yaw] ∈ [−1, 1]³
```

Derived from raw motor encoder differences, normalized and clipped:

| Component | Source motor | Scale mapping |
|-----------|-------------|---------------|
| delta_insert | M1 differential | × insertion_scale (0.0025 m) |
| delta_pitch | M3 differential | × bend_scale (0.05 rad) |
| delta_yaw | M4 differential | × bend_scale (0.05 rad) |

---

## Embodiment Configuration

For use with GR00T-H or other VLA frameworks:

```json
{
  "robot_type": "ercp_daughter_scope",
  "action_space": {
    "type": "continuous",
    "dim": 3,
    "names": ["delta_insert", "delta_pitch", "delta_yaw"],
    "range": [-1.0, 1.0]
  },
  "state_space": {
    "type": "continuous",
    "dim": 3,
    "names": ["cum_insert_deg", "cum_pitch_deg", "cum_yaw_deg"]
  },
  "cameras": ["endoscope"],
  "dof": 3,
  "control_frequency_hz": 30
}
```

---

## Intended Uses

- **Behavior cloning / imitation learning** for biliary access policies
- **Visual representation learning** for endoscopic papilla detection
- **Sim-to-real transfer validation** for Isaac Sim–trained policies
- **VLA fine-tuning** (e.g., GR00T-H new embodiment adaptation)
- **World model training** for surgical endoscopy simulation

---

## Limitations and Out-of-Scope Uses

- **Not for clinical use.** This dataset is research-only, collected on a physical phantom, not from live patients.
- **Single phantom geometry.** The biliary anatomy is fixed. Generalization to diverse patient anatomies is not validated.
- **No force/torque data.** The dataset contains no haptic feedback signals.
- **Partial DOF coverage.** Pitch (M3) is only active from episode 11 onward; yaw (M4) only up to episode 13. Full 3-DOF data is available from episode 14 onward but with varying yaw ranges.
- **No perception labels in v1.** Visual state fields (e_x, e_y, scale, conf) are absent. Use `observation.images.endoscope` video directly for visual tasks.

---

## Related Resources

| Resource | Description |
|----------|-------------|
| `configs/yolo/papilla_dataset.yaml` | YOLOv8-nano training config for papilla detection |
| `src/perception/papilla_detector.py` | Trained detector wrapper |
| `src/perception/state_builder.py` | AccessState 6D state construction from RGB |
| `src/envs/toy_access_env.py` | 2D oracle simulation environment |
| `scripts/train/train_b2_ppo.py` | B2 PPO-State baseline training |

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{ercp_phantom_v1_2026,
  title   = {ERCP Phantom Biliary Access Dataset},
  author  = {[Authors]},
  year    = {2026},
  note    = {Prepared in LeRobot v2.1 format for Open-H-Embodiment contribution},
  url     = {https://github.com/[repo]}
}
```

---

## License

Research use only. Not for clinical or commercial deployment.
Contact the authors for data sharing agreements.

---

*Dataset version: v1.0 | Format: LeRobot v2.1 | Generated: 2026-03-24*
