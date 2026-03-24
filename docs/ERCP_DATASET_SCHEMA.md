# ERCP_DATASET_SCHEMA.md — ERCP Phantom Dataset Schema

> **Version**: v1.0 | **Format**: LeRobot v2.1 | **Last updated**: 2026-03-24

---

## 1. Dataset Identity

| Field | Value |
|-------|-------|
| Dataset name | `ercp_phantom_v1` |
| Task | Near-papilla autonomous biliary access |
| Robot type | `ercp_daughter_scope` (3-DOF cable-driven endoscope) |
| Source | Physical biliary phantom, human expert demonstrations |
| Total episodes | 71 |
| Total frames | 36,297 |
| Sampling rate | 30 fps |
| Exported format | LeRobot dataset v2.1 (parquet + MP4) |

---

## 2. Task Name and Label

### Task string (LeRobot `tasks.jsonl`)

```json
{"task_index": 0, "task": "Guide the endoscope to cannulate the bile duct via the papilla of Vater."}
```

### Phase labels (NOT stored in LeRobot parquet — annotation only)

| Phase index | Name | Description |
|-------------|------|-------------|
| 0 | `alignment` | Steering pitch/yaw to center papilla in FoV |
| 1 | `approach` | Advancing while maintaining alignment |
| 2 | `insertion` | Controlled insertion into bile duct |
| 3 | `recovery` | Retreating and re-aligning after displacement |

> Phase labels require CVAT annotation + YOLO training and are **not** present in the current v1 export. They will be added in v2 as `observation.phase` once the papilla detector (YOLOv8-nano, target mAP@50 ≥ 0.70) is trained.

### Success labels (NOT stored in v1)

Success is defined as: insertion depth exceeds threshold AND papilla was centered at time of insertion. In v1, episodes end at the human demonstration's natural endpoint (presumed successful). Explicit per-episode success labels will be added in v2 after annotation.

---

## 3. Raw Data Format (pre-export)

### Source files

| Type | Location | Format |
|------|----------|--------|
| Motor CSV | `/data/ERCP/ercp_access/data/cannulation/data/` | `ep{N:03d}.csv`, 30 Hz |
| Video | `/data/ERCP/ercp_access/data/cannulation/video/` | `ep{N:03d}.avi`, 1920×1080, 30 fps |

### CSV column indices (0-based)

| Index | Motor | DOF | Active episodes |
|-------|-------|-----|-----------------|
| 5 | M1 (`COL_M1_DEG`) | Insertion / retraction | All (ep 1–71) |
| 13 | M3 (`COL_M3_DEG`) | Pitch up/down (large knob) | ep 11–71 |
| 17 | M4 (`COL_M4_DEG`) | Yaw left/right (small knob) | ep 1–13 |

### Normalization scales (p99 per-frame delta, deg/frame)

| DOF | Scale |
|-----|-------|
| `INSERT_SCALE` | 4.5 deg/frame |
| `PITCH_SCALE` | 8.3 deg/frame |
| `YAW_SCALE` | 4.7 deg/frame |

Action = `clip(raw_delta / scale, -1.0, 1.0)`.

---

## 4. LeRobot v2.1 Export Schema

### Directory structure

```
data/lerobot/ercp_phantom_v1/
├── meta/
│   ├── info.json          # Dataset metadata and feature schema
│   ├── episodes.jsonl     # Per-episode metadata (71 entries)
│   ├── tasks.jsonl        # Task description (1 entry)
│   └── stats.json         # Per-feature statistics (mean/std/min/max)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...            # 71 parquet files total
└── videos/
    └── chunk-000/
        └── observation.images.endoscope/
            ├── episode_000000.mp4
            └── ...        # 71 MP4 files
```

### Parquet row schema (one row per frame)

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `observation.state` | float32 list | [3] | Cumulative motor angles relative to episode start (degrees): `[cum_M1_deg, cum_M3_deg, cum_M4_deg]` |
| `action` | float32 list | [3] | Normalized motor deltas clipped to [-1, 1]: `[delta_insert, delta_pitch, delta_yaw]` |
| `timestamp` | float64 | scalar | Time in seconds from episode start (= `frame_index / 30.0`) |
| `frame_index` | int64 | scalar | Frame index within episode (0-based) |
| `episode_index` | int64 | scalar | Global episode index (0-based) |
| `index` | int64 | scalar | Global frame index across all episodes |
| `task_index` | int64 | scalar | Task index (always 0) |
| `next.done` | bool | scalar | True for the last frame of each episode |

### `observation.state` — cumulative motor proprioception

```python
[cum_M1_deg, cum_M3_deg, cum_M4_deg]
# = cumulative encoder rotation in degrees, relative to episode frame 0
```

| Component | Motor | Proxy for | Note |
|-----------|-------|-----------|------|
| `cum_M1_deg` | M1 | Insertion depth | Active all episodes |
| `cum_M3_deg` | M3 | Pitch angle | 0.0 for ep 1–10 (motor inactive) |
| `cum_M4_deg` | M4 | Yaw angle | 0.0 for ep 14–71 (motor inactive) |

> Note: This is **NOT** the same as `AccessState`. The LeRobot state is raw motor proprioception; AccessState (`e_x, e_y, scale, conf, stability, readiness`) is the visual perception-derived 6D control state.

### `action` — normalized motor delta commands

```python
[delta_insert, delta_pitch, delta_yaw] ∈ [-1.0, 1.0]³
```

| Component | Source | Normalization |
|-----------|--------|---------------|
| `delta_insert` | M1 frame delta | ÷ INSERT_SCALE (4.5 deg/frame), clipped |
| `delta_pitch` | M3 frame delta | ÷ PITCH_SCALE (8.3 deg/frame), clipped |
| `delta_yaw` | M4 frame delta | ÷ YAW_SCALE (4.7 deg/frame), clipped |

### Video format

| Property | Value |
|----------|-------|
| Resolution | 512 × 512 px (resized from 1920×1080 via cv2.INTER_AREA) |
| Codec | `mp4v` |
| Frame rate | 30 fps |
| Color space | YUV420p |
| Storage | One MP4 per episode, named `episode_{N:06d}.mp4` |

---

## 5. Train / Validation Split

| Split | Episodes | Frames | Episode range |
|-------|----------|--------|---------------|
| `train` | 57 | ~29,000 | 0–56 (sorted, seed=42 80/20 split) |
| `val` | 14 | ~7,300 | 57–70 |
| **Total** | **71** | **36,297** | — |

Split rule: stratified random by episode index, 80/20, `random_state=42`.
Train episodes are assigned `episode_index` 0–56; val episodes 57–70 in the exported dataset.

---

## 6. Dataset Statistics (from `stats.json`)

### Action

| Dimension | Mean | Std |
|-----------|------|-----|
| `delta_insert` | 0.037 | 0.169 |
| `delta_pitch` | −0.015 | 0.202 |
| `delta_yaw` | 0.011 | 0.103 |

### Observation state (cumulative motor degrees)

| Dimension | Mean (°) | Std (°) | Min (°) | Max (°) |
|-----------|----------|---------|---------|---------|
| `cum_M1_deg` | 61.7 | 45.1 | −19.3 | 229.4 |
| `cum_M3_deg` | 11.9 | 128.6 | −284.9 | 285.5 |
| `cum_M4_deg` | 16.5 | 59.5 | −66.8 | 280.3 |

### Episode duration

| Metric | Value |
|--------|-------|
| Mean duration | 17.0 s (511 frames) |
| Min duration | 7.3 s (219 frames) |
| Max duration | 78.7 s (2,360 frames) |

---

## 7. `meta/info.json` Schema

```json
{
  "codebase_version": "v2.1",
  "robot_type": "ercp_daughter_scope",
  "total_episodes": 71,
  "total_frames": 36297,
  "fps": 30,
  "splits": {"train": "data/chunk-000/episode_000000.parquet...episode_000056.parquet",
             "val":   "data/chunk-000/episode_000057.parquet...episode_000070.parquet"},
  "features": {
    "observation.state": {"dtype": "float32", "shape": [3], "names": ["cum_M1_deg", "cum_M3_deg", "cum_M4_deg"]},
    "action":            {"dtype": "float32", "shape": [3], "names": ["delta_insert", "delta_pitch", "delta_yaw"]},
    "observation.images.endoscope": {"dtype": "video", "shape": [512, 512, 3], "video_info": {"fps": 30.0, "codec": "mp4v"}},
    "timestamp":   {"dtype": "float64", "shape": []},
    "frame_index": {"dtype": "int64",   "shape": []},
    "episode_index": {"dtype": "int64", "shape": []},
    "index":       {"dtype": "int64",   "shape": []},
    "task_index":  {"dtype": "int64",   "shape": []},
    "next.done":   {"dtype": "bool",    "shape": []}
  }
}
```

---

## 8. `meta/episodes.jsonl` Schema

One JSON line per episode:

```json
{"episode_index": 0, "tasks": ["Guide the endoscope to cannulate the bile duct via the papilla of Vater."], "length": 511}
```

Fields: `episode_index` (int), `tasks` (list[str] of length 1), `length` (int, frames in episode).

---

## 9. Export Script

```bash
cd /data/ERCP/ercp_access
conda run -n ercp python scripts/data/export_lerobot_ercp.py          # full export
conda run -n ercp python scripts/data/export_lerobot_ercp.py --dry_run  # 3-episode test
```

Output: `data/lerobot/ercp_phantom_v1/`

---

## 10. Planned v2 Additions

| Field | Description | Prerequisite |
|-------|-------------|--------------|
| `observation.phase` | Phase label (0–3) per frame | YOLO-based perception + manual validation |
| `observation.images.papilla_bbox` | Bounding box [cx, cy, w, h] | CVAT annotation complete |
| `success` | Per-episode success label | Human annotation of episode outcomes |
| `readiness` | Computed readiness score per frame | PerceptionParser + StateBuilder v1 |

---

## 11. Limitations

- **No force/torque data** — endoscopic tissue contact forces are not recorded.
- **Single phantom geometry** — fixed biliary anatomy; patient anatomy variability not represented.
- **Partial DOF coverage** — pitch (M3) inactive ep 1–10; yaw (M4) inactive ep 14–71.
- **No perception labels in v1** — `e_x, e_y, scale, conf` fields absent; use raw video for visual tasks.
- **Research use only** — not for clinical deployment.

---

## 12. Related Files

| File | Description |
|------|-------------|
| `src/data/phantom_dataset.py` | Raw data loader (AVI + CSV) |
| `scripts/data/export_lerobot_ercp.py` | LeRobot v2.1 export script |
| `docs/dataset_card.md` | Full Open-H-compatible dataset card |
| `configs/yolo/papilla_dataset.yaml` | YOLOv8-nano training config |
| `src/perception/state_builder.py` | AccessState 6D construction from RGB |
