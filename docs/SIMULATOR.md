# Simulator Guide

This document describes the MuJoCo box-stacking simulator used in the [CIS 6280 course assignment](../assignment/README.md). A 22-joint Sharpa Wave five-finger hand mounted on a 3-DoF wrist stage grasps free rigid boxes from a table and stacks them inside a target goal zone.

## Scene and Task

- **Objects** — Four rigid boxes per scene (two 55 mm tall and two 85 mm tall) with square footprints in `{50, 55, 60} mm` (`VariedBoxEnv`, used by the assignment) or fixed `55 mm` footprints (`BoxStackEnv`, where the two short boxes are exact cubes), shuffled across four starting positions with small XY jitter and fixed initial yaw.
- **Materials** — Blue (`600 kg/m³`, friction `0.45`), orange (`950 kg/m³`, friction `0.90`), and purple (`1300 kg/m³`, friction `0.28`) identify synthetic density and sliding-friction combinations. Each scene includes all three materials plus one randomly repeated material.
- **Control rate** — A fixed-orientation XYZ wrist stage and 22 finger joints controlled at `10 Hz` (each control step advances 50 MuJoCo physics substeps of `2 ms`).
- **Horizon and scoring** — In the four-box assignment (`StackTaskEnv`), an episode runs for up to `1795` control steps (`179.5 s`) and tracks the supported tower in the goal zone (`placed` in `{0, 1, 2, 3, 4}`). Objects move strictly through MuJoCo contact physics; the simulator never teleports objects, welds them to the hand, or overwrites their velocities.

Detailed material parameters, support-graph scoring rules, and dataset schemas are documented in [`SHARPA_BOX_STACKING.md`](SHARPA_BOX_STACKING.md) and [`SHARPA_VARIED_BOXES.md`](SHARPA_VARIED_BOXES.md).

## Installation and Interactive Playback

Run from the repository root. Setup creates a virtual environment, installs dependencies, and downloads the Sharpa hand assets (`Python 3.12` is the default interpreter; override with `PYTHON_BIN=python3.11 bash scripts/setup.sh`):

```bash
bash scripts/setup.sh

# macOS: mjpython is required for the interactive MuJoCo viewer.
.venv/bin/mjpython scripts/play_varied_boxes.py --seed 1000 --speed 3

# Physics-only rollout; no graphics context or window required.
.venv/bin/python scripts/play_varied_boxes.py --seed 1000 --headless

# Simulator and assignment unit tests.
.venv/bin/python -m unittest discover -s tests -v
```

On Linux, use `.venv/bin/python` instead of `mjpython` for interactive playback. For offscreen RGB rendering on headless Linux servers, set `export MUJOCO_GL=egl` (GPU) or `export MUJOCO_GL=osmesa` (CPU software rendering, as in [`.github/workflows/smoke.yml`](../.github/workflows/smoke.yml)).

## Observation and Action Interface

| Value | Shape | Meaning |
| --- | --- | --- |
| `rgb` | `(2, 128, 128, 3)` | Front and side wrist cameras (`uint8`) |
| `proprio` | `(75,)` | 25 joint positions, 25 joint velocities, and 25 actuator targets |
| `time_remaining` | scalar | Remaining decision time in seconds |
| `goal_xy` | `(2,)` | Goal-zone center in world meters |
| `action` | `(25,)` | Normalized target increments in `[-1, 1]` |

The first 3 action entries command wrist XYZ target increments (up to `15 mm` per control step), and the remaining 22 entries command finger joint target increments (up to `0.15 rad` per step), clipped to actuator limits. Object poses, contact forces, and material parameters are privileged simulator state and are not exposed to visual student policies or planners.

In the [course assignment](../assignment/README.md), observations are rendered every `STRIDE = 5` control steps (`2 Hz`), the two `128 × 128` camera views are tiled horizontally into a single `(128, 256, 3)` RGB frame, and the 5 consecutive 25-D actions between frames are flattened into a 125-D action block.

```python
from assignment.training.data.task import StackTaskEnv, TaskConfig
import numpy as np

env = StackTaskEnv(TaskConfig(target_count=4, horizon=1795, terminate_on_ground=False), seed=100038)
env.reset(('init', 100038))
env.env.resolution = 128
try:
    frame = np.concatenate(list(env.env.render()), axis=1)  # (128, 256, 3) uint8 tiled front|side
    obs, reward, terminated, truncated, info = env.step(np.zeros(25, dtype=np.float32))
    print(frame.shape, info)
finally:
    env.close()
```
