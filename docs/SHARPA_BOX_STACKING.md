# Same-footprint material stacking

This CIS6280 environment has **four objects with 55 × 55 mm square
footprints**: two 55 mm-high cubes and two 85 mm-high cuboids. It uses
rigid-body MuJoCo physics, a 3-DoF wrist stage with the 22-joint Sharpa hand,
a 25-action interface, a 180-second decision budget, and a verified stable-height score.
For the four-box visual world-model assignment built on the variable-footprint extension (`VariedBoxEnv`), see [`../assignment/README.md`](../assignment/README.md).

## Scene and reference policy

Height classes are shuffled across four stock positions. XY positions have
±6 mm jitter; initial yaw is zero. Every scene includes blue, orange and purple,
plus one independently sampled repeat. Material assignments are shuffled
independently of height. Colours have the existing fixed synthetic definitions:

| Material | Density (kg/m³) | Sliding friction parameter |
| --- | ---: | ---: |
| Blue | 600 | 0.45 |
| Orange | 950 | 0.90 |
| Purple | 1300 | 0.28 |

Mass is density times the object's volume. Contact-pair friction remains the
geometric mean of the two surfaces. These are synthetic rigid materials; colour
does not imply calibrated real wood, metal or rubber. Lighting/table appearance
also vary with the seed. No size or yaw randomization is used in this pilot.

`BoxStackController` uses measured poses and contacts. It places short boxes first,
with heavier boxes first within each height class. It uses the 55 mm grasp profile
with an extra 0.1 rad of flexion, or 0.2 rad for purple material, clipped to joint
limits. This small material-conditioned adjustment was calibrated on development
scenes. It is a privileged scripted demonstrator, not an optimal or learned policy.
Object positions, velocities, forces and material parameters are never overwritten
during control. No welding, attachment constraints or mid-episode resets are used.

```bash
# Native viewer on macOS (fixed camera)
.venv/bin/mjpython scripts/play_box_stacking.py --seed 1000 --speed 3
# Physics only, no window
.venv/bin/python scripts/play_box_stacking.py --seed 1000 --headless
```

## Task score versus dataset success

The task still scores final verified height. A single tall object can legitimately
earn a positive score. The **success dataset uses a stricter admission rule**:
all four objects belong to the final stable component, and its height is at least
95% of the sum of initial upright heights (266 mm of the nominal 280 mm).
This prevents a connected pile of fallen boxes from entering as a completed tower.

At STOP or deadline, version 2 computes enough bounded actions to lift and
park the hand, then runs a 3.1-second hold outside the decision budget.
The archived version 1 uses a fixed 4-second withdrawal. The final score uses the last 31 samples spanning
3 seconds, requiring persistent support contacts, low velocities, low drift,
no hand/ground contact and valid in-zone table support (see
[Stable support score](#stable-support-score)).

## Stable support score

An object counts as supported by the object (or table) below it when their contact
compression exceeds 0.0001 N, the upward normal component exceeds 0.5, and their
centres are vertically ordered. These are explicit task conventions evaluated at
10 Hz, not a general proof of structural stability.

Scoring requires 31 samples spanning three seconds, angular speed below 0.1 rad/s,
linear speed below 0.02 m/s, bounds drift below 5 mm, no hand/ground contact, and
in-zone table roots. Upward support edges must persist throughout the window, and
every lower supporter must itself be accepted, so removing an invalid supporter
cannot grant credit to an unsupported upper block. Side contacts neither combine
towers nor invalidate a stable tower. An unstable top can be excluded while its
stable base keeps earning height. Separate towers are scored separately, and only
the tallest accepted component counts. The privileged NPZ records these relations as
`object_support[T+1,N,N]` (lower → upper) and `object_table_support[T+1,N]`.

## Collection, video and exact replay

```bash
.venv/bin/python scripts/collect_box_dataset.py \
  --out datasets/box-smoke --successes 4 --start-seed 1000 \
  --resolution 128 --workers 4 --render-workers 2
```

The collector runs real physics, records complete numerical rollouts and lossless RGB arrays (`observations.npz` and `privileged.npz`), and writes a preview MP4 with the front and side views side by side. The MP4 is for viewing; use NPZ arrays for training because video compression is lossy.

Every recorded trajectory is simulated again from its initial state using only recorded actions to verify exact state and observation replay on the recording platform. Execution errors retain a valid numerical prefix and `error.json` and are excluded from training.

## Dataset structure and alignment

```text
manifest.json                 selected episodes, all attempts, checksums, splits
validation.json               full action replay and sampled RGB checks
rollouts/seed-001000/
    observations.npz          student inputs, recorded actions, time and masks
    privileged.npz            simulator ground truth and replay state
    metadata.json             scene, controller, outcome, units/scales, hashes
    preview.mp4               front | side, 10 FPS, real time
    record.json               completion/checksum record used for resumption
```

Only IDs listed in `manifest.episodes` belong to the success dataset. Failed or
surplus attempts remain in `manifest.attempts` and their numerical rollout folders;
they have no video/RGB unless selected. Do not glob all folders as training examples.

For **T actions**, all state/image fields have **T+1 observations**:

```text
observation[t] -- action[t], 0.1 simulated seconds --> observation[t+1]
```

The physical integrator uses 2 ms steps, 50 per action. Labels and images are sampled
at action boundaries, not every physics substep. The first image precedes action 0.
The last image is the post-verification terminal state. Video frame k corresponds
to observation k, so the video also has T+1 frames.

### observations.npz

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `rgb` | `(T+1, 2, 128, 128, 3)`, uint8 | Lossless front/side RGB, channels last |
| `proprio` | `(T+1, 75)`, float32 | 25 joint positions, 25 velocities, 25 actuator targets |
| `actions` | `(T, 25)`, float32 | Recorded normalized actuator-target increments in [-1,1] |
| `applied_target_delta` | `(T, 25)`, float64 | Actual target change after actuator-limit clipping |
| `agent_action_mask` | `(T,)`, bool | True for policy control, false for automatic verification |
| `timestamps` | `(T+1,)`, float64 | Simulated seconds since settled episode start |
| `time_remaining` | `(T+1,)`, float32 | Remaining decision budget; frozen during finalization |
| `stop_requested` | `(T+1,)`, bool | STOP decision at the control-end observation; false on deadline |
| `terminated` / `truncated` | `(T,)`, bool | Last transition marks agent STOP / time-limit termination |
| `goal_xy` | `(2,)`, float32 | Build-zone centre in world metres |

The first three action dimensions change XYZ wrist targets, scaled by **0.015 m**;
the remaining 22 change finger joint targets, scaled by **0.15 rad**. These are
neither torques nor measured joint displacements. The exact update is:

```python
target_next = clip(target + action * action_scale, actuator_low, actuator_high)
```

Joint ordering, scales and limits are in metadata. Proprioception units are metres
for wrist positions, radians for finger angles, and their per-second velocities.
`applied_target_delta` preserves target-clipping effects without conflating them
with the commanded action or actual physical joint motion.

For behaviour cloning, train on `agent_action_mask == True`; train the discrete
STOP decision from `stop_requested`. Automatic withdrawal actions are recorded for
complete physical replay but are not demonstrations of student policy decisions.

### privileged.npz: separate supervision / evaluation sidecar

| Field | Shape | Meaning |
| --- | --- | --- |
| `object_pose` | `(T+1, 4, 7)` | World XYZ metres followed by quaternion **w,x,y,z** |
| `object_velocity` | `(T+1, 4, 6)` | World angular velocity first, then linear velocity |
| `object_bounds` | `(T+1, 4, 2, 3)` | World-space lower/upper bounding corners |
| `object_contacts` | `(T+1, 4, 4)` | Object–object contact counts |
| `object_table_contacts` / `object_ground_contacts` / `object_hand_contacts` | `(T+1, 4)` | Contact counts with each surface/group |
| `object_hand_normal_force` | `(T+1, 4, 6)` | Summed normal forces, N: thumb/index/middle/ring/pinky/palm |
| `stable_height_m` | `(T+1,)` | Verified height from the preceding three-second window |
| `stable_component_mask` | `(T+1, 4)` | Members of that scored component |
| `phases` | `(T,)` | Controller/action phase, e.g. lift, transport, release |
| `integration_state` | `(T+1, S)` | Full MuJoCo integration state for exact replay |

The stability fields are **windowed labels**, not instantaneous tower height. They
are zero before sufficient history exists and while the structure remains held.
Episode success, final height, controller phase transitions and reasons for stopping
are in metadata. Static shapes, dimensions and materials are also there; masses
are derived from each volume and the material catalogue's density. Version 2
also saves compiled per-object physics and explicit contact parameters in
`metadata.physics`; inherited unused cube parameters are separated from the scene.

These fields are simulator ground truth. Actions are **demonstrator action labels**,
not unique optimal-action ground truth. A successful episode does not establish
that each action is optimal. Ground-truth simulator fields are excluded by default
from the student loader and should not silently become policy inputs.

```python
import json
from pathlib import Path
from cube_pick.box_data import load_episode

root = Path('datasets/box-material-success-100-v0.1')
manifest = json.loads((root/'manifest.json').read_text())
train_ids = set(manifest['splits']['train'])
record = next(r for r in manifest['episodes'] if r['episode_id'] in train_ids)
data = load_episode(root, record)  # no privileged sidecar
t = 0
obs_t = (data['rgb'][t], data['proprio'][t])
action_t = data['actions'][t]
obs_next = (data['rgb'][t+1], data['proprio'][t+1])
valid_bc_step = data['agent_action_mask'][t]
# Optional dynamics supervision: load_episode(root, record, include_privileged=True)
```

## Intended use and limitations

The 100 selected trajectories are split **80 train / 10 validation / 10 test** by
deterministic seed-disjoint assignment. All splits are success-filtered, public
development data from the same scene family and demonstrator. They do not measure
general task success rate, novel policy behaviour, or hidden-test generalization.

Use this pilot for imitation learning, data-loader development and initial dynamics
prediction. Success-only data is insufficient for learning calibrated failure risk
or recovery. Additional deliberate failures, perturbed actions, alternative orders,
partial stacks and interaction data are needed for those goals. The demonstrator
always uses one height ordering, and dimensions/yaw are fixed in this version.
