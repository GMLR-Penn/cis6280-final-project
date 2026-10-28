# Variable-footprint boxes and paired interventions

This extension retains the four-box, two-height, three-material
task (`SHARPA_BOX_STACKING.md`) while varying square footprint sizes across `{50, 55, 60} mm` and recording all physical outcomes. It serves as the underlying simulator environment (`VariedBoxEnv`) for the [CIS 6280 latent world-model assignment](../assignment/README.md).

Scoring follows the [stable support score](SHARPA_BOX_STACKING.md#stable-support-score) of the fixed-footprint task.

## Geometry and physics

Each scene contains square bases of 50, 55 and 60 mm, plus one randomly repeated
size. Assignments are shuffled independently of height and material. Heights
remain two 55 mm and two 85 mm objects. Length and width change together; this
version does not independently randomize rectangular aspect ratio or yaw.
Initial XY jitter, fixed colour/material mapping, 180-second budget, action
interface and final stable-height evaluator are inherited from the box task.

Collision geometry, volume, mass and inertia change with size. Material density
stays fixed, so size and mass effects are coupled. This is not an experiment
that isolates support area while holding mass constant. Every scene still has
the same nominal upright total height, 280 mm.

The controller interpolates the existing size-conditioned grasp profiles. It
retains the fixed-box material closure margin for 50/55 mm footprints and uses
a 0.3 rad flexion margin for 60 mm footprints, clipped to actuator limits.
The wider-box margin was developed with isolated size/height/material tests.
The policy reads privileged geometry and contacts; a student visual policy
would instead use RGB and proprioception.

## Paired behaviours

Every scene is reset identically for each requested mode. All actions remain
normalized actuator-target increments; there is no state injection, teleporting,
external-force impulse, mid-episode reset or outcome override.

| Mode | Behaviour |
| --- | --- |
| `nominal` | Original short-first, heavier-first-within-height order, now with size-conditioned grasps |
| `shuffled_order` | A reproducible alternative permutation, guaranteed to differ from nominal |
| `offset` | Second placement aims 35–55 mm away from the support centre, in a sampled horizontal direction |
| `weak_grip` | Second grasp uses 45–65% of nominal open-to-closed finger motion |
| `early_release` | Second placement starts opening during transport, within 25–50 mm of the target XY, before lowering |

Offset and release distances and grip fractions are sampled reproducibly per
scene. The last three modes preserve the nominal first-placement prefix.
Perturbations may never trigger if the controller fails earlier; metadata and
the manifest explicitly record whether an intervention was applied. A weakened
grasp is a change to joint targets, not a prescribed force in newtons.
The `target_placement` parameter applies to `offset`, `weak_grip` and
`early_release`; `shuffled_order` changes the complete order from episode start.

Mode names are **not outcome labels**. An intervention can succeed; the nominal
controller can fail. `success` is the existing full-stack admission criterion
(all four objects in the final stable component and at least 266 mm tall).
`outcome_class` distinguishes `full_stack`, `partial_stack` (positive verified
height without full success), and `no_stable_height`. A partial stack is a valid
positive task score, not necessarily a collapsed tower. No generic failure flag
should be interpreted as a verified drop or collapse event.

## Run and collect

After [setup](SIMULATOR.md#installation-and-interactive-playback), from the repository root:

```bash
# Native macOS viewer; use python instead of mjpython on Linux.
.venv/bin/mjpython scripts/play_varied_boxes.py --seed 2000 --mode nominal --speed 3
.venv/bin/mjpython scripts/play_varied_boxes.py --seed 2000 --mode offset --speed 3

# Six scenes × five behaviours = 30 rollouts, including all failures.
.venv/bin/python scripts/collect_varied_box_dataset.py \
  --out datasets/varied-box-interventions-30-v0.2 \
  --scenes 6 --start-seed 2000 --resolution 128 --workers 4 --render-workers 2
```

No success-based resampling or filtering is performed. Numeric recording comes
first, then every valid simulation receives RGB and video and is replayed for
verification. Execution errors retain a numerical prefix and `error.json`, are
excluded from training episodes, and are not retried by `--resume`. `--modes` can select a subset. For example, a short
pipeline smoke test is:

```bash
.venv/bin/python scripts/collect_varied_box_dataset.py \
  --out datasets/varied-box-smoke --scenes 1 --modes nominal weak_grip \
  --time-limit 2 --resolution 64 --workers 1 --render-workers 1
```

That short test intentionally reaches the deadline before interventions; it
checks recording and failure handling, not representative stacking performance.

Use `--stage collect` for numerical recording without graphics. Continue with
the same arguments plus `--stage render --resume`, then `--stage validate --resume`.
`--resume` rejects changed configuration or source fingerprints. Use a new output
directory after changing the controller. Rendering needs a working native or
offscreen MuJoCo graphics backend as described in the simulator guide.

## Schema, splits and validation

The current schema is `varied-box-interventions-v0.2`. Episode IDs include both scene and
mode, for example `seed-002000-offset`. All attempted records live in `manifest.attempts`; valid simulations live in
`manifest.episodes`, including physical failures. There is no success-selected
subset. Execution errors are listed explicitly and yield `complete_with_errors`
after validation, not a complete requested dataset. Each episode includes:

```text
rollouts/seed-002000-offset/
    observations.npz    Two-view RGB, proprio, actions, timing and control masks
    privileged.npz      Poses, velocities, contacts, forces, height and replay state
    metadata.json       Scene, actual outcome, mode, intervention parameters/events
    preview.mp4         Two cameras side by side, real time at 10 FPS
    record.json         Completion and file checksums
```

The per-step arrays and units match the [fixed-box schema](SHARPA_BOX_STACKING.md#dataset-structure-and-alignment):
T actions align with T+1 observations, at 10 Hz; each RGB view is 128 × 128 by
default. Automatic terminal withdrawal remains distinguished by `agent_action_mask`.
The current loader is `cube_pick.varied_box_data.load_episode`; privileged fields
are excluded by default. Labels and intervention metadata should not be leaked
into a visual model's observation inputs.

**Split by scene, not trajectory.** All modes for a scene stay in the same split,
including their shared prefixes. With six scenes the deterministic split is
four training scenes, one validation scene and one test scene (20/5/5 episodes).
Larger collections use approximately 80/10/10 scene splits. One- or two-scene
smoke collections put all episodes in training. These tiny public splits are
for development and cannot establish generalization reliably.

Validation checks checksums, every replayed physical state and per-step label,
sampled RGB equality, video dimensions/FPS/frame count, and actual terminal
success **or failure**. Paired checks require identical initial physics and RGB
for all modes of a scene; they measure the common action prefix and require
identical physical states throughout that prefix. Exact replay is expected only
on the recording MuJoCo version/platform.

## Research scope

These trajectories add alternative orders, grasp failures, off-centre placement
and opening before contact. They still cover a small scripted action family,
upright square bases, a fixed wrist orientation and one inventory size. The
controller stops after an unstable placement; recovery is not implemented.
Intervention frequencies are chosen for coverage, not natural failure-rate
estimation. Add more scenes and local action branches before claiming broad
counterfactual or calibrated-risk prediction.
