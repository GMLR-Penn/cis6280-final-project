# Training Track

This directory contains the five-stage pipeline that produced the checkpoints used in the [planning track](../planning/README.md): data collection, world-model pretraining, policy training, world-model finetuning, and evaluation. Every stage has a pretrained artifact on Hugging Face, so you can start at whichever stage you want to study or modify.

All commands below should be run from the repository root after setting up the local environment (`bash scripts/setup.sh`, `source .venv/bin/activate`, and `export MUJOCO_GL=egl`).

<details>
<summary><b>Stage 1: Dataset</b> (Can skip by downloading our dataset. Click to expand.)</summary>

We record 20,000 episodes of a behavior-cloning teacher that sees the full simulator state. A world model trained only on clean demonstrations never sees failed grasps or slips, so it cannot judge bad actions at planning time. Each episode therefore adds Gaussian noise of a fixed level $\sigma$ to the teacher's actions ($\sigma = 0$ in 10% of episodes, otherwise between 0.005 and 0.2), and we record the noisy actions that were actually executed.

To skip data collection, download the prepared dataset from [`cis6280-stack-datasets-v2`](https://huggingface.co/datasets/YongYong/cis6280-stack-datasets-v2):

```bash
# If you only need Stage 3 (policy training), Stage 4 (on-policy finetuning), or Stage 5 (evaluation),
# you can skip the 40 raw episode shards (~190 GB) and download only the derived caches and goals (~13 GB):
hf download YongYong/cis6280-stack-datasets-v2 --repo-type dataset --local-dir dataset \
  --include "derived/*" "onpolicy-smwm-c3/*" "goals/*"

# Or download the entire dataset repository (~200 GB), including all 20,000 raw episode shards:
hf download YongYong/cis6280-stack-datasets-v2 --repo-type dataset --local-dir dataset
```

The dataset repository contains:

- `bc-noise-4cube-20k/` — the 20,000 episodes (train / val / test split by scene), in 40 tar shards.
- `derived/jepa-cache-4cube-20k/` — an episode index and the recorded actions, grouped into blocks of 5 control steps.
- `derived/latents-smwm-c3/` — every frame already encoded by our world model, so **you can train a policy (Stage 3) without the raw images**.
- `onpolicy-smwm-c3/` — rollouts of the released policy, used in Stage 4.
- `goals/` — subgoal images for the 300 benchmark scenes, plus a 100-scene version used by Colab.

The raw image file (`frames.u8`, about 722 GB) is not on Hugging Face. You only need to build it if you want to pretrain your own world model in Stage 2.

<details>
<summary>How to collect the dataset from scratch</summary>

Download the teacher checkpoint, collect the episodes, and build the cache:

```bash
hf download YongYong/cis6280-stack-world-models-v2 bc/step-050000.pt --local-dir checkpoints

python -m assignment.training.data.collect --out bc-noise-4cube-20k --source bc \
  --checkpoints checkpoints/bc/step-050000.pt --execute 5 --stop-count 4 --max-steps 1795 \
  --rgb-every 5 --conditions 'iid:0.005-0.2,log,z=0.1' --scenes 20000 --start-seed 400000

python -m assignment.training.data.cache --data bc-noise-4cube-20k --out cache-4cube-20k
```

</details>

</details>

<details>
<summary><b>Stage 2: World Model</b> (Can skip by downloading <code>jepa/smwm-c3/pretrain.pt</code>. Click to expand.)</summary>

The world model (`StackJEPA` in [`jepa.py`](training/jepa.py)) has an image encoder that turns each camera frame into a 192-D latent $z$, and a predictor that takes the last 3 latents and the actions in between and predicts the next latent. It is trained to make that prediction accurate:
$$\mathcal{L} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2 + \lambda \, \|\text{InvDyn}(z_t, z_{t+1}) - a_t\|_2^2.$$
The second term asks a small network to recover the action from two consecutive latents. Without it, the encoder could make prediction trivially easy by mapping every image to the same latent. (`--method lewm` swaps it for the SIGReg regularizer from [LeWM](https://github.com/lucas-maes/le-wm).)

To skip this stage, download `pretrain.pt`, which is the output of this stage (`model.pt` is the same model after Stage 4):

```bash
hf download YongYong/cis6280-stack-world-models-v2 jepa/smwm-c3/pretrain.pt --local-dir checkpoints
```

<details>
<summary>How to pretrain the world model from scratch</summary>

Pretraining reads raw images, so if you downloaded the dataset, first unpack the episode shards and build the image cache. Then train on 2 GPUs:

```bash
cd dataset/bc-noise-4cube-20k && for f in episodes-*.tar; do tar xf "$f"; done && cd ../..

python -m assignment.training.data.cache --data dataset/bc-noise-4cube-20k --out cache-4cube-20k

torchrun --nproc_per_node 2 -m assignment.training.training.train_jepa \
  --cache cache-4cube-20k --out smwm-c3 --method smwm --context 3 --batch 256 --steps 145000
```

</details>

</details>

<details>
<summary><b>Stage 3: Policy</b> (Can skip by downloading <code>policy/df-smwm-c1-sigma04/policy.pt</code>. Click to expand.)</summary>

The policy (`DFPolicy` in [`policy.py`](training/policy.py)) looks at the current latent and generates the next 16 action blocks (8 s) with [Diffusion Forcing](https://arxiv.org/abs/2407.01392), a diffusion model in which each block gets its own noise level. Noisy demonstrations help the world model but would teach the policy jittery motions, so the policy only trains on episodes with $\sigma \le 0.04$ (`--max-sigma 0.04`).

To skip this stage, download the trained policy:

```bash
hf download YongYong/cis6280-stack-world-models-v2 policy/df-smwm-c1-sigma04/policy.pt --local-dir checkpoints
```

<details>
<summary>How to train the policy</summary>

With the downloaded `derived/` folders, you can train the policy on one GPU right away:

```bash
python -m assignment.training.training.train_policy \
  --cache dataset/derived/jepa-cache-4cube-20k \
  --latents dataset/derived/latents-smwm-c3 \
  --out df-smwm-c1-sigma04 --context 1 --max-sigma 0.04 --steps 200000
```

If you trained your own world model in Stage 2, encode the images with it first (use `--model checkpoints/jepa/smwm-c3/pretrain.pt` instead if you downloaded it):

```bash
python -m assignment.training.data.encode \
  --model smwm-c3/model.pt --cache cache-4cube-20k --out latents-smwm-c3

python -m assignment.training.training.train_policy \
  --cache cache-4cube-20k --latents latents-smwm-c3 \
  --out df-smwm-c1-sigma04 --context 1 --max-sigma 0.04 --steps 200000
```

</details>

</details>

<details>
<summary><b>Stage 4: On-policy predictor finetuning</b> (Can skip by downloading <code>jepa/smwm-c3/model.pt</code>. Click to expand.)</summary>

The Stage 2 world model only learned one-step predictions on the teacher's actions, but the planner needs 16-step predictions for the policy's own action chunks. In this stage, we run the policy, execute each of its 16 candidate chunks in the simulator from the same saved state, and finetune the predictor so that its 16-step predictions match what actually happened. The image encoder stays frozen, so the latents used by the policy and the subgoals do not change.

To skip this stage, download the finetuned world model:

```bash
hf download YongYong/cis6280-stack-world-models-v2 jepa/smwm-c3/model.pt --local-dir checkpoints
```

<details>
<summary>How to finetune the predictor (and optionally collect fresh on-policy rollouts)</summary>

With the downloaded rollouts, goals, and `pretrain.pt`, finetune on one GPU:

```bash
python -m assignment.training.training.finetune_onpolicy \
  --data dataset/onpolicy-smwm-c3 --init checkpoints/jepa/smwm-c3/pretrain.pt \
  --goals dataset/goals/scripted-val300-every80/goals.npz --out smwm-c3-onpolicy
```

If you trained a new policy in Stage 3, collect rollouts of your own policy first and finetune on them (if you also pretrained your own world model in Stage 2, use `smwm-c3/model.pt` for `--world-model` and `--init`):

```bash
python -m assignment.training.data.collect_onpolicy \
  --policy df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/pretrain.pt \
  --goals dataset/goals/scripted-val300-every80/goals.npz \
  --out onpolicy-mine --gpus 0,1,2,3 --workers 80 --val-workers 14 --train-scenes 4000

python -m assignment.training.training.finetune_onpolicy \
  --data onpolicy-mine --init checkpoints/jepa/smwm-c3/pretrain.pt \
  --goals dataset/goals/scripted-val300-every80/goals.npz --out smwm-c3-onpolicy-mine
```

During finetuning, the script checks how often the world model picks the same candidate as the simulator would (the one ending closest to the subgoal). After 60,000 steps this agreement rises from 70.2% (`pretrain.pt`) to 77.6% (`model.pt`).[^val-subset]

[^val-subset]: These numbers were logged on the 1,152 validation states that had been written when our finetuning run started; across all 6,208 released validation states, agreement rises from 69.4% to 76.0%.

</details>

</details>

<details>
<summary><b>Stage 5: Planning &amp; Evaluation</b> (Click to expand.)</summary>

To try a single scene interactively, use [`planning/colab.ipynb`](../planning/colab.ipynb).

<details>
<summary>How to reproduce the 300-scene benchmark table</summary>

These commands run the 300 validation scenes with the three policy seeds used in the table. Always pass `--world-model` with the world model you want to plan with; otherwise the evaluator uses the one recorded in the policy checkpoint.

```bash
# Row 1: one sample, no goal (LatentPolicyAgent)
python -m assignment.training.planning.evaluate \
  checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --scenes 300 --target-count 4 --horizon 1795 --sample-seeds 0 2 3 \
  --configs '{"generate":16,"execute":1}'

# Row 2: 16 samples, subgoals every 80 steps, tau = 0.25 (GoalPlanningAgent)
python -m assignment.training.planning.evaluate \
  checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --kind goal --goals dataset/goals/scripted-val300-every80/goals.npz \
  --scenes 300 --target-count 4 --horizon 1795 --sample-seeds 0 2 3 \
  --configs '{"candidates":16,"generate":16,"execute":1,"tau":0.25}'
```

Set `--gpus` and `--workers` for your machine (for example, `--gpus 0,1 --workers 64`). To regenerate the subgoal images, run:

```bash
python -m assignment.training.planning.goals \
  --out dataset/goals/scripted-val300-every80 --scenes 300 --every 80 --gpus 0 --workers 8
```

</details>

</details>
