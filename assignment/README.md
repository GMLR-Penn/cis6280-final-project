# Stacking with a Latent World Model

In this project, you control a simulated 22-joint hand to stack four boxes using only two wrist-camera RGB views (see [`docs/SIMULATOR.md`](../docs/SIMULATOR.md) for the simulator details). Our baseline pairs an action-chunk policy with a latent world model: at each step, the policy proposes candidate action chunks, the world model predicts where each chunk leads in latent space, and the planner picks the chunk that gets closest to the current visual subgoal.

## Assignment Tracks

- **[Planning Track (`planning/README.md`)](planning/README.md)** — Keep our pretrained world model and policy frozen, and use the provided subgoal images (spaced every 80 control steps, or 16 action blocks) to build a better test-time planner. Start with [`planning/colab.ipynb`](planning/colab.ipynb).
- **[Training Track (`training/README.md`)](training/README.md)** — Collect the 20,000-episode demonstration dataset, pretrain the latent world model, train the diffusion-forcing policy, and finetune the world-model predictor on on-policy candidate rollouts. Pretrained artifacts are hosted on Hugging Face for every stage so that you can skip ahead to any part of the pipeline.
