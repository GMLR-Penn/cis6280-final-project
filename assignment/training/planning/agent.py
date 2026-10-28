"""Closed-loop policy agent and goal-conditioned planning agent."""
import contextlib
from pathlib import Path

import numpy as np
import torch

from assignment.training.data.cache import RAW_ACTION_DIM, STRIDE
from assignment.training.training.jepa import amp_dtype
from assignment.training.training.policy import load_policy


class LatentPolicyAgent:
    """Closed-loop controller that samples action chunks from the diffusion-forcing policy.

    Call step(frame) at every block boundary (every STRIDE=5 control steps) with the tiled
    (128, 256, 3) uint8 RGB frame; returns the (STRIDE, 25) float32 raw actions in [-1, 1]
    for that block.
    """

    def __init__(self, policy_path, device='cuda', generate=None, execute=1, n_steps=10, lag=0, seed=0,
                 world_model=None):
        from assignment.training.training.jepa import load_model
        self.policy, info = load_policy(policy_path, device)
        wm_path = Path(world_model or info['encoder'])
        if not wm_path.exists():
            raise FileNotFoundError(
                f"World-model checkpoint not found at '{wm_path}'. Pass world_model='path/to/model.pt' "
                f"or set ckpt['info']['encoder'] in '{policy_path}' to your local world-model file."
            )
        self.encoder, _ = load_model(wm_path, device)
        self.device, self.info = str(device), info
        self._device_type = torch.device(device).type
        self._amp_dtype = amp_dtype()
        self.generate = generate or self.policy.cfg.horizon
        self.execute, self.n_steps, self.lag = execute, n_steps, lag
        if not 1 <= execute <= self.generate <= self.policy.cfg.horizon:
            raise ValueError('need 1 <= execute <= generate <= horizon')
        self.z_mean = torch.tensor(info['z_mean'], device=device)
        self.z_std = torch.tensor(info['z_std'], device=device)
        self.a_mean = np.asarray(info['action_mean'], np.float32)
        self.a_std = np.asarray(info['action_std'], np.float32)
        self.generator = torch.Generator(device).manual_seed(seed)
        self.reset()

    def _autocast(self):
        if self._device_type == 'cuda':
            return torch.autocast('cuda', self._amp_dtype)
        return contextlib.nullcontext()

    def reset(self, seed=None):
        """Start an episode. Passing a seed (e.g. the scene seed) makes the episode's samples
        independent of which episodes this agent ran before."""
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.history, self.queue = [], []

    @torch.no_grad()
    def step(self, frame):
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)[None, None]
        with self._autocast():
            z = self.encoder.encode(x)[0, 0].float()
        self.history = (self.history+[(z-self.z_mean)/self.z_std])[-self.policy.cfg.context:]
        if not self.queue:
            ctx = [self.history[0]]*(self.policy.cfg.context-len(self.history))+self.history
            blocks = self.policy.sample(torch.stack(ctx)[None], self.generate, self.n_steps, self.lag,
                                        self.generator)[0, :self.execute].cpu().numpy()
            raw = np.clip(blocks*self.a_std+self.a_mean, -1, 1).reshape(-1, STRIDE, RAW_ACTION_DIM)
            self.queue = list(raw)
        return self.queue.pop(0)


class GoalPlanningAgent(LatentPolicyAgent):
    """Test-time planner that scores candidate policy chunks with the latent world model.

    At each replanning step:
      1. Encode the current tiled RGB frame I_k into a latent z_k in R^192 and advance the
         subgoal index m whenever ||z_k - g_m||_2^2 < tau * ||g_m - g_{m-1}||_2^2.
      2. Draw `candidates` chunks of `generate` action blocks from the policy.
      3. Roll each candidate chunk forward in the world model from the last C<=3 real frame
         latents (and the action blocks executed between them) to predict latents z_hat.
      4. Execute the first `execute` blocks of the candidate whose final predicted latent
         z_hat_{k+generate} is closest in squared L2 distance to the active subgoal g_m.

    Goals are an ordered subgoal sequence; the index advances monotonically
    while ||z - g_m||_2^2 < tau * ||g_m - g_{m-1}||_2^2, with g_{-1} the latent of init_frame.
    Call reset(goal_frames, init_frame, seed) per episode and step(frame) every STRIDE steps.
    """

    def __init__(self, policy_path, device='cuda', candidates=16, tau=.25, **kwargs):
        super().__init__(policy_path, device, **kwargs)
        self.candidates, self.tau = candidates, tau
        self.goals = None

    def reset(self, goal_frames=None, init_frame=None, seed=None):
        if goal_frames is not None and init_frame is None:
            raise ValueError('init_frame is required with goal_frames')
        super().reset(seed)
        self.frames, self.executed, self.index = [], [], 0
        self.last_distance = self.goals = self.gap = None
        if goal_frames is not None:
            self.goals = self._encode(np.asarray(goal_frames))
            prev = torch.cat([self._encode(np.asarray(init_frame)[None]), self.goals[:-1]])
            self.gap = (self.goals-prev).square().sum(-1)

    def goal_index(self, z):
        while self.index < len(self.goals)-1 and \
                (z-self.goals[self.index]).square().sum() < self.tau*self.gap[self.index]:
            self.index += 1
        return self.index

    @torch.no_grad()
    def _encode(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)[:, None]
        with self._autocast():
            return self.encoder.encode(x)[:, 0].float()

    @torch.no_grad()
    def step(self, frame):
        if self.goals is None:
            raise RuntimeError('Call agent.reset(goal_frames, init_frame) before step().')
        z = self._encode(frame[None])[0]
        ctx = self.encoder.cfg.context
        self.frames = (self.frames+[z])[-ctx:]
        self.history = (self.history+[(z-self.z_mean)/self.z_std])[-self.policy.cfg.context:]
        goal = self.goals[self.goal_index(z)]
        if not self.queue:
            pctx = [self.history[0]]*(self.policy.cfg.context-len(self.history))+self.history
            n = self.candidates
            blocks = self.policy.sample(torch.stack(pctx)[None].expand(n, -1, -1), self.generate, self.n_steps,
                                        self.lag, self.generator)
            best = 0
            if n > 1:
                # Pair the C-1 earlier context frames with the C-1 blocks executed after them,
                # and the current frame with the n candidate chunks of length `generate`.
                past = self.executed[len(self.executed)-len(self.frames)+1:] if len(self.frames) > 1 else []
                acts = torch.cat([torch.stack(past)[None].expand(n, -1, -1), blocks], 1) if past else blocks
                with self._autocast():
                    pred = self.encoder.rollout(torch.stack(self.frames)[None].expand(n, -1, -1),
                                                self.encoder.action_encoder(acts), self.generate)
                self.last_distance = (pred[:, -1].float()-goal).square().sum(-1)
                best = int(self.last_distance.argmin())
            raw = np.clip(blocks[best, :self.execute].cpu().numpy()*self.a_std+self.a_mean, -1, 1)
            self.queue = list(raw.reshape(-1, STRIDE, RAW_ACTION_DIM))
        block = self.queue.pop(0)
        self.executed.append(torch.from_numpy((block.reshape(-1)-self.a_mean)/self.a_std).to(self.device))
        return block
