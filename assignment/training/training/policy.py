"""Diffusion-forcing action-chunk policy on frozen JEPA latents.

Observation: the last `context` frame latents z (frozen encoder, one frame per
STRIDE control steps), z-scored with train statistics. Output: `horizon` action
blocks, each the STRIDE raw executed actions that follow a frame (z-scored like
the JEPA cache). Training labels are executed actions of every episode; blocks
that overlap the environment's post-stop finalization are masked.

Diffusion forcing (Chen et al., arXiv 2407.01392): every action token gets an
independent noise level, attention over action tokens is causal, so at test
time any prefix of the chunk can be generated (variable horizon) and the noise
levels can follow a pyramid schedule. Noise process: rectified flow,
x_t = (1-t) a + t eps, the network predicts v = eps - a.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.data.cache import RAW_ACTION_DIM, STRIDE
from assignment.training.training.jepa import amp_dtype, modulate


# ----------------------------------------------------------------------------- data

def episode_limits(index):
    """Per episode: number of leading frames whose action block is all agent (pre-stop) steps."""
    episodes = index['episodes']
    if episodes and 'control_steps' in episodes[0]:
        return np.array([min(e['control_steps']//STRIDE, e['length']-1) for e in episodes])
    control = {}
    for root in index.get('roots', ()):
        manifest = Path(root)/'manifest.json'
        if manifest.exists():
            for r in json.loads(manifest.read_text())['episodes']:
                control[r['episode_id']] = r['control_steps']
    missing = sum(e['episode_id'] not in control for e in episodes)
    if missing:
        warnings.warn(f'{missing}/{len(episodes)} episodes have no control_steps in index.json or a readable '
                      f'manifest; their post-stop frames will be used as training targets')
    return np.array([min(control[e['episode_id']]//STRIDE, e['length']-1)
                     if e['episode_id'] in control else e['length']-1 for e in episodes])


class LatentData:
    """GPU-resident latents and action blocks; samples (context latents, target blocks, mask)."""

    def __init__(self, cache, latents, split, context, horizon, device, max_sigma=None):
        cache, latents = Path(cache), Path(latents)
        index = json.loads((cache/'index.json').read_text())
        stats = json.loads((latents/'latents.json').read_text())
        total, dim = index['frames'], stats['dim']
        self.context, self.horizon, self.device = context, horizon, device
        self.z_mean = torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        self.z_std = torch.tensor(stats['std'], dtype=torch.float32, device=device)
        self.a_mean = np.tile(np.asarray(index['action_mean'], np.float32), STRIDE)
        self.a_std = np.tile(np.asarray(index['action_std'], np.float32), STRIDE)
        self.z = torch.from_numpy(np.array(np.memmap(latents/'latents.f16', np.float16, 'r',
                                                     shape=(total, dim)))).to(device)
        actions = np.memmap(cache/'actions.f32', np.float32, 'r', shape=(total, STRIDE*RAW_ACTION_DIM))
        act_dtype = amp_dtype()
        self.actions = torch.from_numpy((np.asarray(actions)-self.a_mean)/self.a_std).to(device, act_dtype)
        first, limit, anchors = np.zeros(total, np.int64), np.zeros(total, np.int64), []
        for e, n in zip(index['episodes'], episode_limits(index)):
            s = e['start']
            first[s:s+e['length']] = s
            limit[s:s+e['length']] = s+n
            if e['split'] == split and n > 0 and (max_sigma is None or (e['sigma'] or 0.) <= max_sigma):
                anchors.append(np.arange(s, s+n))
        self.first = torch.from_numpy(first).to(device)
        self.limit = torch.from_numpy(limit).to(device)
        self.anchors = torch.from_numpy(np.concatenate(anchors)).to(device)

    def normalize(self, z):
        return (z.float()-self.z_mean)/self.z_std

    def sample(self, generator, batch):
        g = self.anchors[torch.randint(len(self.anchors), (batch,), device=self.device, generator=generator)]
        back = torch.arange(self.context-1, -1, -1, device=self.device)
        ctx = torch.maximum(g[:, None]-back, self.first[g][:, None])
        steps = g[:, None]+torch.arange(self.horizon, device=self.device)
        mask = steps < self.limit[g][:, None]
        steps = torch.minimum(steps, self.limit[g][:, None]-1)
        return self.normalize(self.z[ctx]), self.actions[steps].float(), mask


# ----------------------------------------------------------------------------- model

def timestep_embedding(t, dim, max_period=10000):
    half = dim//2
    freqs = torch.exp(-math.log(max_period)*torch.arange(half, device=t.device, dtype=torch.float32)/half)
    args = 1000*t.float()[..., None]*freqs
    return torch.cat([args.cos(), args.sin()], -1)


class Block(nn.Module):
    """Pre-LN transformer block with per-token AdaLN-zero conditioning and an attention mask."""

    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, mask):
        s1, g1, a1, s2, g2, a2 = self.ada(c).chunk(6, -1)
        b, n, d = x.shape
        h = modulate(self.norm1(x), s1, g1)
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        x = x+a1*self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, n, d))
        return x+a2*self.mlp(modulate(self.norm2(x), s2, g2))


@dataclass
class PolicyConfig:
    latent_dim: int = 192
    action_dim: int = STRIDE*RAW_ACTION_DIM
    context: int = 1
    horizon: int = 16
    dim: int = 512
    depth: int = 8
    heads: int = 8
    mlp: int = 2048


class DFPolicy(nn.Module):
    """Tokens: `context` clean latent tokens, then `horizon` noisy action tokens.

    Action token i attends to every latent token and to action tokens <= i.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, c, h = cfg.dim, cfg.context, cfg.horizon
        self.obs_in = nn.Sequential(nn.Linear(cfg.latent_dim, d), nn.GELU(), nn.Linear(d, d))
        self.act_in = nn.Linear(cfg.action_dim, d)
        self.pos = nn.Parameter(torch.randn(1, c+h, d)*.02)
        self.t_embed = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.obs_cond = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2*d))
        self.out = nn.Linear(d, cfg.action_dim)
        for m in (self.out_ada[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        q, k = torch.arange(c+h)[:, None], torch.arange(c+h)[None]
        self.register_buffer('mask', (k < c) | (k <= q), persistent=False)

    def forward(self, z, x, t):
        """z: (B, C, latent), x: noisy actions (B, h, action), t: (B, h) noise levels -> velocity (B, h, action)."""
        c, h = z.size(1), x.size(1)
        tokens = torch.cat([self.obs_in(z), self.act_in(x)], 1)+self.pos[:, :c+h]
        cond = torch.cat([self.obs_cond.expand(len(z), c, -1), self.t_embed(timestep_embedding(t, 256))], 1)
        mask = self.mask[:c+h, :c+h]
        for block in self.blocks:
            tokens = block(tokens, cond, mask)
        shift, scale = self.out_ada(cond[:, c:]).chunk(2, -1)
        return self.out(modulate(self.norm(tokens[:, c:]), shift, scale))

    def loss(self, z, actions, mask, generator=None):
        """Diffusion-forcing flow-matching loss with independent per-token noise levels."""
        t = torch.rand(actions.shape[:2], device=actions.device, generator=generator)
        eps = torch.randn(actions.shape, device=actions.device, generator=generator)
        x = (1-t[..., None])*actions+t[..., None]*eps
        err = (self(z, x, t).float()-(eps-actions)).square().mean(-1)
        return (err*mask).sum()/mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, z, steps=None, n_steps=10, lag=0, generator=None):
        """Generate `steps` action blocks by Euler integration from t=1 (noise) to t=0 (data).

        lag=0: all tokens share one noise level at each step (full-sequence generation).
        lag=k>0: pyramid schedule where token i trails token i-1 by k solver steps; tokens
        with t_now == t_next at a given step receive a zero update.
        """
        h = steps or self.cfg.horizon
        x = torch.randn(len(z), h, self.cfg.action_dim, device=z.device, generator=generator)
        offset = torch.arange(h, device=z.device)*lag
        for s in range(n_steps+(h-1)*lag):
            t_now = (1-(s-offset)/n_steps).clamp(0, 1)
            t_next = (1-(s+1-offset)/n_steps).clamp(0, 1)
            if not (t_now > t_next).any():
                continue
            v = self(z, x, t_now.expand(len(z), h)).float()
            x = x+(t_next-t_now)[None, :, None]*v
        return x


def save_policy(path, model, info):
    torch.save(dict(state_dict=model.state_dict(), config=asdict(model.cfg), info=info), path)


def load_policy(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DFPolicy(PolicyConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(), ckpt['info']


