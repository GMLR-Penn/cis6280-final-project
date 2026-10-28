"""LeWM and SMWM world models for the stacking frame cache."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.data.cache import (
    ACTION_DIM, IMAGENET_MEAN, IMAGENET_STD, PROPRIO_CLIP, STRIDE, _proprio_array, load_cache,
)


def amp_dtype(training=False):
    """bfloat16 on GPUs with native bf16 (Ampere and newer). Otherwise (e.g. Colab T4, where
    bf16 is only emulated) float16 for inference and float32 for training, since float16
    training would need loss scaling."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32 if training else torch.float16


def autocast(dtype):
    return torch.autocast('cuda', dtype, enabled=dtype != torch.float32)


class Windows:
    """Uniform random windows of `length` consecutive frames inside one episode.

    With proprio=True, also z-scored proprioception clipped to +-PROPRIO_CLIP
    (contact transients reach >200 sigma in joint velocities).
    """

    def __init__(self, path, split, length, proprio=False):
        self.path, self.split, self.length = str(path), split, length
        index, self.frames, self.actions = load_cache(path)
        self.proprio = _proprio_array(path, index['frames'], 'r') if proprio else None
        if proprio:
            self.p_mean = np.asarray(index['proprio_mean'], np.float32)
            self.p_std = np.asarray(index['proprio_std'], np.float32)
        eps = [e for e in index['episodes'] if e['split'] == split and e['length'] >= length]
        if not eps:
            raise ValueError(f'no {split} episodes with at least {length} frames in {path}')
        self.starts = np.array([e['start'] for e in eps])
        self.counts = np.array([e['length']-length+1 for e in eps])
        self.cum = np.cumsum(self.counts)
        self.mean = np.tile(np.asarray(index['action_mean'], np.float32), STRIDE)
        self.std = np.tile(np.asarray(index['action_std'], np.float32), STRIDE)

    def __len__(self):
        return int(self.cum[-1])

    def sample(self, rng, batch):
        r = rng.integers(self.cum[-1], size=batch)
        e = np.searchsorted(self.cum, r, side='right')
        first = self.starts[e]+r-(self.cum[e]-self.counts[e])
        frames = np.stack([self.frames[s:s+self.length] for s in first])
        actions = np.stack([self.actions[s:s+self.length] for s in first])
        out = (torch.from_numpy(frames), torch.from_numpy((actions-self.mean)/self.std))
        if self.proprio is not None:
            p = np.stack([self.proprio[s:s+self.length] for s in first])
            out += (torch.from_numpy(np.clip((p-self.p_mean)/self.p_std, -PROPRIO_CLIP, PROPRIO_CLIP)),)
        return out


class WindowStream(torch.utils.data.IterableDataset):
    """Endless batches for a DataLoader; each worker draws from its own seeded stream."""

    def __init__(self, path, split, length, batch, seed, proprio=False):
        super().__init__()
        self.args, self.batch, self.seed = (path, split, length, proprio), batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        windows = Windows(*self.args)
        rng = np.random.default_rng([*np.atleast_1d(self.seed).tolist(), info.id if info else 0])
        while True:
            yield windows.sample(rng, self.batch)


# ----------------------------------------------------------------------------- model

def modulate(x, shift, scale):
    return x*(1+scale)+shift


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (Epps-Pulley statistic on random 1D projections)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3/(knots-1)
        weights = torch.full((knots,), 2*dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square()/2)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights*window)

    def forward(self, proj):
        """proj: (T, N, D); compares empirical characteristic function over batch N against N(0,1) = exp(-t^2/2)."""
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a/a.norm(dim=0)
        x = (proj@a).unsqueeze(-1)*self.t  # (T, N, num_proj, knots); dim -3 is batch N
        err = (x.cos().mean(-3)-self.phi).square()+x.sin().mean(-3).square()
        return ((err@self.weights)*proj.size(-2)).mean()


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-12), nn.LayerNorm(dim, eps=1e-12)
        self.qkv, self.proj = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        x = x+self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x+self.mlp(self.norm2(x))


class ViT(nn.Module):
    """ViT-Tiny (HF ViT layout: pre-LN blocks, CLS token, learned positions, final LN).

    With proprio_dim > 0, a linear proprio token (with its own position embedding) is
    appended to the patch tokens, so the CLS summary covers image and proprioception.
    """

    def __init__(self, image=(128, 256), patch=8, dim=192, depth=12, heads=3, mlp=768, proprio_dim=0):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, patch)
        tokens = (image[0]//patch)*(image[1]//patch)+(1 if proprio_dim else 0)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens+1, dim))
        self.proprio = nn.Linear(proprio_dim, dim) if proprio_dim else None
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads, mlp) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-12)
        for p in (self.cls, self.pos):
            nn.init.trunc_normal_(p, std=.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.zeros_(m.bias)

    def forward(self, x, proprio=None):
        x = self.embed(x).flatten(2).transpose(1, 2)
        parts = [self.cls.expand(len(x), -1, -1), x]
        if self.proprio is not None:
            parts.append(self.proprio(proprio.to(x.dtype))[:, None])
        x = torch.cat(parts, 1)+self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


class MLP(nn.Module):
    def __init__(self, dim_in, hidden, dim_out, norm=nn.BatchNorm1d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), norm(hidden), nn.GELU(), nn.Linear(hidden, dim_out))

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, dim_in, smoothed, dim, scale=4):
        super().__init__()
        self.patch = nn.Conv1d(dim_in, smoothed, 1)
        self.embed = nn.Sequential(nn.Linear(smoothed, scale*dim), nn.SiLU(), nn.Linear(scale*dim, dim))

    def forward(self, x):
        return self.embed(self.patch(x.float().transpose(1, 2)).transpose(1, 2))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner = heads*dim_head
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*inner, bias=False)
        self.out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0., is_causal=True)
        return self.out(x.transpose(1, 2).reshape(b, t, -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning."""

    def __init__(self, dim, heads, dim_head, mlp, dropout):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp, dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, g1, a1, s2, g2, a2 = self.ada(c).chunk(6, -1)
        x = x+a1*self.attn(modulate(self.norm1(x), s1, g1))
        return x+a2*self.mlp(modulate(self.norm2(x), s2, g2))


class Predictor(nn.Module):
    """Causal transformer over frame embeddings; position t predicts frame t+1."""

    def __init__(self, frames, dim=192, depth=6, heads=16, dim_head=64, mlp=2048, dropout=.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, frames, dim))
        self.blocks = nn.ModuleList(ConditionalBlock(dim, heads, dim_head, mlp, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        x = x+self.pos[:, :x.size(1)]
        for block in self.blocks:
            x = block(x, c)
        return self.norm(x)


@dataclass
class JEPAConfig:
    method: str = 'lewm'          # lewm | smwm
    context: int = 3              # predictor history length (frames)
    image: tuple = (128, 256)
    patch: int = 8
    dim: int = 192
    action_dim: int = ACTION_DIM
    action_smoothed: int = 128
    lam: float = .09              # SIGReg weight (lewm) or inverse-dynamics weight (smwm)
    sigreg_group: int = 0         # >0: SIGReg over random groups of this many (window x batch) embeddings
                                  # instead of per time step (keeps the sample size of short windows)
    sigreg_proj: int = 1024
    inverse_hidden: int = 256
    proprio_dim: int = 0          # >0: proprio token in the encoder plus a detached proprio probe on z


class StackJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.encoder = ViT(tuple(cfg.image), cfg.patch, d, proprio_dim=cfg.proprio_dim)
        if cfg.proprio_dim:
            self.probe = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                       nn.Linear(512, cfg.proprio_dim))
        self.projector = MLP(d, 2048, d)
        self.action_encoder = Embedder(cfg.action_dim, cfg.action_smoothed, d)
        self.predictor = Predictor(cfg.context, d)
        self.pred_proj = MLP(d, 2048, d)
        self.sigreg = SIGReg(num_proj=cfg.sigreg_proj)
        if cfg.method == 'smwm':
            h = cfg.inverse_hidden
            self.inverse = nn.Sequential(nn.Linear(2*d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                         nn.Linear(h, cfg.action_dim))
        self.register_buffer('pixel_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def encode(self, frames, proprio=None):
        """frames: uint8 (B, T, H, W, 3), proprio: normalized (B, T, P) -> (B, T, D)."""
        b = frames.size(0)
        x = frames.flatten(0, 1).permute(0, 3, 1, 2).float()/255
        x = (x-self.pixel_mean)/self.pixel_std
        p = proprio.flatten(0, 1) if self.cfg.proprio_dim else None
        return self.projector(self.encoder(x, p)).view(b, -1, self.cfg.dim)

    def predict(self, emb, act_emb):
        b = emb.size(0)
        return self.pred_proj(self.predictor(emb, act_emb).flatten(0, 1)).view(b, -1, self.cfg.dim)

    def forward(self, frames, actions, proprio=None):
        """Teacher-forced next-embedding loss on a window of context+1 frames, plus the regularizer.

        The proprio probe sees detached embeddings, so it never shapes the representation.
        """
        ctx = self.cfg.context
        emb = self.encode(frames, proprio)
        act_emb = self.action_encoder(actions)
        pred = self.predict(emb[:, :ctx], act_emb[:, :ctx])
        out = dict(pred=(pred.float()-emb[:, 1:ctx+1].float()).square().mean())
        if self.cfg.method == 'lewm':
            z = emb.float()
            if self.cfg.sigreg_group:
                z = z.reshape(-1, z.size(-1))
                groups = len(z)//self.cfg.sigreg_group
                z = z[torch.randperm(len(z), device=z.device)[:groups*self.cfg.sigreg_group]]
                z = z.view(groups, self.cfg.sigreg_group, -1)
            else:
                z = z.transpose(0, 1)
            out['sigreg'] = self.sigreg(z)
            out['loss'] = out['pred']+self.cfg.lam*out['sigreg']
        else:
            pair = torch.cat([emb[:, :-1], emb[:, 1:]], -1)
            out['inverse'] = (self.inverse(pair).float()-actions[:, :-1].float()).square().mean()
            out['loss'] = out['pred']+self.cfg.lam*out['inverse']
        if self.cfg.proprio_dim:
            out['probe'] = (self.probe(emb.detach()).float()-proprio.float()).square().mean()
            out['loss'] = out['loss']+out['probe']
        return out

    @torch.no_grad()
    def rollout(self, emb, act_emb, steps):
        """Autoregressive latent rollout. emb: (B, C, D) real context; act_emb covers C+steps-1 frames."""
        ctx = self.cfg.context
        hist = emb
        for h in range(steps):
            window = hist[:, -ctx:]
            first = hist.size(1)-window.size(1)
            pred = self.predict(window, act_emb[:, first:first+window.size(1)])[:, -1:]
            hist = torch.cat([hist, pred], 1)
        return hist[:, emb.size(1):]


def save_model(path, model, info):
    torch.save(dict(state_dict=model.state_dict(), config=asdict(model.cfg), info=info), path)


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StackJEPA(JEPAConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(), ckpt['info']
