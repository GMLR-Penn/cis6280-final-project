"""Stage 3. Encode cached frames with a frozen world model into a float16 latent memmap.

Writes <out>/latents.f16 of shape (frames, dim) aligned with the frame cache, plus
latents.json storing the encoder path, cache path, and train-split latent mean and std.
One process per GPU encodes a contiguous shard (pass --gpus 0,1,... for multi-GPU encoding).
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def shard(args, rank, gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[rank])
    import torch
    from assignment.training.data.cache import load_cache
    from assignment.training.training.jepa import amp_dtype, autocast, load_model
    model, _ = load_model(args.model, 'cuda')
    index, frames, _ = load_cache(args.cache)
    total, dim = index['frames'], model.cfg.dim
    out = np.memmap(args.out/'latents.f16', np.float16, 'r+', shape=(total, dim))
    bounds = np.linspace(0, total, len(gpus)+1).astype(int)
    lo, hi = bounds[rank], bounds[rank+1]
    dtype = amp_dtype()
    with torch.no_grad(), autocast(dtype):
        for i, s in enumerate(range(lo, hi, args.batch)):
            x = torch.from_numpy(np.ascontiguousarray(frames[s:min(s+args.batch, hi)])).cuda(non_blocking=True)
            out[s:s+len(x)] = model.encode(x[:, None])[:, 0].float().cpu().numpy().astype(np.float16)
            if rank == 0 and i % 100 == 0:
                print(json.dumps(dict(stage='encode', done=int(s-lo), total=int(hi-lo))), flush=True)
    out.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--batch', type=int, default=2048)
    args = p.parse_args()
    from assignment.training.data.cache import load_cache
    from assignment.training.training.jepa import load_model
    index, _, _ = load_cache(args.cache)
    model, _ = load_model(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    np.memmap(args.out/'latents.f16', np.float16, 'w+', shape=(index['frames'], model.cfg.dim)).flush()
    gpus = [int(g) for g in args.gpus.split(',')]
    procs = [mp.get_context('spawn').Process(target=shard, args=(args, r, gpus)) for r in range(len(gpus))]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
        if proc.exitcode:
            raise SystemExit(f'encoder shard failed with exit code {proc.exitcode}')
    z = np.memmap(args.out/'latents.f16', np.float16, 'r', shape=(index['frames'], model.cfg.dim))
    train_eps = [e for e in index['episodes'] if e['split'] == 'train'] or index['episodes']
    train = np.concatenate([np.arange(e['start'], e['start']+e['length']) for e in train_eps])
    sample = z[np.sort(np.random.default_rng(0).choice(train, min(len(train), 200000), replace=False))].astype(np.float64)
    if not np.isfinite(sample).all():
        raise SystemExit('non-finite latents')
    (args.out/'latents.json').write_text(json.dumps(dict(
        model=str(args.model), model_config=asdict(model.cfg), cache=str(args.cache), frames=index['frames'],
        dim=model.cfg.dim, mean=sample.mean(0).tolist(), std=sample.std(0).tolist()), indent=1))
    print(json.dumps(dict(stage='done', frames=index['frames'])), flush=True)


if __name__ == '__main__':
    main()
