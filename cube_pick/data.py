"""Lossless visual episodes, privileged sidecars, and leakage-aware window loading."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .env import CubePickEnv
from .policy import ScriptedCollector, summarize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_episode(job):
    root, split, scene_index, scene_seed, mode, resolution, robot, *options = job
    cube_edge = options[0] if options else None
    grasp_policy = options[1] if len(options) > 1 else 'fixed'
    root = Path(root)
    scene_id = f"{split}-{scene_index:04d}"
    episode_id = f"{scene_id}-{mode}"
    obs_path = root/"observations"/split/f"{episode_id}.npz"
    truth_path = root/"privileged"/split/f"{episode_id}.npz"
    metadata_path = root/"privileged"/split/f"{episode_id}.json"
    for path in (obs_path, truth_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
    if robot == "sharpa_multicube":
        from .multicube import MultiCubeStackEnv, MultiCubeCollector, summarize_multicube
        env = MultiCubeStackEnv(seed=scene_seed, resolution=resolution, cube_edge=cube_edge or .050,
                                num_cubes=options[2] if len(options) > 2 else 3)
        policy = MultiCubeCollector(env, mode, scene_seed)
        outcome_fn = summarize_multicube
    elif robot == "sharpa_stack":
        from .stacking import SharpaStackEnv, StackingCollector, summarize_stack
        env = SharpaStackEnv(seed=scene_seed, resolution=resolution, cube_edge=cube_edge)
        policy = StackingCollector(env, mode, scene_seed)
        outcome_fn = summarize_stack
    elif robot == "sharpa_whole_hand":
        from .whole_hand import SharpaWholeHandEnv, WholeHandCollector, SizeConditionedCollector, summarize_whole_hand
        env = SharpaWholeHandEnv(seed=scene_seed, resolution=resolution, cube_edge=cube_edge)
        if grasp_policy not in ('fixed', 'size_conditioned'):
            raise ValueError(f'Unknown grasp policy: {grasp_policy}')
        collector = SizeConditionedCollector if grasp_policy == 'size_conditioned' else WholeHandCollector
        policy = collector(env, mode, scene_seed)
        outcome_fn = summarize_whole_hand
    elif robot == "sharpa":
        from .sharpa import SharpaCubeEnv
        from .sharpa_policy import SharpaCollector, summarize_sharpa
        env = SharpaCubeEnv(seed=scene_seed, resolution=resolution)
        policy = SharpaCollector(env, mode, scene_seed)
        outcome_fn = summarize_sharpa
    else:
        env = CubePickEnv(seed=scene_seed, resolution=resolution)
        policy = ScriptedCollector(env, mode, scene_seed)
        outcome_fn = summarize
    frames, proprio, truth, times = [env.render()], [env.proprio()], [env.privileged()], [0.]
    actions, phases = [], []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    sim_states = []
    started = time.perf_counter()
    try:
        for t in range(policy.horizon + 1):
            state = np.empty(mujoco.mj_stateSize(env.model, spec))
            mujoco.mj_getState(env.model, env.data, state, spec)
            sim_states.append(state)
            if t == policy.horizon:
                break
            action, phase = policy.action(env, t)
            proprio.append(env.step(action))
            frames.append(env.render())
            truth.append(env.privileged())
            actions.append(action.copy())
            phases.append(phase)
            times.append(float(env.data.time-env.start_time))
        proprio = np.stack(proprio)
        outcome = outcome_fn(env, truth, proprio)
        np.savez_compressed(obs_path, rgb=np.stack(frames), proprio=proprio,
                            actions=np.stack(actions), timestamps=np.asarray(times, np.float64),
                            goal_xy=np.array([env.config.goal_x, env.config.goal_y], np.float32))
        np.savez_compressed(truth_path, **{key: np.stack([item[key] for item in truth]) for key in truth[0]},
                            integration_state=np.stack(sim_states), phases=np.asarray(phases))
        metadata = {"robot": robot, "scene": asdict(env.config), "behavior_seed": scene_seed, "mode": mode,
                    "resolution": resolution, "stage_schedule": policy.stages,
                    "mjcf_sha256": hashlib.sha256(env.xml.encode()).hexdigest(), "outcome": outcome,
                    "integration_state_spec": int(spec), "mujoco": mujoco.__version__}
        if robot in ('sharpa_whole_hand','sharpa_stack','sharpa_multicube'): metadata['grasp_policy'] = grasp_policy
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return {"episode_id": episode_id, "scene_id": scene_id, "split": split, "collector_mode": mode,
                "transitions": len(actions), "frames": len(frames), "outcome": outcome,
                "observations": str(obs_path.relative_to(root)), "observations_sha256": sha256(obs_path),
                "privileged": str(truth_path.relative_to(root)), "privileged_sha256": sha256(truth_path),
                "metadata": str(metadata_path.relative_to(root)), "metadata_sha256": sha256(metadata_path),
                "observation_bytes": obs_path.stat().st_size,
                "collection_wall_seconds": round(time.perf_counter()-started, 3)}
    finally:
        env.close()
