"""Shared numerical recorder. Simulator errors are audited, never task labels."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np

from .env import DT, PHYSICS_DT, CAMERAS
from .mixed_stacking import MATERIALS, MixedSceneConfig

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def scene_config(meta):
    return MixedSceneConfig(**meta.get('legacy_unused_scene_fields', {}), **meta['scene'])


def physics_metadata(env):
    return dict(contact_profile='synthetic-rigid-v1',
        note='Explicit contact pairs override geom solref/solimp, including fingertip pads.',
        objects=[dict(index=i, mass_kg=float(env.model.body_mass[b]),
                      principal_inertia_kg_m2=env.model.body_inertia[b].tolist(),
                      geom_friction=env.model.geom_friction[g].tolist())
                 for i, (b, g) in enumerate(zip(env.object_ids, env.object_geoms))],
        contact_pairs=[dict(geom1=env.model.geom(int(a)).name, geom2=env.model.geom(int(b)).name,
                            friction=f.tolist(), solref=r.tolist(), solimp=s.tolist())
                       for a, b, f, r, s in zip(env.model.pair_geom1, env.model.pair_geom2,
                           env.model.pair_friction, env.model.pair_solref, env.model.pair_solimp)])


def record_rollout(root, seed, mode, resolution, time_limit, *, env_factory,
                   policy_factory, schema, source_hashes):
    # Import here to avoid a module cycle through the public compatibility API.
    from .box_data import digest, write_json
    from .box_stacking import full_stack_success
    root = Path(root)
    name = f'seed-{seed:06d}' + (f'-{mode}' if mode != 'box-feedback' else '')
    folder = root/'rollouts'/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'record.json').exists():
        record = json.loads((folder/'record.json').read_text())
        for key, checksum in record['sha256'].items():
            if digest(root/record[key]) != checksum:
                raise ValueError(f'Corrupt existing attempt: {name}/{key}')
        return record
    paths = {k: folder/f for k, f in [('observations', 'observations.npz'),
             ('privileged', 'privileged.npz'), ('metadata', 'metadata.json')]}
    rows, states, proprio, times, remaining, heights, members = [], [], [], [], [], [], []
    actions, phases, agent_mask, deltas = [], [], [], []
    env = policy = None
    error = None
    stage, action, phase = 'initialize', None, None

    def snapshot():
        # Gather everything before mutating the retained prefix.
        row = env.privileged()
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        prop = env.proprio()
        score = env.score()
        mask = np.zeros(len(env.config.objects), bool)
        mask[score['object_indices']] = True
        if not (np.isfinite(state).all() and np.isfinite(prop).all()):
            raise RuntimeError('Non-finite recorded state')
        rows.append(row); states.append(state); proprio.append(prop)
        times.append(float(env.data.time-env.start_time)); remaining.append(env.time_remaining)
        heights.append(np.float32(score['height_m'])); members.append(mask)

    try:
        try:
            env = env_factory(seed=seed, resolution=resolution, time_limit=time_limit)
            policy = policy_factory(env)
            snapshot()
            while env.status != 'done':
                agent = env.status == 'running'
                previous_targets = env.targets.copy()
                stage, action, phase = 'policy' if agent else 'finalize', None, None
                if agent:
                    action, phase = policy.action(env, env.elapsed_steps)
                    if action is None:
                        env.stop()
                        continue
                    stage = 'step'
                    env.step(action)
                else:
                    action, phase = env.finalize_step()
                stage = 'snapshot'
                snapshot()
                actions.append(action.copy()); phases.append(phase); agent_mask.append(agent)
                deltas.append(env.targets-previous_targets)
            stage = 'outcome'
            outcome = env.outcome()
            success = bool(full_stack_success(env))
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc), stage=stage,
                         valid_transitions=len(actions), valid_observations=len(states),
                         failed_action=None if action is None else np.asarray(action).tolist(),
                         phase=phase, traceback=traceback.format_exc())
            outcome, success = None, None
        # I/O errors intentionally propagate: disk corruption/full disk must not
        # be misreported as a recoverable simulation or controller error.
        n = len(actions)
        stop_requested = np.zeros(len(states), bool)
        terminated, truncated = np.zeros(n, bool), np.zeros(n, bool)
        if error is None:
            if env.stop_reason == 'agent_stop':
                stop_requested[env.elapsed_steps] = True
            if n:
                terminated[-1] = env.stop_reason == 'agent_stop'
                truncated[-1] = env.stop_reason == 'time_limit'
        def stack(values, shape, dtype=float):
            return np.stack(values) if values else np.empty((0, *shape), dtype=dtype)
        np.savez_compressed(paths['observations'], proprio=stack(proprio, (75,)),
            actions=stack(actions, (25,), np.float32), applied_target_delta=stack(deltas, (25,)),
            agent_action_mask=np.asarray(agent_mask, bool), timestamps=np.asarray(times),
            time_remaining=np.asarray(remaining, np.float32), stop_requested=stop_requested,
            terminated=terminated, truncated=truncated,
            goal_xy=np.array([env.config.goal_x, env.config.goal_y], np.float32) if env else np.empty(0, np.float32))
        np.savez_compressed(paths['privileged'],
            **({k: np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}),
            integration_state=stack(states, (mujoco.mj_stateSize(env.model, STATE_SPEC) if env else 0,)),
            phases=np.asarray(phases, str), stable_height_m=np.asarray(heights, np.float32),
            stable_component_mask=stack(members, (len(env.config.objects) if env else 0,), bool))
        outcome_class = ('execution_error' if error else 'full_stack' if success else
                         'partial_stack' if outcome['final_stable_height_m'] > 0 else 'no_stable_height')
        meta = dict(schema=schema, scene_seed=seed, mode=mode, execution_status='error' if error else 'ok',
            outcome_class=outcome_class, resolution=resolution, outcome=outcome, success=success,
            source_sha256=source_hashes(), mujoco=mujoco.__version__,
            control_steps=int(sum(agent_mask)), camera_names=list(CAMERAS),
            action_dt=DT, physics_dt=PHYSICS_DT, integration_state_spec=int(STATE_SPEC),
            controller='Privileged scripted actuator-only feedback; not optimal-action labels',
            success_rule='all objects in terminal stable component and height >= 0.95 * sum(initial upright heights)')
        if env is not None:
            scene = asdict(env.config)
            legacy = {k: scene.pop(k) for k in ('cube_mass', 'cube_friction', 'cube_rgb')}
            meta.update(scene=scene, legacy_unused_scene_fields=legacy,
                physics=physics_metadata(env), material_catalog=MATERIALS,
                action_scale=env.action_scale.tolist(), joint_names=env.joint_names,
                actuator_limits=np.stack([env.low, env.high], axis=1).tolist(),
                mjcf_sha256=hashlib.sha256(env.xml.encode()).hexdigest())
        if policy is not None:
            meta.update(order=policy.order, controller_events=policy.events)
            if hasattr(policy, 'parameters'):
                meta.update(intervention=policy.parameters, intervention_step=policy.intervention_step)
        if error:
            paths['error'] = folder/'error.json'
            write_json(paths['error'], error)
        write_json(paths['metadata'], meta)
        result = dict(episode_id=name, scene_seed=seed, mode=mode, success=success,
            execution_status=meta['execution_status'], outcome_class=outcome_class,
            intervention_applied=bool(policy is not None and getattr(policy, 'intervention_step', None) is not None),
            transitions=n, control_steps=int(sum(agent_mask)), outcome=outcome, rendered=False,
            **{k: str(p.relative_to(root)) for k, p in paths.items()},
            sha256={k: digest(p) for k, p in paths.items()})
        write_json(folder/'record.json', result)
        return result
    finally:
        if env is not None:
            env.close()
