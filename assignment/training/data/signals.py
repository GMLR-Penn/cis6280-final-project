"""Shared action noise, rewards, task stages and failure/recovery events.

Used identically by scripted and BC-teacher collection. Rewards, stages and events are
recomputed from recorded privileged arrays, so they can also label archived
v0.2 rollouts. These are task conventions, not physical ground truth.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

LABELER_VERSION = 4
CONTACT_FORCE_N = .05
FALL_M = .02
LIFT_CLEARANCE_M = .005
DEBOUNCE_STEPS = 5
SETTLE_STEPS = 20
STAGES = ('free_hand', 'touch', 'lifted', 'placing')
SHAPING = ('reach', 'grasp', 'lift', 'carry', 'drop', 'action')
EVENTS = ('grasp_attempt', 'lift', 'grasp_failure', 'placement', 'drop', 'collapse', 'ground_fall', 'regrasp')


def scene_split(seed, train=80, val=10):
    """Fixed per-scene split, independent of which other scenes or sources exist."""
    bucket = int(hashlib.sha256(f'stacking-split-v1/{int(seed)}'.encode()).hexdigest(), 16) % 100
    return 'train' if bucket < train else 'val' if bucket < train+val else 'test'


@dataclass(frozen=True)
class NoiseSpec:
    """kind: none | iid | ou. sigma is fixed or a per-episode (low, high) range.

    Units are normalized actions in [-1, 1]. `ou` is a stationary AR(1)
    process: x_t = rho x_{t-1} + sqrt(1-rho^2) eps_t, so its std is sigma.
    p_random replaces a whole action with Uniform[-1, 1]^25 (OGBench-style).
    scale_vector (25 values) replaces the wrist/finger scales with per-dimension
    scales, e.g. a DART noise level fitted to a learner's action errors.
    log_sigma draws a sigma range log-uniformly; p_zero sets the drawn sigma to
    0 for that fraction of episodes.
    Noise is applied only to policy-controlled steps, never to finalization.
    """
    kind: str = 'none'
    sigma: object = 0.
    rho: float = 0.
    wrist_scale: float = 1.
    finger_scale: float = 1.
    p_random: float = 0.
    scale_vector: object = None
    log_sigma: bool = False
    p_zero: float = 0.

    def __post_init__(self):
        if (self.kind not in ('none', 'iid', 'ou') or not 0 <= self.rho < 1 or not 0 <= self.p_random <= 1
                or not 0 <= self.p_zero <= 1):
            raise ValueError(f'Invalid noise spec: {self}')
        if self.log_sigma and not (isinstance(self.sigma, (tuple, list)) and min(self.sigma) > 0):
            raise ValueError('log_sigma needs a positive sigma range')
        if self.scale_vector is not None and len(self.scale_vector) != 25:
            raise ValueError('scale_vector needs 25 values')

    @classmethod
    def parse(cls, text):
        """'nominal', 'iid:0.1', 'iid:0.05-0.3' (per-episode range), 'ou:0.1:0.9',
        with optional ',w=SCALE,f=SCALE,p=PROB' for wrist/finger scales and p_random,
        ',v=FILE.json' for 25 per-dimension scales, ',log' for a log-uniform sigma
        range and ',z=PROB' for p_zero. Example: 'iid:0.005-0.2,log,z=0.1'."""
        text, *options = text.split(',')
        extra = {}
        for option in options:
            key, _, value = option.partition('=')
            if key == 'log':
                extra['log_sigma'] = True
            elif key == 'v':
                with open(value) as f:
                    extra['scale_vector'] = tuple(float(x) for x in json.load(f))
            else:
                extra[dict(w='wrist_scale', f='finger_scale', p='p_random', z='p_zero')[key]] = float(value)
        if text in ('none', 'nominal'):
            return cls(**extra)
        parts = text.split(':')
        sigma = tuple(float(x) for x in parts[1].split('-')) if '-' in parts[1] else float(parts[1])
        return cls(kind=parts[0], sigma=sigma, rho=float(parts[2]) if len(parts) > 2 else 0., **extra)


class ActionNoise:
    def __init__(self, spec, seed):
        self.spec = spec if isinstance(spec, NoiseSpec) else NoiseSpec(**spec)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 6280]))
        sigma = self.spec.sigma
        if not isinstance(sigma, (tuple, list)):
            self.sigma = float(sigma)
        elif self.spec.log_sigma:
            self.sigma = float(np.exp(self.rng.uniform(*np.log(sigma))))
        else:
            self.sigma = float(self.rng.uniform(*sigma))
        if self.spec.p_zero and self.rng.random() < self.spec.p_zero:
            self.sigma = 0.
        if self.spec.scale_vector is not None:
            self.scale = self.sigma*np.asarray(self.spec.scale_vector, float)
        else:
            self.scale = self.sigma*np.r_[np.full(3, self.spec.wrist_scale), np.full(22, self.spec.finger_scale)]
        self.state = np.zeros(25)

    def resolved(self):
        return dict(asdict(self.spec), sampled_sigma=self.sigma)

    def __call__(self, action):
        """Returns the executed action and whether it was a random replacement."""
        action = np.asarray(action, np.float64)
        if self.spec.kind == 'iid':
            action = action+self.rng.normal(size=25)*self.scale
        elif self.spec.kind == 'ou':
            rho = self.spec.rho
            self.state = rho*self.state+np.sqrt(1-rho**2)*self.rng.normal(size=25)
            action = action+self.state*self.scale
        replaced = bool(self.spec.p_random and self.rng.random() < self.spec.p_random)
        if replaced:
            action = self.rng.uniform(-1, 1, 25)
        return np.clip(action, -1, 1).astype(np.float32), replaced


def hand_point(env):
    ids = [env.model.body(n).id for n in ('right_thumb_DP', 'right_middle_DP')]
    return env.data.xpos[ids].mean(axis=0).copy()


def in_goal_zone(bounds, goal_xy, half_width):
    return ((bounds[..., 0, :2] >= np.asarray(goal_xy)-half_width)
            & (bounds[..., 1, :2] <= np.asarray(goal_xy)+half_width)).all(-1)


def at(truth, t):
    return {k: v[t] for k, v in truth.items() if k.startswith('object_')}


def structure(row, goal_xy, half_width):
    """Hands-free, table-rooted support tree in the goal zone at one instant.

    No three-second window: this is a progress signal, not the verified score.
    """
    free = (row['object_hand_contacts'] == 0) & (row['object_ground_contacts'] == 0)
    members = row['object_table_support'] & in_goal_zone(row['object_bounds'], goal_xy, half_width) & free
    todo = list(np.flatnonzero(members))
    while todo:
        for j in np.flatnonzero(row['object_support'][todo.pop()] & free & ~members):
            members[j] = True
            todo.append(j)
    height = float(row['object_bounds'][members, 1, 2].max()) if members.any() else 0.
    return members, height


def tower(row, goal_xy, half_width):
    """Largest connected structure among structure() members; separate structures are not added together."""
    rooted, _ = structure(row, goal_xy, half_width)
    support = row['object_support'] > 0
    linked = support | support.T
    members, seen, best = np.zeros_like(rooted), np.zeros_like(rooted), (0, 0.)
    for start in np.flatnonzero(rooted):
        if seen[start]:
            continue
        component, todo = np.zeros_like(rooted), [start]
        component[start] = True
        while todo:
            for j in np.flatnonzero(linked[todo.pop()] & rooted & ~component):
                component[j] = True
                todo.append(j)
        seen |= component
        key = (int(component.sum()), float(row['object_bounds'][component, 1, 2].max()))
        if key > best:
            members, best = component, key
    return members, best[1]


def instant_structure(truth, t, goal_xy, half_width):
    return structure(at(truth, t), goal_xy, half_width)


def row_hand_parts(row):
    force = row['object_hand_normal_force'] > CONTACT_FORCE_N
    return force.sum(-1), force[:, 0]


def hand_parts(truth, t):
    """(N,) number of hand parts pressing each object, and thumb contact."""
    return row_hand_parts(at(truth, t))


def row_lifted(row):
    resting = (row['object_table_contacts'] > 0) | (row['object_contacts'].sum(-1) > 0)
    return (row['object_hand_contacts'] > 0) & ~resting & (row['object_bounds'][:, 0, 2] > LIFT_CLEARANCE_M)


def lifted(truth, t):
    return row_lifted(at(truth, t))


def shaping_levels(row, hand, goal_xy, half_width, members=None):
    """Dense state levels (reach, grasp, lift, carry) used as shaping potentials."""
    if members is None:
        members, _ = structure(row, goal_xy, half_width)
    centres = row['object_pose'][:, :3]
    free = ~members & (row['object_ground_contacts'] == 0)
    parts, thumb = row_hand_parts(row)
    up = row_lifted(row) & free
    top = row['object_bounds'][:, 1, 2]
    target = centres[members][np.argmax(top[members])][:2] if members.any() else np.asarray(goal_xy)
    reach = -float(np.linalg.norm(centres[free]-hand, axis=-1).min()) if free.any() else 0.
    grasp = float(((np.minimum(parts, 3)/3)*thumb*free).max())
    bottom = row['object_bounds'][:, 0, 2]
    lift = float(np.clip(bottom[up], 0, .15).max()) if up.any() else 0.
    carry = -float(np.linalg.norm(centres[up, :2]-target, axis=-1).min()) if up.any() else 0.
    return np.array([reach, grasp, lift, carry])


def task_rewards(truth, goal_xy, half_width, final_height=None):
    """Potential-based progress reward; terminal step corrects to the verified score.

    Phi is the instantaneous supported height. With a verified final height,
    the undiscounted return equals final_height - Phi(s_0), i.e. the task score
    for scenes that start without a structure in the goal zone.
    """
    n = len(truth['object_bounds'])
    phi = np.array([instant_structure(truth, t, goal_xy, half_width)[1] for t in range(n)])
    reward = np.diff(phi)
    if final_height is not None and len(reward):
        reward[-1] += final_height-phi[-1]
    return reward.astype(np.float32), phi.astype(np.float32)


def shaping_terms(truth, hands, actions, goal_xy, half_width):
    """(T, len(SHAPING)) dense terms evaluated at s_{t+1}; TaskConfig.shaping chooses weights."""
    n = len(actions)
    terms = np.zeros((n, len(SHAPING)), np.float32)
    for t in range(n):
        s = t+1
        terms[t, :4] = shaping_levels(at(truth, s), hands[s], goal_xy, half_width)
        new_ground = (truth['object_ground_contacts'][s] > 0) & (truth['object_ground_contacts'][s-1] == 0)
        terms[t, 4] = -float(new_ground.sum())
        terms[t, 5] = -float(np.mean(np.square(actions[t])))
    return terms


def stages(truth):
    """Per-observation hand stage index into STAGES."""
    n = len(truth['object_bounds'])
    out = np.zeros(n, np.int8)
    for t in range(n):
        parts, _ = hand_parts(truth, t)
        touching = truth['object_hand_contacts'][t] > 0
        if not touching.any():
            continue
        up = lifted(truth, t)
        support = (truth['object_contacts'][t].sum(-1) > 0) | (truth['object_table_contacts'][t] > 0)
        held = touching & (parts >= 2)
        out[t] = 3 if (held & support & (truth['object_bounds'][t, :, 0, 2] > .03)).any() else 2 if up.any() else 1
    return out


def events(truth, goal_xy, half_width):
    """Debounced object-level events with observation indices.

    grasp_attempt: >=2 hand parts press a non-structure object.
    lift / grasp_failure: the attempt becomes airborne, or ends without lifting.
    placement / drop: after a lifted object leaves the hand, whether it joins
    the structure once settled. collapse: a structure member persistently
    leaves it and either drops >2 cm before any lift, or, once hands-free and
    settled, has not rejoined without having been lifted in a grasp (hand
    knocks count; deliberate unstacking does not). regrasp: lift of an object that previously failed, dropped or
    collapsed (a recovery attempt).
    """
    n, k = len(truth['object_bounds']), truth['object_bounds'].shape[1]
    member = np.stack([instant_structure(truth, t, goal_xy, half_width)[0] for t in range(n)])
    out = []
    attempt = [None]*k
    was_lifted = [False]*k
    released = [None]*k
    troubled = [False]*k
    stable_member = member[0].copy()
    change = np.zeros(k, int)
    left = [None]*k
    left_bottom = [0.]*k
    last_contact = [-SETTLE_STEPS]*k
    for t in range(n):
        parts, _ = hand_parts(truth, t)
        up = lifted(truth, t)
        for i in range(k):
            if attempt[i] is None and parts[i] >= 2 and not stable_member[i]:
                attempt[i] = t
                was_lifted[i] = False
                released[i] = None
                out.append(dict(step=t, event='grasp_attempt', object=i))
            if released[i] is not None and truth['object_hand_contacts'][t, i] > 0:
                # Fingertips can graze the object while opening: settle from the last contact.
                released[i] = t
            if attempt[i] is not None and up[i] and not was_lifted[i]:
                was_lifted[i] = True
                out.append(dict(step=t, event='regrasp' if troubled[i] else 'lift', object=i))
                troubled[i] = False
            if attempt[i] is not None and truth['object_hand_contacts'][t, i] == 0:
                if not was_lifted[i]:
                    out.append(dict(step=t, event='grasp_failure', object=i))
                    troubled[i] = True
                else:
                    released[i] = t
                attempt[i] = None
            if released[i] is not None and t-released[i] >= SETTLE_STEPS:
                joined = member[t, i]
                out.append(dict(step=released[i], event='placement' if joined else 'drop', object=i))
                troubled[i] |= not joined
                released[i] = None
            if t and truth['object_ground_contacts'][t, i] > 0 and truth['object_ground_contacts'][t-1, i] == 0:
                out.append(dict(step=t, event='ground_fall', object=i))
            if truth['object_hand_contacts'][t, i] > 0:
                last_contact[i] = t
            change[i] = change[i]+1 if member[t, i] != stable_member[i] else 0
            if change[i] >= DEBOUNCE_STEPS:
                left[i] = t-DEBOUNCE_STEPS+1 if stable_member[i] else None
                if left[i] is not None:
                    left_bottom[i] = truth['object_bounds'][left[i], i, 0, 2]
                stable_member[i] = member[t, i]
                change[i] = 0
            if left[i] is not None:
                fell = truth['object_bounds'][t, i, 0, 2] < left_bottom[i]-FALL_M
                if attempt[i] is not None and up[i] and not fell:
                    left[i] = None  # deliberately picked off the structure
                elif fell or (t-max(left[i], last_contact[i]) >= SETTLE_STEPS and not member[t, i]):
                    out.append(dict(step=left[i], event='collapse', object=i))
                    troubled[i] = True
                    left[i] = None
    for i in range(k):
        if released[i] is not None:
            out.append(dict(step=released[i], event='placement' if member[-1, i] else 'drop', object=i,
                            unsettled=True))
        if left[i] is not None and not member[-1, i]:
            out.append(dict(step=left[i], event='collapse', object=i, unsettled=True))
    return sorted(out, key=lambda e: (e['step'], e['object']))


def episode_summary(truth, goal_xy, half_width, final_height, scored_count, rewards):
    labels = stages(truth)
    found = events(truth, goal_xy, half_width)
    structure = [instant_structure(truth, t, goal_xy, half_width) for t in range(len(labels))]
    counts = {name: sum(e['event'] == name for e in found) for name in EVENTS}
    return dict(final_stable_height_m=final_height, scored_object_count=scored_count,
                max_instant_height_m=max(h for _, h in structure),
                max_instant_structure_count=int(max(m.sum() for m, _ in structure)),
                task_return=float(np.sum(rewards)), stage_fraction={s: float(np.mean(labels == i))
                                                                   for i, s in enumerate(STAGES)},
                event_counts=counts, recovered=counts['regrasp'] > 0,
                failure_types=sorted(k for k in ('grasp_failure', 'drop', 'collapse', 'ground_fall') if counts[k]))
