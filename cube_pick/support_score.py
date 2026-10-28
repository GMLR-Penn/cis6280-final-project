"""Version 2: score stable structures through upward, load-bearing contacts."""
import numpy as np


def supported_height(records, goal_xy, goal_half_width=.085, window=31):
    empty = dict(height_m=0., object_indices=[], object_count=0)
    if len(records) < window:
        return empty
    rows = list(records)[-window:]
    velocity = np.stack([r['object_velocity'] for r in rows])
    bounds = np.stack([r['object_bounds'] for r in rows])
    eligible = ((np.linalg.norm(velocity[:, :, :3], axis=-1) < .1)
                & (np.linalg.norm(velocity[:, :, 3:], axis=-1) < .02)).all(axis=0)
    eligible &= (np.ptp(bounds, axis=0) < .005).all(axis=(1, 2))
    for key in ('object_hand_contacts', 'object_ground_contacts'):
        eligible &= ~np.stack([r[key] for r in rows]).any(axis=0)
    in_zone = ((bounds[:, :, 0, :2] >= np.asarray(goal_xy)-goal_half_width)
               & (bounds[:, :, 1, :2] <= np.asarray(goal_xy)+goal_half_width)).all(axis=(0, 2))
    table = np.stack([r['object_table_contacts'] for r in rows]) > 0
    roots = np.stack([r['object_table_support'] for r in rows]).all(axis=0) & in_zone & eligible
    eligible &= ~table.any(axis=0) | roots
    supports = np.stack([r['object_support'] for r in rows]) > 0
    persistent, ever = supports.all(axis=0), supports.any(axis=0)
    # Every observed lower supporter must itself be valid. A dangling upper
    # object cannot earn credit after its unstable/hand-held support is removed.
    accepted = set()
    for _ in range(len(eligible)):
        added = {i for i in range(len(eligible)) if eligible[i] and i not in accepted
                 and set(np.flatnonzero(ever[:, i])).issubset(accepted)
                 and (roots[i] or any(persistent[j, i] for j in accepted))}
        if not added:
            break
        accepted.update(added)
    # Separate towers are separate candidates; do not call four isolated bases
    # a four-object stack just because one of them is tall.
    graph = persistent | persistent.T
    best, seen = empty, set()
    for start in sorted(accepted):
        if start in seen:
            continue
        component, todo = set(), [start]
        while todo:
            i = todo.pop()
            if i in component:
                continue
            component.add(i)
            todo.extend(j for j in accepted if graph[i, j] and j not in component)
        seen.update(component)
        ids = sorted(component)
        height = float(bounds[:, ids, 1, 2].max(axis=1).min())
        if height > best['height_m']:
            best = dict(height_m=height, object_indices=ids, object_count=len(ids))
    return best
