"""
body_init.py — Positions et vitesses initiales (satellites : parent + pos_rel).

Source unique pour config.py, periods.py et la simulation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from core.body_positions import _parent_index, resolve_body_positions, sync_absolute_positions
from core.celestial_velocities import (
    manual_velocity_with_inclination,
    relative_circular_velocity,
)


def orbit_plane_normal(
    ref_pos: Sequence[float],
    ref_vel: Sequence[float],
) -> List[float]:
    """Normale au plan orbital (r × v). Retombe sur Z global si indéterminé."""
    rx, ry, rz = float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])
    vx, vy, vz = float(ref_vel[0]), float(ref_vel[1]), float(ref_vel[2])
    nx = ry * vz - rz * vy
    ny = rz * vx - rx * vz
    nz = rx * vy - ry * vx
    n = (nx * nx + ny * ny + nz * nz) ** 0.5
    if n < 1e-30:
        return [0.0, 0.0, 1.0]
    return [nx / n, ny / n, nz / n]


def _relative_position(
    body: dict,
    body_index: int,
    abs_positions: Sequence[Sequence[float]],
) -> List[float]:
    pidx = _parent_index(body, body_index)
    if pidx is not None and "pos_rel" in body:
        rel = body["pos_rel"]
        return [
            float(rel[k]) if k < len(rel) else 0.0
            for k in range(3)
        ]
    if pidx is not None:
        return [
            abs_positions[body_index][k] - abs_positions[pidx][k]
            for k in range(3)
        ]
    return [float(abs_positions[body_index][k]) for k in range(3)]


def resolve_body_velocities(
    bodies: Sequence[dict],
    abs_positions: Sequence[Sequence[float]],
    G: float,
) -> List[List[float]]:
    """
    Vitesses absolues initiales.
    Satellites : v_abs = v_parent + v_orbite_locale (auto ou manuelle).
    """
    n = len(bodies)
    vels: List[List[float]] = []
    for i, body in enumerate(bodies):
        if i == 0:
            vm = body.get("vel_manual", body.get("vel", [0.0, 0.0, 0.0]))
            vels.append([float(vm[0]), float(vm[1]), float(vm[2])])
            continue

        pidx = _parent_index(body, i)
        if body.get("use_auto_vel", False):
            if pidx is None:
                pidx = 0
            pidx = min(max(int(pidx), 0), i - 1)
            rel = _relative_position(body, i, abs_positions)

            # Normale du plan orbital du satellite :
            # On prend la position et vitesse RELATIVES du parent par rapport à
            # son propre parent (ex : Terre/Soleil pour la Lune).
            # Cela donne le bon plan de référence pour l'inclinaison du satellite.
            # On utilise des vitesses relatives pour s'affranchir du biais
            # barycentrique (correction pas encore appliquée à ce stade).
            grandparent_idx = _parent_index(bodies[pidx], pidx) if pidx > 0 else None
            if grandparent_idx is None:
                grandparent_idx = 0
            rp = [
                abs_positions[pidx][k] - abs_positions[grandparent_idx][k]
                for k in range(3)
            ]
            rv = [
                vels[pidx][k] - vels[grandparent_idx][k]
                for k in range(3)
            ]
            plane_n = orbit_plane_normal(rp, rv)
            # Fallback : plan XY si parent au repos ou positions dégénérées
            if all(abs(x) < 1e-30 for x in plane_n):
                plane_n = [0.0, 0.0, 1.0]
            v_rel = relative_circular_velocity(
                rel,
                body.get("incl_deg", 0.0),
                body.get("sens", 1),
                G,
                float(bodies[pidx].get("mass", 1.0)),
                float(body.get("mass", 1.0)),
                plane_normal=plane_n,
            )
            vels.append([
                vels[pidx][k] + v_rel[k]
                for k in range(3)
            ])
        else:
            v_loc = manual_velocity_with_inclination(
                body.get("vel_manual", body.get("vel", [0.0, 0.0, 0.0])),
                body.get("incl_deg", 0.0),
                body.get("sens", 1),
            )
            if pidx is not None:
                vels.append([
                    vels[pidx][k] + v_loc[k]
                    for k in range(3)
                ])
            else:
                vels.append([float(v_loc[0]), float(v_loc[1]), float(v_loc[2])])

    return apply_barycentric_correction(bodies, vels)


def apply_barycentric_correction(
    bodies: Sequence[dict],
    vels: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Soustrait la vitesse du centre de masse (barycentre approx. fixe)."""
    n = len(bodies)
    if n == 0:
        return []
    total_m = sum(float(b.get("mass", 1.0)) for b in bodies)
    if total_m <= 0:
        return [list(v) for v in vels]
    vcm = [
        sum(float(bodies[i].get("mass", 1.0)) * float(vels[i][k]) for i in range(n)) / total_m
        for k in range(3)
    ]
    return [
        [float(vels[i][k]) - vcm[k] for k in range(3)]
        for i in range(n)
    ]


def body_parents_map(bodies: Sequence[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, body in enumerate(bodies):
        pidx = _parent_index(body, i)
        out[i] = pidx if pidx is not None else 0
    return out


__all__ = [
    "resolve_body_positions",
    "sync_absolute_positions",
    "resolve_body_velocities",
    "apply_barycentric_correction",
    "orbit_plane_normal",
    "body_parents_map",
]
