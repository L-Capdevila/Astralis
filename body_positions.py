"""
body_positions.py — Positions absolues des corps (satellites : parent + pos_rel).
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def _parent_index(body: dict, body_index: int) -> Optional[int]:
    pi = body.get("parent_index")
    if pi is None:
        return None
    try:
        pidx = int(pi)
    except (TypeError, ValueError):
        return None
    if pidx < 0 or pidx >= body_index:
        return None
    return pidx


def resolve_body_positions(bodies: Sequence[dict]) -> List[List[float]]:
    """
    Retourne les positions absolues (m) de chaque corps.
    Si parent_index et pos_rel sont définis : pos_abs = pos_parent + pos_rel.
    """
    positions: List[List[float]] = []
    for body in bodies:
        raw = body.get("pos", [0.0, 0.0, 0.0])
        positions.append([
            float(raw[0]) if len(raw) > 0 else 0.0,
            float(raw[1]) if len(raw) > 1 else 0.0,
            float(raw[2]) if len(raw) > 2 else 0.0,
        ])

    for i, body in enumerate(bodies):
        pidx = _parent_index(body, i)
        if pidx is None or "pos_rel" not in body:
            continue
        rel = body["pos_rel"]
        positions[i] = [
            positions[pidx][k] + (float(rel[k]) if k < len(rel) else 0.0)
            for k in range(3)
        ]
    return positions


def sync_absolute_positions(bodies: list) -> None:
    """Met à jour le champ pos de chaque satellite (parent + pos_rel)."""
    resolved = resolve_body_positions(bodies)
    for i, body in enumerate(bodies):
        if not isinstance(body, dict):
            continue
        if _parent_index(body, i) is not None and "pos_rel" in body:
            body["pos"] = resolved[i]
