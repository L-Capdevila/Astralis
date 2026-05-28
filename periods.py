"""
periods.py — Périodes orbitales (Kepler) et de rotation propre.

Utilisé par le dashboard (configuration + résumé) et par metrics_core.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


def format_period_seconds(
    T_sec: Optional[float],
    none_label: str = "—",
) -> str:
    """Affiche une durée lisible (s, h, jours, ans)."""
    if T_sec is None or not math.isfinite(T_sec) or T_sec <= 0:
        return none_label
    days = T_sec / SECONDS_PER_DAY
    if days >= 365.25:
        years = days / 365.25
        return f"{years:.4f} ans ({days:.1f} j)"
    if days >= 1.0:
        return f"{days:.3f} j ({T_sec / 3600.0:.2f} h)"
    if T_sec >= 60.0:
        return f"{T_sec / 3600.0:.3f} h ({T_sec:.1f} s)"
    return f"{T_sec:.2f} s"


from core.body_init import resolve_body_positions
from core.body_init import resolve_body_velocities as _resolve_body_velocities


def resolve_body_velocities(bodies: Sequence[dict], G: float) -> List[List[float]]:
    """Vitesses initiales + correction barycentrique (comme config.py)."""
    abs_pos = resolve_body_positions(bodies)
    return _resolve_body_velocities(bodies, abs_pos, G)


def _orbit_parent_index(body: dict, body_index: int) -> int:
    pi = body.get("parent_index")
    if pi is not None and int(pi) >= 0:
        return min(int(pi), body_index - 1) if body_index > 0 else 0
    return 0 if body_index > 0 else 0


def orbital_period_sec(
    pos_rel: np.ndarray,
    vel_rel: np.ndarray,
    m_central: float,
    m_body: float,
    G: float,
) -> Optional[float]:
    """
    Période de Kepler pour une orbite liée (ε < 0).
    μ = G·(M + m), a = -μ/(2ε), T = 2π√(a³/μ).
    """
    r = float(np.linalg.norm(pos_rel))
    if r < 1.0:
        return None
    mu = G * (m_central + m_body)
    if mu <= 0:
        return None
    v2 = float(np.dot(vel_rel, vel_rel))
    eps = 0.5 * v2 - mu / r
    if eps >= -1e-18:
        return None
    a = -mu / (2.0 * eps)
    if a <= 0 or not math.isfinite(a):
        return None
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def circular_period_sec(r_m: float, m_central: float, G: float) -> Optional[float]:
    """Estimation circulaire T = 2π√(r³/(G·M)) si orbite non liée."""
    if r_m < 1.0 or m_central <= 0:
        return None
    return 2.0 * math.pi * math.sqrt(r_m ** 3 / (G * m_central))


def spin_period_sec(spin_rate_rad_s: float) -> Optional[float]:
    """Période de rotation propre T = 2π/|ω|."""
    w = abs(float(spin_rate_rad_s))
    if w < 1e-30:
        return None
    return 2.0 * math.pi / w


def compute_body_periods(
    bodies: Sequence[dict],
    G: float,
    central_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Périodes estimées pour chaque corps (config dashboard)."""
    if not bodies:
        return []
    central_idx = int(central_idx)
    if central_idx < 0 or central_idx >= len(bodies):
        central_idx = 0

    abs_pos = resolve_body_positions(bodies)
    vels = resolve_body_velocities(bodies, G)

    out: List[Dict[str, Any]] = []
    for i, body in enumerate(bodies):
        name = str(body.get("name", f"Corps {i}"))
        entry: Dict[str, Any] = {
            "index": i,
            "name": name,
            "orbital_sec": None,
            "orbital_label": "—",
            "orbital_hint": "",
            "rotation_sec": None,
            "rotation_label": "—",
            "rotation_hint": "",
        }

        T_rot = spin_period_sec(float(body.get("spin_rate", 0.0)))
        if T_rot is not None:
            entry["rotation_sec"] = T_rot
            entry["rotation_label"] = format_period_seconds(T_rot)
            entry["rotation_hint"] = "Rotation propre : T = 2π / |ω| (rad/s)."
        else:
            entry["rotation_hint"] = "Aucune rotation propre (ω = 0)."

        if i == central_idx:
            entry["orbital_label"] = "— (référence)"
            entry["orbital_hint"] = "Corps central : référence pour les orbites des autres corps."
        else:
            ref_idx = _orbit_parent_index(body, i)
            pos = np.array(abs_pos[i], dtype=float)
            vel = np.array(vels[i], dtype=float)
            pos_c = np.array(abs_pos[ref_idx], dtype=float)
            vel_c = np.array(vels[ref_idx], dtype=float)
            m_c = float(bodies[ref_idx].get("mass", 1.0))
            central_name = str(bodies[ref_idx].get("name", "parent"))
            r_rel = pos - pos_c
            v_rel = vel - vel_c
            T = orbital_period_sec(r_rel, v_rel, m_c, float(body.get("mass", 1.0)), G)
            if T is not None:
                entry["orbital_sec"] = T
                entry["orbital_label"] = format_period_seconds(T)
                entry["orbital_hint"] = (
                    f"Orbite liée autour de « {central_name} » "
                    "(Kepler depuis position et vitesse initiales)."
                )
            else:
                r = float(np.linalg.norm(r_rel))
                Tc = circular_period_sec(r, m_c, G)
                if Tc is not None:
                    entry["orbital_sec"] = Tc
                    entry["orbital_label"] = (
                        format_period_seconds(Tc) + " (circ. approx.)"
                    )
                    entry["orbital_hint"] = (
                        f"Orbite non liée ou parabolique : estimation circulaire "
                        f"autour de « {central_name} »."
                    )
                else:
                    entry["orbital_label"] = "indéterminée"
                    entry["orbital_hint"] = "Impossible d'estimer la période orbitale."

        out.append(entry)
    return out


def kepler_periods_from_dataframe(
    df,
    body_idx: Sequence[int],
    masses: Dict[int, float],
    central_idx: int = 0,
    G: float = 6.674e-11,
    ua_m: float = 1.496e11,
) -> Dict[int, Optional[float]]:
    """Périodes Kepler (s) depuis la première ligne du parquet (état initial enregistré)."""
    if df is None or len(df) == 0:
        return {}
    row = df.iloc[0]
    central_idx = int(central_idx)
    if central_idx not in body_idx:
        return {}

    pos_c = np.array([
        float(row[f"X{central_idx}"]) * ua_m,
        float(row[f"Y{central_idx}"]) * ua_m,
        float(row[f"Z{central_idx}"]) * ua_m,
    ])
    vz0 = f"Vz{central_idx}"
    vel_c = np.array([
        float(row[f"Vx{central_idx}"]),
        float(row[f"Vy{central_idx}"]),
        float(row[vz0]) if vz0 in df.columns else 0.0,
    ])

    m_c = float(masses.get(central_idx, 1.0))
    periods: Dict[int, Optional[float]] = {}
    for i in body_idx:
        if i == central_idx:
            periods[i] = None
            continue
        pos = np.array([
            float(row[f"X{i}"]) * ua_m,
            float(row[f"Y{i}"]) * ua_m,
            float(row[f"Z{i}"]) * ua_m,
        ])
        vzi = f"Vz{i}"
        vel = np.array([
            float(row[f"Vx{i}"]),
            float(row[f"Vy{i}"]),
            float(row[vzi]) if vzi in df.columns else 0.0,
        ])
        T = orbital_period_sec(
            pos - pos_c, vel - vel_c, m_c, float(masses.get(i, 1.0)), G
        )
        periods[i] = T
    return periods


def periods_days_to_seconds(days: Optional[float]) -> Optional[float]:
    if days is None:
        return None
    try:
        d = float(days)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d <= 0:
        return None
    return d * SECONDS_PER_DAY
