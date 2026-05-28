"""
celestial_velocities.py — Vitesses circulaires 3D et projection d'inclinaison.

Convention incl_deg : inclinaison du plan orbital par rapport au plan XY global,
rotation du vecteur tangent autour de l'axe séparation (position relative).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Optional


def _vec_cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vec_scale(a: Sequence[float], s: float) -> List[float]:
    return [a[0] * s, a[1] * s, a[2] * s]


def _rodrigues(v: Sequence[float], k_hat: Sequence[float], angle_rad: float) -> List[float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    cross = _vec_cross(k_hat, v)
    dot = k_hat[0] * v[0] + k_hat[1] * v[1] + k_hat[2] * v[2]
    return [
        v[0] * c + cross[0] * s + k_hat[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + k_hat[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + k_hat[2] * dot * (1.0 - c),
    ]


def relative_circular_velocity(
    rel_pos: Sequence[float],
    incl_deg: float,
    sens: float,
    g_const: float,
    m_parent: float,
    m_child: float = 0.0,
    plane_normal: Optional[Sequence[float]] = None,
) -> List[float]:
    """
    Vitesse circulaire autour du parent, perpendiculaire à rel_pos,
    avec inclinaison incl_deg (degrés) et sens ±1 (prograde/rétrograde).

    plane_normal : normale du plan de référence (ex. plan orbital du parent
    autour du Soleil). Par défaut : axe Z global.
    """
    r = _vec_norm(rel_pos)
    r = max(r, 1.0)
    mu = g_const * (float(m_parent) + float(m_child))
    speed = math.sqrt(mu / r)

    rx, ry, rz = rel_pos[0] / r, rel_pos[1] / r, rel_pos[2] / r
    r_hat = [rx, ry, rz]

    if plane_normal is not None:
        pn = _vec_norm(plane_normal)
        if pn > 1e-30:
            n_hat = [plane_normal[0] / pn, plane_normal[1] / pn, plane_normal[2] / pn]
        else:
            n_hat = [0.0, 0.0, 1.0]
    else:
        n_hat = [0.0, 0.0, 1.0]

    t0 = _vec_cross(n_hat, r_hat)
    tn = _vec_norm(t0)
    if tn < 1e-30:
        # rel_pos // normale : fallback perpendiculaire stable
        t0 = _vec_cross(r_hat, [1.0, 0.0, 0.0] if abs(r_hat[0]) < 0.9 else [0.0, 1.0, 0.0])
        tn = _vec_norm(t0)
    t0 = _vec_scale(t0, 1.0 / tn)

    inc = math.radians(float(incl_deg))
    t = _rodrigues(t0, r_hat, inc)
    sign = float(sens) if abs(float(sens)) >= 1e-30 else 1.0
    return _vec_scale(t, sign * speed)


def auto_orbit_velocity(
    radius_m: float, incl_deg: float, sens: float, g_const: float, m_central: float
) -> List[float]:
    """Orbite circulaire autour de l'origine (séparation selon +X)."""
    return relative_circular_velocity(
        [float(radius_m), 0.0, 0.0], incl_deg, sens, g_const, m_central, 0.0
    )


def manual_velocity_with_inclination(
    vel_manual: Sequence[float],
    incl_deg: float,
    sens: float,
    fix_missing_z: bool = True,
) -> List[float]:
    """
    Si incl_deg ≠ 0 et la vitesse manuelle n'a pas de composante Z cohérente
    (cas typique : tout sur Vy), répartit |v| sur Vy/Vz via cos/sin(incl).
  """
    vx, vy, vz = (float(vel_manual[0]), float(vel_manual[1]), float(vel_manual[2]))
    inc = abs(float(incl_deg))
    if inc < 1e-12:
        return [vx, vy, vz]

    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-30:
        return [0.0, 0.0, 0.0]

    inc_r = math.radians(inc)
    expected_vz = abs(speed * math.sin(inc_r))
    sign = float(sens) if abs(float(sens)) >= 1e-30 else 1.0

    if fix_missing_z and expected_vz > 1.0 and abs(vz) < 0.05 * expected_vz:
        return [vx, sign * speed * math.cos(inc_r), sign * speed * math.sin(inc_r)]

    return [vx, vy, vz]
