"""
metrics_core.py — Calcul pur des métriques d'analyse (sans Streamlit).

Sépare les fonctions de calcul du dashboard pour qu'elles soient :
  • appelables depuis la simulation principale (génération du sidecar) ;
  • indépendantes de Streamlit (testables, importables sans surcoût) ;
  • réutilisables par tout outil d'analyse (multirun, tests, scripts).

Le dashboard expose des wrappers cachés, mais la logique numérique vit ici.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import re
from itertools import combinations

import numpy as np
import pandas as pd

import config


def detect_body_indices(df):
    idx = []
    for c in df.columns:
        m = re.match(r"^X(\d+)$", c)
        if m is not None:
            i = int(m.group(1))
            needed = [f"Y{i}", f"Z{i}", f"Vx{i}", f"Vy{i}", f"Vz{i}"]
            if all(col in df.columns for col in needed):
                idx.append(i)
    return sorted(idx)


def get_mass(i):
    if i < len(config.BODY_MASSES):
        return float(config.BODY_MASSES[i])
    return 1.0


def _body_parent_indices(body_idx):
    """Lit parent_index depuis settings.json / config (satellites)."""
    parents = {int(i): 0 for i in body_idx}
    try:
        import config as _cfg
        for i, body in enumerate(_cfg.BODIES):
            if i not in parents:
                continue
            pi = body.get("parent_index")
            parents[i] = int(pi) if pi is not None and int(pi) >= 0 else 0
    except Exception:
        pass
    return parents


def _kepler_periods_summary(df, body_idx):
    """Périodes Kepler (secondes) depuis la 1ère ligne enregistrée."""
    from core.periods import kepler_periods_from_dataframe
    masses = {i: get_mass(i) for i in body_idx}
    return kepler_periods_from_dataframe(
        df, body_idx, masses, central_idx=0, G=config.G, ua_m=config.UA
    )


def calc_period(x_col, y_col, t_col):
    passages = []
    y_arr = y_col.values if hasattr(y_col, "values") else np.asarray(y_col)
    x_arr = x_col.values if hasattr(x_col, "values") else np.asarray(x_col)
    t_arr = t_col.values if hasattr(t_col, "values") else np.asarray(t_col)
    for i in range(1, len(y_arr)):
        if y_arr[i - 1] < 0 and y_arr[i] >= 0 and x_arr[i] > 0:
            passages.append(t_arr[i])
    if len(passages) < 2:
        return None
    periodes = [passages[k + 1] - passages[k] for k in range(len(passages) - 1)]
    return float(np.mean(periodes))


def _downsample_df(df, n_max):
    """Sous-échantillonnage uniforme à n_max lignes (no-op si déjà petit)."""
    if n_max is None or len(df) <= n_max:
        return df
    idx = np.linspace(0, len(df) - 1, int(n_max), dtype=np.int64)
    return df.iloc[idx].reset_index(drop=True)


def compute_metrics(df, n_max=None, softening=None):
    """
    Calcule l'ensemble des métriques physiques utilisées par le dashboard.

    Si `n_max` est défini et que `df` est plus grand, le DataFrame est
    sous-échantillonné uniformément avant tout calcul. Les dérives
    (max - min) sur l'énergie/moment deviennent alors approximatives mais
    le coût bascule de O(N) à O(n_max).
    """
    df = _downsample_df(df, n_max)
    body_idx = detect_body_indices(df)
    if not body_idx:
        return None

    temps = df["Temps (jours)"]
    temps_annees = temps / 365.25
    positions_m = {}
    positions_ua = {}
    velocity_components = {}
    speeds = {}
    energy_kin = {}
    mom_components = {}
    mom_norm = {}
    pair_dist_ua = {}
    radial_dist_ua = {}

    for i in body_idx:
        x = df[f"X{i}"].values * config.UA
        y = df[f"Y{i}"].values * config.UA
        z = df[f"Z{i}"].values * config.UA
        x_ua = df[f"X{i}"].values
        y_ua = df[f"Y{i}"].values
        z_ua = df[f"Z{i}"].values
        vx = df[f"Vx{i}"].values
        vy = df[f"Vy{i}"].values
        vz = df[f"Vz{i}"].values
        m = get_mass(i)

        positions_m[i] = (x, y, z)
        positions_ua[i] = (x_ua, y_ua, z_ua)
        velocity_components[i] = (vx, vy, vz)
        speeds[i] = np.sqrt(vx * vx + vy * vy + vz * vz)
        energy_kin[i] = 0.5 * m * speeds[i] * speeds[i]
        radial_dist_ua[i] = np.sqrt(x_ua * x_ua + y_ua * y_ua + z_ua * z_ua)

        lx = m * (y * vz - z * vy)
        ly = m * (z * vx - x * vz)
        lz = m * (x * vy - y * vx)
        mom_components[i] = (lx, ly, lz)
        mom_norm[i] = np.sqrt(lx * lx + ly * ly + lz * lz)

    e_kin_total = np.zeros(len(df))
    for i in body_idx:
        e_kin_total += energy_kin[i]

    e_pot_total = np.zeros(len(df))
    for i, j in combinations(body_idx, 2):
        xi, yi, zi = positions_m[i]
        xj, yj, zj = positions_m[j]
        rij = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        pair_dist_ua[(i, j)] = rij / config.UA
        _soft = softening if softening is not None else config.SOFTENING
        rij_soft = np.sqrt(rij ** 2 + _soft ** 2)
        e_pot_total += -config.G * get_mass(i) * get_mass(j) / rij_soft

    e_total = e_kin_total + e_pot_total

    lx_total = np.zeros(len(df))
    ly_total = np.zeros(len(df))
    lz_total = np.zeros(len(df))
    for i in body_idx:
        lx, ly, lz = mom_components[i]
        lx_total += lx
        ly_total += ly
        lz_total += lz
    l_total = np.sqrt(lx_total * lx_total + ly_total * ly_total + lz_total * lz_total)
    l_total_xy = np.sqrt(lx_total * lx_total + ly_total * ly_total)

    ref_e = e_total[0]
    ref_l = l_total[0]
    # Dérive finale/initiale (pas max-min qui capture les oscillations orbitales)
    derive_e = abs((e_total[-1] - ref_e) / ref_e) * 100 \
        if abs(ref_e) > 1e-10 else float("nan")
    derive_l = abs((l_total[-1] - ref_l) / ref_l) * 100 \
        if abs(ref_l) > 1e-10 else float("nan")
    derive_relative = ((e_total - ref_e) / abs(ref_e) * 100
                       if abs(ref_e) > 1e-10 else np.zeros_like(e_total))
    # Dérive moyenne par an (cohérente avec derive_e)
    _t_ans = float(temps_annees.iloc[-1] if hasattr(temps_annees, "iloc") else temps_annees[-1])
    derive_e_mean = (derive_e / _t_ans
                     if (not np.isnan(derive_e) and _t_ans > 0)
                     else float("nan"))

    periods = {}
    periods_kepler_sec = {}
    eccentricities = {}
    eccentricities_helio = {}
    eccentricities_parent = {}
    parent_idx = _body_parent_indices(body_idx)
    kepler_init = _kepler_periods_summary(df, body_idx)
    for i in body_idx:
        periods[i] = calc_period(df[f"X{i}"], df[f"Y{i}"], temps)
        periods_kepler_sec[i] = kepler_init.get(i)
        r_body = radial_dist_ua[i]
        r_min = np.min(r_body)
        r_max = np.max(r_body)
        e_h = ((r_max - r_min) / (r_max + r_min)
               if (r_max + r_min) > 1e-12 else float("nan"))
        eccentricities[i] = e_h
        eccentricities_helio[i] = e_h
        p = parent_idx.get(i, 0)
        if i != p:
            key = (min(i, p), max(i, p))
            if key in pair_dist_ua:
                d = pair_dist_ua[key]
                d_min, d_max = float(np.min(d)), float(np.max(d))
                eccentricities_parent[i] = (
                    (d_max - d_min) / (d_max + d_min)
                    if (d_max + d_min) > 1e-12 else float("nan")
                )
            else:
                eccentricities_parent[i] = float("nan")
        else:
            eccentricities_parent[i] = float("nan")

    return {
        "body_idx": body_idx,
        "temps": temps,
        "temps_annees": temps_annees,
        "positions_m": positions_m,
        "positions_ua": positions_ua,
        "velocity_components": velocity_components,
        "e_kin_total": e_kin_total,
        "e_pot_total": e_pot_total,
        "e_total": e_total,
        "derive_relative": derive_relative,
        "mom_norm": mom_norm,
        "mom_components_total": (lx_total, ly_total, lz_total),
        "l_total": l_total,
        "l_total_xy": l_total_xy,
        "speeds": speeds,
        "radial_dist_ua": radial_dist_ua,
        "pair_dist_ua": pair_dist_ua,
        "derive_e": derive_e,
        "derive_e_mean": derive_e_mean,
        "derive_l": derive_l,
        "periods": periods,
        "periods_kepler_sec": periods_kepler_sec,
        "eccentricities": eccentricities,
        "eccentricities_helio": eccentricities_helio,
        "eccentricities_parent": eccentricities_parent,
        "body_parents": parent_idx,
        "n_rows": int(len(df)),
    }


def compute_orbital_elements(metrics, frame_mode="heliocentrique", central_idx=0):
    body_idx = metrics["body_idx"]
    n = len(metrics["temps"])
    if not body_idx:
        return {}

    masses = {i: get_mass(i) for i in body_idx}
    total_mass = sum(masses.values())

    frame_pos = {}
    frame_vel = {}
    for i in body_idx:
        px, py, pz = metrics["positions_m"][i]
        vx, vy, vz = metrics["velocity_components"][i]
        frame_pos[i] = np.column_stack((px, py, pz))
        frame_vel[i] = np.column_stack((vx, vy, vz))

    if frame_mode == "barycentrique":
        ref_pos = np.zeros((n, 3), dtype=float)
        ref_vel = np.zeros((n, 3), dtype=float)
        for i in body_idx:
            mi = masses[i]
            ref_pos += mi * frame_pos[i]
            ref_vel += mi * frame_vel[i]
        if total_mass > 0:
            ref_pos /= total_mass
            ref_vel /= total_mass
    else:
        ref_pos = frame_pos.get(central_idx, np.zeros((n, 3), dtype=float))
        ref_vel = frame_vel.get(central_idx, np.zeros((n, 3), dtype=float))

    elements = {}
    for i in body_idx:
        if frame_mode == "heliocentrique" and i == central_idx:
            elements[i] = {
                "r_ua": np.zeros(n),
                "a_ua": np.full(n, np.nan),
                "e": np.full(n, np.nan),
                "i_deg": np.full(n, np.nan),
            }
            continue

        r = frame_pos[i] - ref_pos
        v = frame_vel[i] - ref_vel
        r_norm = np.linalg.norm(r, axis=1)
        v_sq = np.sum(v * v, axis=1)

        if frame_mode == "barycentrique":
            mu = config.G * max(total_mass, 1.0)
        else:
            m_central = masses.get(central_idx, 0.0)
            mu = config.G * max(m_central + masses[i], 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.cross(r, v)
            h_norm = np.linalg.norm(h, axis=1)
            specific_energy = 0.5 * v_sq - mu / r_norm
            a = -mu / (2.0 * specific_energy)
            e_vec = np.cross(v, h) / mu - (r / r_norm[:, None])
            e = np.linalg.norm(e_vec, axis=1)
            i_deg = np.degrees(np.arccos(np.clip(h[:, 2] / h_norm, -1.0, 1.0)))

        invalid = ((r_norm < 1e-12) | (~np.isfinite(a))
                   | (~np.isfinite(e)) | (~np.isfinite(i_deg)))
        a[invalid] = np.nan
        e[invalid] = np.nan
        i_deg[invalid] = np.nan

        elements[i] = {
            "r_ua": r_norm / config.UA,
            "a_ua": a / config.UA,
            "e": e,
            "i_deg": i_deg,
        }

    return elements


def compute_exact_summary(df, softening=None):
    """
    Calcule les *scalaires* exacts (derive_e, derive_l, périodes, distances
    minimales par paire) sur la totalité du DataFrame, en O(N) sans stocker
    de gros arrays. Conçu pour le sidecar et l'overlay des chiffres exacts
    dans le dashboard, même quand l'affichage utilise un échantillonnage.
    """
    body_idx = detect_body_indices(df)
    if not body_idx:
        return None

    temps = df["Temps (jours)"].values
    e_kin = np.zeros(len(df))
    lx_tot = np.zeros(len(df))
    ly_tot = np.zeros(len(df))
    lz_tot = np.zeros(len(df))
    radial = {}
    pair_min = {}
    pair_min_t = {}
    pair_dist_ua_stats = {}

    pos_m = {}
    for i in body_idx:
        x = df[f"X{i}"].values * config.UA
        y = df[f"Y{i}"].values * config.UA
        z = df[f"Z{i}"].values * config.UA
        vx = df[f"Vx{i}"].values
        vy = df[f"Vy{i}"].values
        vz = df[f"Vz{i}"].values
        m = get_mass(i)
        speeds_sq = vx * vx + vy * vy + vz * vz
        e_kin += 0.5 * m * speeds_sq
        lx_tot += m * (y * vz - z * vy)
        ly_tot += m * (z * vx - x * vz)
        lz_tot += m * (x * vy - y * vx)
        radial[i] = (float(np.min(np.sqrt(
            df[f"X{i}"].values ** 2
            + df[f"Y{i}"].values ** 2
            + df[f"Z{i}"].values ** 2))),
                     float(np.max(np.sqrt(
            df[f"X{i}"].values ** 2
            + df[f"Y{i}"].values ** 2
            + df[f"Z{i}"].values ** 2))))
        pos_m[i] = (x, y, z)

    e_pot = np.zeros(len(df))
    for i, j in combinations(body_idx, 2):
        xi, yi, zi = pos_m[i]
        xj, yj, zj = pos_m[j]
        rij = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        _soft = softening if softening is not None else config.SOFTENING
        rij_soft = np.sqrt(rij ** 2 + _soft ** 2)
        e_pot += -config.G * get_mass(i) * get_mass(j) / rij_soft
        kmin = int(np.argmin(rij))
        rij_ua = rij / config.UA
        pair_min[(i, j)] = float(rij_ua[kmin])
        pair_min_t[(i, j)] = float(temps[kmin] / 365.25)
        pair_dist_ua_stats[(i, j)] = (float(np.min(rij_ua)), float(np.max(rij_ua)))

    e_total = e_kin + e_pot
    l_total = np.sqrt(lx_tot * lx_tot + ly_tot * ly_tot + lz_tot * lz_tot)

    ref_e = e_total[0]
    ref_l = l_total[0]
    # Dérive finale/initiale (pas max-min qui capture les oscillations orbitales)
    derive_e = abs((e_total[-1] - ref_e) / ref_e) * 100 \
        if abs(ref_e) > 1e-10 else float("nan")
    derive_l = abs((l_total[-1] - ref_l) / ref_l) * 100 \
        if abs(ref_l) > 1e-10 else float("nan")
    # Dérive moyenne par an (cohérente avec derive_e)
    _duree_ans = float(temps[-1]) / 365.25 if len(temps) else 0.0
    derive_e_mean = (derive_e / _duree_ans
                     if (not np.isnan(derive_e) and _duree_ans > 0)
                     else float("nan"))

    periods = {}
    periods_kepler_sec = {}
    eccentricities = {}
    eccentricities_helio = {}
    eccentricities_parent = {}
    parent_idx = _body_parent_indices(body_idx)
    kepler_init = _kepler_periods_summary(df, body_idx)
    for i in body_idx:
        periods[i] = calc_period(df[f"X{i}"], df[f"Y{i}"], df["Temps (jours)"])
        periods_kepler_sec[i] = kepler_init.get(i)
        r_min, r_max = radial[i]
        e_h = ((r_max - r_min) / (r_max + r_min)
               if (r_max + r_min) > 1e-12 else float("nan"))
        eccentricities[i] = e_h
        eccentricities_helio[i] = e_h
        p = parent_idx.get(i, 0)
        if i != p:
            key = (min(i, p), max(i, p))
            if key in pair_dist_ua_stats:
                d_min, d_max = pair_dist_ua_stats[key]
                eccentricities_parent[i] = (
                    (d_max - d_min) / (d_max + d_min)
                    if (d_max + d_min) > 1e-12 else float("nan")
                )
            else:
                eccentricities_parent[i] = float("nan")
        else:
            eccentricities_parent[i] = float("nan")

    return {
        "body_idx":       body_idx,
        "n_rows":         int(len(df)),
        "duree_ans":      float(temps[-1] / 365.25) if len(temps) else 0.0,
        "derive_e":       float(derive_e),
        "derive_l":       float(derive_l),
        "derive_e_mean":  float(derive_e_mean),
        "periods":        periods,
        "periods_kepler_sec": periods_kepler_sec,
        "eccentricities": eccentricities,
        "eccentricities_helio": eccentricities_helio,
        "eccentricities_parent": eccentricities_parent,
        "body_parents": parent_idx,
        "pair_min_ua":    pair_min,
        "pair_min_t_ans": pair_min_t,
        "radial_min_max_ua": radial,
    }
