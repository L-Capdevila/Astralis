"""
metrics_sidecar.py — Pré-calcul et stockage des métriques d'analyse.

Pour chaque `simulation_X.parquet`, un sidecar `simulation_X_metrics.json`
contient les *scalaires exacts* calculés sur la totalité du dataset
(dérives énergétiques exactes, périodes, distances minimales par paire).

Le dashboard :
  • charge le sidecar instantanément (≈ 1 ms) si présent et frais ;
  • bascule sur un calcul live (sur DataFrame sous-échantillonné) sinon.

Format JSON (sans pickle). Écriture atomique (fichier .tmp + os.replace).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import json
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

SIDECAR_VERSION = 4
SIDECAR_SUFFIX = "_metrics.json"


def sidecar_path(parquet_path: str) -> str:
    base, _ = os.path.splitext(parquet_path)
    return base + SIDECAR_SUFFIX


def _to_jsonable(obj: Any) -> Any:
    """Convertit récursivement les types NumPy / tuples en types JSON natifs."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                json_key = ",".join(str(part) for part in key)
            else:
                json_key = key
            out[json_key] = _to_jsonable(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        value = obj.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def is_sidecar_fresh(parquet_path: str) -> bool:
    sp = sidecar_path(parquet_path)
    if not os.path.exists(sp):
        return False
    try:
        return os.path.getmtime(sp) >= os.path.getmtime(parquet_path)
    except OSError:
        return False


def load_sidecar(parquet_path: str) -> Optional[dict]:
    sp = sidecar_path(parquet_path)
    if not os.path.exists(sp):
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != SIDECAR_VERSION:
            return None
        return data
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def save_sidecar(parquet_path: str, summary: dict) -> str:
    payload = {
        "version": SIDECAR_VERSION,
        "parquet": os.path.basename(parquet_path),
        "summary": summary,
    }
    sp = sidecar_path(parquet_path)
    tmp = sp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False)
    os.replace(tmp, sp)
    return sp


def build_sidecar(parquet_path: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """
    Calcule le résumé exact depuis le parquet et écrit le sidecar.
    Retourne le chemin du sidecar, ou None en cas d'échec.
    """
    from core.metrics_core import (
        compute_exact_summary,
        compute_metrics,
        compute_orbital_elements,
    )

    if df is None:
        try:
            df = pd.read_parquet(parquet_path)
        except (OSError, ValueError):
            return None

    # Transmettre le softening réel utilisé pendant la simulation
    try:
        import config as _cfg
        _softening = float(_cfg.SOFTENING)
    except Exception:
        _softening = None

    summary = compute_exact_summary(df, softening=_softening)
    if summary is None:
        return None

    try:
        import config as _cfg
        summary["body_names"] = list(_cfg.BODY_NAMES)
        # Sauvegarder la relation parent pour que le dashboard reconnaisse
        # les satellites (ex: Lune → parent Terre).
        body_parents = {}
        for i, body in enumerate(_cfg.BODIES):
            pi = body.get("parent_index")
            if pi is not None and int(pi) >= 0:
                body_parents[i] = int(pi)
        if body_parents:
            summary["body_parents"] = body_parents
    except Exception:
        pass

    metrics = compute_metrics(df, softening=_softening)
    if metrics is None:
        return None

    body_idx = metrics.get("body_idx", [])
    orbital_elements_by_frame = {
        "barycentrique": compute_orbital_elements(
            metrics, frame_mode="barycentrique", central_idx=0
        ),
        "heliocentrique": {},
    }
    for central_idx in body_idx:
        orbital_elements_by_frame["heliocentrique"][int(central_idx)] = (
            compute_orbital_elements(
                metrics,
                frame_mode="heliocentrique",
                central_idx=int(central_idx),
            )
        )

    summary["full_metrics"] = {
        "body_idx": metrics["body_idx"],
        "temps": metrics["temps"],
        "temps_annees": metrics["temps_annees"],
        "positions_m": metrics["positions_m"],
        "positions_ua": metrics["positions_ua"],
        "velocity_components": metrics["velocity_components"],
        "speeds": metrics["speeds"],
        "pair_dist_ua": metrics["pair_dist_ua"],
        "e_kin_total": metrics["e_kin_total"],
        "e_pot_total": metrics["e_pot_total"],
        "e_total": metrics["e_total"],
        "derive_relative": metrics["derive_relative"],
        "mom_norm": metrics["mom_norm"],
        "mom_components_total": metrics["mom_components_total"],
        "l_total": metrics["l_total"],
        "l_total_xy": metrics["l_total_xy"],
        "radial_dist_ua": metrics["radial_dist_ua"],
        "derive_e": metrics["derive_e"],
        "derive_e_mean": metrics["derive_e_mean"],
        "derive_l": metrics["derive_l"],
        "periods": metrics["periods"],
        "eccentricities": metrics["eccentricities"],
        "n_rows": metrics["n_rows"],
        "orbital_elements": orbital_elements_by_frame,
    }

    try:
        return save_sidecar(parquet_path, summary)
    except OSError:
        return None
