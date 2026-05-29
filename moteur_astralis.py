"""
moteur_astralis.py — Moteur de simulation N-corps 3D.

Moteur : intégrateur symplectique Yoshida 4ème ordre (integrator.py)
         forces découplées (forces.py)
         conteneur SoA JIT-compilé (state.py).

Visualisation temps réel (optionnelle) :
  --realtime  : fenêtre matplotlib en direct
  --web       : interface Three.js dans le navigateur (http://localhost:5050)
  Les deux flags peuvent être combinés.

CORRECTIONS :
  #5  : last_H et last_L sont maintenant recalculés à chaque SAVE_EVERY,
        garantissant que le CSV contient toujours les valeurs courantes et
        non des valeurs figées sur MONITOR_EVERY frames.
  #8  : REALTIME_EVERY lu directement depuis le module `config` importé,
        sans antipattern __import__.
"""

import sys
import argparse
import json
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import subprocess

import numba

import config                         # module entier — #8 corrigé
from config import (
    G, UA,
    DT_MIN, DT_MAX, SOFTENING,
    DIST_SEUIL, ALPHA,
    BODY_NAMES, BODY_MASSES, BODY_POSITIONS, BODY_VELOCITIES,
    BODY_J2R2, BODY_MDOT, BODY_SPIN, BODY_K2, NBODIES,
    BODY_RADII,
    CHEMIN_SAUVEGARDE,
    APP_DIR,
    RUNTIME_DIR,
    DUREE_ANS, SAVE_EVERY, MONITOR_EVERY,
    CHECKPOINT_EVERY_ANS, CHECKPOINT_DIR,
    ENABLE_PN1, C_LIGHT,
    NUM_THREADS, OUTPUT_NAME,
)
from core.state      import SystemState
from core.forces     import (compute_forces, compute_min_dist,
                             compute_adaptive_dt_from_min_dist)
from core.integrator import yoshida4_step, compute_potential_energy
from core.monitor    import (compute_hamiltonian, compute_angular_momentum,
                             compute_diagnostics, summarize_drift_series)

# CORRECTION BUG #8 : lecture directe depuis le module, pas via __import__
REALTIME_EVERY = getattr(config, 'REALTIME_EVERY', 50)
FLUSH_EVERY = getattr(config, 'FLUSH_EVERY', 50000)

def _sanitize_output_basename(name: str) -> str:
    """Nom de fichier sûr pour le Parquet de sortie."""
    name = (name or "").strip()
    if not name:
        return ""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:120]


def _build_parquet_path(output_name: str) -> str:
    """Chemin complet du fichier Parquet (préfixe personnalisé + horodatage)."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = _sanitize_output_basename(output_name)
    fname = f"{label}_{ts}.parquet" if label else f"simulation_{ts}.parquet"
    return os.path.join(CHEMIN_SAUVEGARDE, fname)


# ══════════════════════════════════════════════════════════════════════════════
#  THREADS NUMBA
# ══════════════════════════════════════════════════════════════════════════════

def configure_numba_threads(num_threads: int = 0) -> int:
    """
    Alloue les threads Numba.
    num_threads > 0 : valeur imposée (depuis settings.json / dashboard).
    num_threads == 0 : réglage automatique selon les cœurs disponibles.
    """
    cores = os.cpu_count() or 1
    nt = num_threads or NUM_THREADS
    if nt and nt > 0:
        n_threads = min(int(nt), cores)
    elif cores <= 2:
        n_threads = 1
    elif cores <= 4:
        n_threads = cores - 1
    else:
        n_threads = cores - 2
    n_threads = max(1, n_threads)
    numba.set_num_threads(n_threads)
    return n_threads


# ══════════════════════════════════════════════════════════════════════════════
#  FLUSH PARQUET INCRÉMENTAL
# ══════════════════════════════════════════════════════════════════════════════

def _build_output_columns(n: int):
    columns = ['Frame', 'Temps (jours)']
    for i in range(n):
        columns.extend([f'X{i}', f'Y{i}', f'Z{i}'])
    for i in range(n):
        columns.extend([f'Vx{i}', f'Vy{i}', f'Vz{i}'])
    columns.extend(['E_totale', 'L_total', 'dt'])
    return columns


def _flush_data_log(data_log, columns, chemin_parquet, part_index):
    """Écrit un morceau de data_log sur disque puis vide la RAM."""
    if not data_log:
        return part_index
    df = pd.DataFrame(data_log, columns=columns)
    part_path = f"{chemin_parquet}.part{part_index:05d}.parquet"
    df.to_parquet(part_path, index=False)
    data_log.clear()
    return part_index + 1


def _finalize_parquet(chemin_parquet, data_log, columns):
    """Fusionne les morceaux .part*.parquet et le reliquat RAM en un seul fichier."""
    parts = sorted(glob.glob(f"{chemin_parquet}.part*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    if data_log:
        dfs.append(pd.DataFrame(data_log, columns=columns))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(chemin_parquet) or ".", exist_ok=True)
    df.to_parquet(chemin_parquet, index=False)
    for p in parts:
        os.remove(p)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def build_state() -> SystemState:
    """Construit l'état initial en relisant settings.json (positions satellites incluses)."""
    import importlib
    import config as cfg
    importlib.reload(cfg)

    s = SystemState(cfg.NBODIES)
    for i in range(cfg.NBODIES):
        s.masses[i]   = cfg.BODY_MASSES[i]
        s.j2r2[i]     = cfg.BODY_J2R2[i]
        s.mdot[i]     = cfg.BODY_MDOT[i]
        s.k2[i]       = cfg.BODY_K2[i]
        s.spin[i, 0]  = cfg.BODY_SPIN[i][0]
        s.spin[i, 1]  = cfg.BODY_SPIN[i][1]
        s.spin[i, 2]  = cfg.BODY_SPIN[i][2]
        s.pos[i, 0]   = cfg.BODY_POSITIONS[i][0]
        s.pos[i, 1]   = cfg.BODY_POSITIONS[i][1]
        s.pos[i, 2]   = cfg.BODY_POSITIONS[i][2]
        s.vel[i, 0]   = cfg.BODY_VELOCITIES[i][0]
        s.vel[i, 1]   = cfg.BODY_VELOCITIES[i][1]
        s.vel[i, 2]   = cfg.BODY_VELOCITIES[i][2]

    accel, _ = compute_forces(s, cfg.G, cfg.SOFTENING)
    s.accel[:, :] = accel
    s.E0 = compute_hamiltonian(s, cfg.G, cfg.SOFTENING)
    s.U_ref = abs(compute_potential_energy(s, cfg.G, cfg.SOFTENING))
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION DE COLLISION
# ══════════════════════════════════════════════════════════════════════════════

def check_collision(dist_min: float, i: int, j: int):
    """
    Détection de collision basée sur les rayons physiques individuels.
    Collision si dist_min < rayon_i + rayon_j.
    Les rayons sont définis par corps dans settings.json (champ "rayon").
    """
    seuil = BODY_RADII[i] + BODY_RADII[j]
    if dist_min < seuil:
        return i, j
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT / REPRISE
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_FILENAME = "checkpoint_latest.npz"
META_FILENAME = "checkpoint_latest.meta.json"
LAST_RUN_FILENAME = "last_run.json"


def save_last_run(project_dir: str, parquet_path: str, status: str = "done"):
    """Pointeur vers le dernier .parquet produit (lu par le dashboard)."""
    if not parquet_path:
        return
    path = os.path.join(project_dir, LAST_RUN_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "parquet": os.path.abspath(parquet_path),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    os.replace(tmp, path)


def load_last_run(project_dir: str):
    path = os.path.join(project_dir, LAST_RUN_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("parquet", "")
        return p if p and os.path.isfile(p) else None
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_checkpoint_dir():
    if os.path.isabs(CHECKPOINT_DIR):
        return CHECKPOINT_DIR
    return os.path.join(CHEMIN_SAUVEGARDE, CHECKPOINT_DIR)


def save_checkpoint(s, frame, ckpt_dir):
    """
    Checkpoint léger : état instantané uniquement (pas d'historique data_log).
    Format NumPy .npz (sans pickle). Écriture atomique via .tmp + os.replace().

    Note : np.savez ajoute automatiquement l'extension .npz — le fichier temporaire
    ne doit pas se terminer par .npz.tmp (sinon -> .npz.tmp.npz introuvable au replace).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, CHECKPOINT_FILENAME)
    tmp_stem = os.path.join(ckpt_dir, "_checkpoint_latest_tmp")
    tmp_file = tmp_stem + ".npz"
    if os.path.isfile(tmp_file):
        try:
            os.remove(tmp_file)
        except OSError:
            pass
    np.savez(
        tmp_stem,
        frame=np.int64(frame),
        t=np.float64(s.t),
        s=np.float64(s.s),
        U_ref=np.float64(s.U_ref),
        E0=np.float64(s.E0),
        pos=np.asarray(s.pos, dtype=np.float64),
        vel=np.asarray(s.vel, dtype=np.float64),
        spin=np.asarray(s.spin, dtype=np.float64),
    )
    if not os.path.isfile(tmp_file):
        raise FileNotFoundError(f"Fichier checkpoint temporaire absent : {tmp_file}")
    os.replace(tmp_file, path)
    return path


def save_run_meta(ckpt_dir, chemin_parquet, part_index):
    """Métadonnées légères pour reprendre le flush Parquet après un crash."""
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {
        "chemin_parquet": chemin_parquet,
        "part_index":     int(part_index),
    }
    path = os.path.join(ckpt_dir, META_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, path)


def load_run_meta(ckpt_dir):
    path = os.path.join(ckpt_dir, META_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_run_meta(ckpt_dir):
    path = os.path.join(ckpt_dir, META_FILENAME)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def load_checkpoint(ckpt_dir):
    """Charge le dernier checkpoint .npz si disponible. Retourne None sinon."""
    path = os.path.join(ckpt_dir, CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                "frame": int(data["frame"]),
                "t":     float(data["t"]),
                "s":     float(data["s"]),
                "U_ref": float(data["U_ref"]),
                "E0":    float(data["E0"]),
                "pos":   np.array(data["pos"], dtype=np.float64),
                "vel":   np.array(data["vel"], dtype=np.float64),
                "spin":  np.array(data["spin"], dtype=np.float64),
            }
    except (OSError, KeyError, ValueError):
        return None


def restore_state(s, checkpoint):
    """Restaure pos, vel, t, s, U_ref depuis un checkpoint ; recalcule le reste."""
    s.t         = float(checkpoint["t"])
    s.s         = float(checkpoint.get("s", 0.0))
    s.s_comp    = 0.0
    s.pos[:, :] = checkpoint["pos"]
    s.vel[:, :] = checkpoint["vel"]
    if "spin" in checkpoint:
        s.spin[:, :] = checkpoint["spin"]
    accel, _    = compute_forces(s, G, SOFTENING)
    s.accel[:, :] = accel
    if "E0" in checkpoint:
        s.E0 = float(checkpoint["E0"])
    else:
        s.E0 = compute_hamiltonian(s, G, SOFTENING)
    if "U_ref" in checkpoint:
        s.U_ref = float(checkpoint["U_ref"])
    else:
        s.U_ref = abs(compute_potential_energy(s, G, SOFTENING))


# ══════════════════════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Arguments CLI ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Simulation N-corps 3D — Yoshida 4 · Kahan · SoA"
    )
    parser.add_argument("--realtime", action="store_true",
                        help="Visualisation temps réel matplotlib.")
    parser.add_argument("--web",      action="store_true",
                        help="Visualisation temps réel navigateur (Three.js).")
    parser.add_argument("--port",     type=int, default=5050,
                        help="Port du serveur web (défaut : 5050).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Désactive la barre de progression console (tqdm).")
    parser.add_argument("--open-dashboard", action="store_true",
                        help="Ouvre le dashboard Astralis (dashboard_orbite.py) à la fin du run.")
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis le dernier checkpoint si disponible.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Désactive la sauvegarde périodique de checkpoints.")
    args = parser.parse_args()

    n_threads = configure_numba_threads()

    # ── État initial ──────────────────────────────────────────────────────────
    s           = build_state()
    import config as _cfg_run
    for i, body in enumerate(_cfg_run.BODIES):
        pi = body.get("parent_index")
        if pi is not None and int(pi) >= 0 and int(pi) < i:
            p = int(pi)
            dx = s.pos[i, 0] - s.pos[p, 0]
            dy = s.pos[i, 1] - s.pos[p, 1]
            dz = s.pos[i, 2] - s.pos[p, 2]
            dist_km = (dx * dx + dy * dy + dz * dz) ** 0.5 / 1e3
            print(
                f"  Satellite « {body.get('name', i)} » → parent "
                f"« {_cfg_run.BODIES[p].get('name', p)} » : "
                f"distance initiale = {dist_km:,.1f} km"
            )
    temps_cible = DUREE_ANS * 365.25 * 86400
    frame       = 0
    collision   = None
    data_log    = []
    part_index  = 0

    columns = _build_output_columns(s.n)
    os.makedirs(CHEMIN_SAUVEGARDE, exist_ok=True)
    chemin_parquet = _build_parquet_path(OUTPUT_NAME)

    # CORRECTION BUG #5 : initialisation explicite de last_H et last_L
    # depuis l'état initial, pour que le CSV soit cohérent dès la frame 0.
    last_H = s.E0
    last_L = compute_angular_momentum(s)
    dH_samples = []

    # ── Reprise depuis checkpoint ─────────────────────────────────────────────
    ckpt_dir = _resolve_checkpoint_dir()
    resumed_from = None
    if args.resume:
        ckpt = load_checkpoint(ckpt_dir)
        if ckpt is not None:
            try:
                if ckpt["pos"].shape[0] != s.n:
                    raise ValueError(
                        f"Checkpoint incompatible : {ckpt['pos'].shape[0]} corps, "
                        f"config en attend {s.n}."
                    )
                restore_state(s, ckpt)
                frame = int(ckpt.get("frame", 0))
                meta = load_run_meta(ckpt_dir)
                if meta and os.path.exists(meta.get("chemin_parquet", "")):
                    chemin_parquet = meta["chemin_parquet"]
                    part_index = int(meta.get("part_index", 0))
                resumed_from = datetime.now().isoformat()
                print(f"\n  ↻ Reprise depuis checkpoint")
                print(f"    t = {s.t/86400:.2f} jours ({s.t/86400/365.25:.4f} ans)")
                print(f"    frame = {frame:,}")
                if meta:
                    print(f"    parquet (reprise) → {chemin_parquet}")
            except (KeyError, ValueError) as e:
                print(f"  ⚠ Checkpoint incompatible, démarrage à zéro : {e}")
        else:
            print("  ⚠ --resume demandé mais aucun checkpoint trouvé : démarrage à zéro.")

    # ── Viewer temps réel ─────────────────────────────────────────────────────
    viewer = None
    if args.realtime or args.web:
        try:
            from core.realtime_viewer import RealtimeViewer
            if args.realtime and args.web:
                mode = "both"
            elif args.web:
                mode = "web"
            else:
                mode = "matplotlib"
            viewer = RealtimeViewer(mode=mode, port=args.port)
            viewer.start(n_bodies=NBODIES, body_names=list(BODY_NAMES))
        except ImportError:
            print("  ⚠ realtime_viewer.py introuvable — visualisation désactivée.")
            viewer = None

    # ── En-tête console ───────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  ASTRALIS — MOTEUR N-CORPS 3D  |  YOSHIDA 4 · KAHAN · SoA")
    print(f"{'═'*54}")
    print(f"  Durée cible  : {DUREE_ANS} ans")
    print(f"  Intégrateur  : Yoshida 4 + Sundman (dt adaptatif {DT_MIN}…{DT_MAX} s)")
    print(f"  U_ref        : {s.U_ref:.6e} J")
    print(f"  Énergie init : {s.E0:.6e} J")
    print(f"  Corps        : {NBODIES} ({', '.join(BODY_NAMES)})")
    print(f"  Mode         : 3D (Z actif)")
    print(f"  J2 actif     : {any(BODY_J2R2)}")
    print(f"  PN1 actif    : {ENABLE_PN1}"
          + (f"  (c={C_LIGHT:.3e} m/s)" if ENABLE_PN1 else ""))
    print(f"  Threads Numba: {n_threads} / {os.cpu_count() or '?'} cœurs"
          + (" (auto)" if not NUM_THREADS else f" (manuel, config={NUM_THREADS})"))
    print(f"  Flush disque : toutes les {FLUSH_EVERY:,} lignes → {chemin_parquet}")
    if viewer:
        mode_str = {"matplotlib": "matplotlib", "web": "web",
                    "both": "matplotlib + web"}
        print(f"  Viewer       : {mode_str.get(mode, mode)} "
              f"(toutes les {REALTIME_EVERY} frames)")
    print(f"{'─'*54}")

    # ── Fichier de progression (lu par le dashboard en polling) ──────────────
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    progress_file = os.path.join(RUNTIME_DIR, "sim_progress.json")
    PROGRESS_EVERY = max(MONITOR_EVERY, 50)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    enable_checkpoint = (not args.no_checkpoint) and CHECKPOINT_EVERY_ANS > 0
    checkpoint_period_s = CHECKPOINT_EVERY_ANS * 365.25 * 86400
    last_checkpoint_t = s.t

    def _write_progress(t_current, t_total, frame_n, status="running"):
        try:
            tmp = progress_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "t": round(t_current, 2),
                    "t_total": round(t_total, 2),
                    "ratio": round(min(t_current / t_total, 1.0), 6) if t_total > 0 else 0.0,
                    "frame": frame_n,
                    "status": status,
                }, f)
            os.replace(tmp, progress_file)
        except OSError:
            pass

    _write_progress(0, temps_cible, 0, status="running")

    # ── Boucle principale ─────────────────────────────────────────────────────
    with tqdm(total=int(temps_cible), unit='s',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                         '[{elapsed}<{remaining}]',
              disable=args.no_progress) as barre:

        while s.t < temps_cible:
            dist_min, i_close, j_close = compute_min_dist(s)
            dt = compute_adaptive_dt_from_min_dist(
                s, s.accel, DT_MAX, DT_MIN, DIST_SEUIL,
                dist_min, i_close, j_close, ALPHA)

            _, dt = yoshida4_step(s, dt, G, softening=SOFTENING,
                                  compute_forces_fn=compute_forces,
                                  enable_pn1=ENABLE_PN1, c_light=C_LIGHT)
            frame += 1

            # ── Collision ─────────────────────────────────────────────────────
            col = check_collision(dist_min, i_close, j_close)
            if col:
                collision = (*col, s.t / 86400)
                tqdm.write(f"  ⚠ Collision objets {col[0]} & {col[1]} "
                           f"à {s.t/86400:.2f} jours")
                break

            # ── Monitoring énergie / moment ───────────────────────────────────
            if frame % MONITOR_EVERY == 0:
                last_H, last_L, dH = compute_diagnostics(s, G, SOFTENING)
                if not np.isnan(dH):
                    dH_samples.append(dH)

            # ── Sauvegarde CSV ────────────────────────────────────────────────
            # CORRECTION BUG #5 : si SAVE_EVERY < MONITOR_EVERY, on recalcule
            # last_H et last_L ici pour que le CSV reflète l'état courant.
            if frame % SAVE_EVERY == 0:
                if frame % MONITOR_EVERY != 0:
                    last_H, last_L, dH = compute_diagnostics(s, G, SOFTENING)
                    if not np.isnan(dH):
                        dH_samples.append(dH)

                row = [frame, round(s.t / 86400, 4)]
                for i in range(s.n):
                    row.extend([
                        round(s.pos[i, 0] / UA, 7),
                        round(s.pos[i, 1] / UA, 7),
                        round(s.pos[i, 2] / UA, 7),
                    ])
                for i in range(s.n):
                    row.extend([
                        round(s.vel[i, 0], 4),
                        round(s.vel[i, 1], 4),
                        round(s.vel[i, 2], 4),
                    ])
                row.extend([round(last_H, 4), round(last_L, 4), round(dt, 2)])
                data_log.append(row)

                if len(data_log) >= FLUSH_EVERY:
                    part_index = _flush_data_log(
                        data_log, columns, chemin_parquet, part_index
                    )
                    save_run_meta(ckpt_dir, chemin_parquet, part_index)

            # ── Envoi au viewer (non-bloquant) ────────────────────────────────
            if viewer and frame % REALTIME_EVERY == 0:
                viewer.send(s, last_H, last_L, dt)

            # ── Progression JSON ──────────────────────────────────────────────
            if frame % PROGRESS_EVERY == 0:
                _write_progress(s.t, temps_cible, frame)

            # ── Checkpoint périodique (atomique) ──────────────────────────────
            if enable_checkpoint and (s.t - last_checkpoint_t) >= checkpoint_period_s:
                try:
                    ckpt_path = save_checkpoint(s, frame, ckpt_dir)
                    save_run_meta(ckpt_dir, chemin_parquet, part_index)
                    last_checkpoint_t = s.t
                    tqdm.write(f"  💾 Checkpoint @ {s.t/86400/365.25:.2f} ans → {ckpt_path}")
                except OSError as e:
                    tqdm.write(f"  ⚠ Échec écriture checkpoint : {e}")

            barre.update(int(dt))

    _write_progress(s.t, temps_cible, frame, status="done")

    # Nettoyage du checkpoint en fin de run réussi (s.t >= temps_cible) :
    # le run a abouti, donc le checkpoint n'a plus de raison d'exister.
    # En cas de collision/crash, le checkpoint est conservé pour permettre
    # l'analyse post-mortem ou une reprise manuelle.
    if enable_checkpoint and collision is None and s.t >= temps_cible:
        ckpt_path = os.path.join(ckpt_dir, CHECKPOINT_FILENAME)
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            clear_run_meta(ckpt_dir)
        except OSError:
            pass

    # ── Arrêt propre du viewer ────────────────────────────────────────────────
    if viewer:
        viewer.stop()

    # ══════════════════════════════════════════════════════════════════════════
    #  SAUVEGARDE PARQUET (fusion des morceaux flushés + reliquat RAM)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  Sauvegarde...")

    df = _finalize_parquet(chemin_parquet, data_log, columns)
    data_log.clear()
    if df is None:
        print("  ⚠ Aucune donnée à sauvegarder.")
        chemin_parquet = ""

    # ── Sidecar de métriques exactes (accélère le dashboard) ──────────────────
    try:
        from core.metrics_sidecar import build_sidecar
        if df is not None:
            sidecar_path = build_sidecar(chemin_parquet, df=df)
            if sidecar_path:
                print(f"  Sidecar metrics → {sidecar_path}")
    except (ImportError, Exception) as e:
        print(f"  ⚠ Sidecar metrics non généré ({type(e).__name__}: {e})")

    H_final, L_final, dH_final = compute_diagnostics(s, G, SOFTENING)
    derive_E     = abs(dH_final) * 100 if not np.isnan(dH_final) else float('nan')
    drift_stats  = summarize_drift_series(dH_samples)
    derive_E_rms     = (drift_stats["rms"]     * 100
                        if not np.isnan(drift_stats["rms"])     else float('nan'))
    derive_E_max_mon = (drift_stats["max_abs"] * 100
                        if not np.isnan(drift_stats["max_abs"]) else float('nan'))

    print(f"{'═'*54}")
    print(f"  RÉSUMÉ")
    print(f"{'═'*54}")
    print(f"  Durée simulée       : {s.t/86400:.2f} jours "
          f"({s.t/86400/365.25:.2f} ans)")
    print(f"  Frames totales      : {frame:,}")
    print(f"  Lignes sauvegardées : {len(df):,}" if df is not None else "  Lignes sauvegardées : 0")
    print(f"  Dérive énergie      : {derive_E:.8f} %")
    print(f"  Dérive max (monit.) : {derive_E_max_mon:.8f} %")
    print(f"  Dérive RMS E        : {derive_E_rms:.8f} %")
    if collision:
        print(f"  ⚠ Collision         : objets {collision[0]} & "
              f"{collision[1]} à {collision[2]:.2f} jours")
    if chemin_parquet:
        print(f"  Fichier Parquet     : {chemin_parquet}")
        print(f"  Analyse locale      : python \"dashboard_orbite.py\" --file \"{chemin_parquet}\"")
        save_last_run(str(RUNTIME_DIR), chemin_parquet, status="done")
    print(f"{'═'*54}")

    if args.open_dashboard:
        try:
            subprocess.Popen(
                [sys.executable, "dashboard_orbite.py"],
                cwd=str(APP_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("  Dashboard lancé sur http://127.0.0.1:8050")
        except Exception as e:
            print(f"  ⚠ Impossible de lancer le dashboard automatiquement: {e}")


if __name__ == "__main__":
    main()
