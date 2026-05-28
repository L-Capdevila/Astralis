import json
import shutil
import sys
from pathlib import Path

import numpy as np

from core.body_init import resolve_body_positions, resolve_body_velocities
from core.celestial_velocities import (
    auto_orbit_velocity as _auto_orbit_velocity,
    manual_velocity_with_inclination as _manual_velocity_with_inclination,
    relative_circular_velocity as _relative_circular_velocity,
)

if getattr(sys, "frozen", False):
    # Install .exe : données modifiables à côté de Astralis.exe (pas dans _internal/)
    APP_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", str(APP_DIR)))
    PROJECT_ROOT = APP_DIR
else:
    APP_DIR = Path(__file__).resolve().parent
    BUNDLE_DIR = APP_DIR
    PROJECT_ROOT = APP_DIR.parent

SETTINGS_PATH = APP_DIR / "settings.json"
RUNTIME_DIR = PROJECT_ROOT / "outputs"


def _seed_settings_if_needed():
    """Copie settings.json embarqué vers le dossier d'installation (écriture autorisée)."""
    if not getattr(sys, "frozen", False) or SETTINGS_PATH.exists():
        return
    bundled = BUNDLE_DIR / "settings.json"
    try:
        if bundled.is_file():
            shutil.copy2(bundled, SETTINGS_PATH)
        else:
            with SETTINGS_PATH.open("w", encoding="utf-8") as f:
                json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

DEFAULT_SETTINGS = {
    "physics": {
        "G": 6.674e-11,
        "UA": 1.496e11,
    },
    "simulation": {
        "duree_ans": 200000.0,
        "save_every": 100,
        "monitor_every": 100,
        "realtime_every": 50,
        "output_dir": "outputs",
        "output_name": "",
        "dt_max": 7180.2,
        "dt_min": 9.677,
        "dist_seuil_m": 2.959e9,
        "softening": 4.5568e4,
        "alpha": 0.1,
        "rayon_collision": 1e3,
        "checkpoint_every_ans": 10.0,
        "checkpoint_dir": "checkpoints",
        "enable_pn1": False,
        "c_light": 299792458.0,
        "num_threads": 0,
        "flush_every": 50000,
        "web_port": 5050,
    },
    "bodies": [
        {
            "name": "Central",
            "mass": 1.989e30,
            "rayon": 6.957e8,
            "pos": [0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "use_auto_vel": False,
            "vel_manual": [0.0, 0.0, 0.0],
            "incl_deg": 0.0,
            "sens": 1,
            "j2r2": 0.0,
            "mdot": 0.0,
            "spin_rate": 0.0,
            "spin_axis": [0.0, 0.0, 1.0],
            "k2": 0.0,
            "inertia_factor": 0.4,
        },
        {
            "name": "Corps 1",
            "mass": 5.972e24,
            "rayon": 6.371e6,
            "pos": [1.0 * 1.496e11, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "use_auto_vel": True,
            "vel_manual": [0.0, 0.0, 0.0],
            "incl_deg": 23.4,
            "sens": 1,
            "j2r2": 0.0,
            "mdot": 0.0,
            "spin_rate": 0.0,
            "spin_axis": [0.0, 0.0, 1.0],
            "k2": 0.0,
            "inertia_factor": 0.4,
        },
        {
            "name": "Corps 2",
            "mass": 6.418e23,
            "rayon": 3.389e6,
            "pos": [1.52 * 1.496e11, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "use_auto_vel": True,
            "vel_manual": [0.0, 0.0, 0.0],
            "incl_deg": 25.2,
            "sens": 1,
            "j2r2": 0.0,
            "mdot": 0.0,
            "spin_rate": 0.0,
            "spin_axis": [0.0, 0.0, 1.0],
            "k2": 0.0,
            "inertia_factor": 0.4,
        },
    ],
}


def _deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_settings(settings):
    out = json.loads(json.dumps(settings))
    if "physics" not in out:
        out["physics"] = {}
    if "simulation" not in out:
        out["simulation"] = {}
    if "bodies" not in out or not isinstance(out["bodies"], list):
        out["bodies"] = list(DEFAULT_SETTINGS["bodies"])
    else:
        from core.body_positions import sync_absolute_positions
        sync_absolute_positions(out["bodies"])
    sim = out["simulation"]
    if "dist_seuil_m" not in sim and "dist_seuil_ua" in sim:
        ua = float(out.get("physics", {}).get("UA", DEFAULT_SETTINGS["physics"]["UA"]))
        sim["dist_seuil_m"] = float(sim["dist_seuil_ua"]) * ua
    return out


def load_settings():
    settings = _normalize_settings(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                user_settings = json.load(f)
            settings = _deep_merge(settings, _normalize_settings(user_settings))
        except (json.JSONDecodeError, OSError, TypeError):
            # Fallback robuste: si le fichier utilisateur est invalide/inaccessible,
            # on conserve la configuration par défaut normalisée.
            pass
    return settings


def save_settings(settings):
    normalized = _normalize_settings(settings)
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)


def _resolve_mdot(body: dict) -> float:
    """
    dM/dt signé pour la simulation (masses += mdot * dt).
    Négatif = perte, positif = gain.
    Ancien JSON : mdot positif sans mdot_loss = perte (corrige l'inversion UI).
    """
    raw = float(body.get("mdot", 0.0))
    if raw == 0.0:
        return 0.0
    if "mdot_loss" in body:
        mag = abs(raw)
        return -mag if body["mdot_loss"] else mag
    if raw < 0.0:
        return raw
    # Legacy : valeur positive saisie comme « perte » dans l'ancien dashboard
    return -raw


_seed_settings_if_needed()
SETTINGS = load_settings()
PHYSICS = SETTINGS["physics"]
SIM = SETTINGS["simulation"]
BODIES = SETTINGS["bodies"]

# Constantes physiques
G = float(PHYSICS["G"])
UA = float(PHYSICS["UA"])

# Contrôle simulation / export
DUREE_ANS = float(SIM["duree_ans"])
SAVE_EVERY = int(SIM["save_every"])
MONITOR_EVERY = int(SIM["monitor_every"])
REALTIME_EVERY = int(SIM["realtime_every"])
OUTPUT_DIR = str(SIM.get("output_dir", "outputs"))
if Path(OUTPUT_DIR).is_absolute():
    CHEMIN_SAUVEGARDE = OUTPUT_DIR
else:
    CHEMIN_SAUVEGARDE = str((PROJECT_ROOT / OUTPUT_DIR).resolve())


def _move_if_possible(src: Path, dst: Path):
    """Déplace src vers dst sans écraser si dst existe déjà."""
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(dst)
    except OSError:
        # Fallback robuste (ex: volume différent)
        try:
            import shutil
            shutil.move(str(src), str(dst))
        except OSError:
            pass


def _migrate_legacy_outputs():
    """
    Déplace les sorties historiques depuis la racine du projet
    vers le dossier de sortie configuré.
    """
    save_dir = Path(CHEMIN_SAUVEGARDE).resolve()
    app_dir = APP_DIR.resolve()
    root_dir = PROJECT_ROOT.resolve()
    if save_dir == app_dir:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    for pattern in ("simulation_*.parquet", "simulation_*.csv", "simulation_*.metrics.pkl",
                    "sim_progress.json", "last_run.json"):
        for src in list(app_dir.glob(pattern)) + list(root_dir.glob(pattern)):
            _move_if_possible(src, save_dir / src.name)

    # Migration des checkpoints historiques
    for legacy_ckpt in (app_dir / "checkpoints", root_dir / "checkpoints"):
        if not legacy_ckpt.exists() or not legacy_ckpt.is_dir():
            continue
        target_ckpt = save_dir / "checkpoints"
        target_ckpt.mkdir(parents=True, exist_ok=True)
        for src in legacy_ckpt.glob("*"):
            _move_if_possible(src, target_ckpt / src.name)
        try:
            legacy_ckpt.rmdir()
        except OSError:
            pass


_migrate_legacy_outputs()

# Précision numérique / stabilité
DT_MAX = float(SIM["dt_max"])
DT_MIN = float(SIM["dt_min"])
SOFTENING = float(SIM["softening"])
ALPHA = float(SIM.get("alpha", 0.1))
RAYON_COLLISION = float(SIM["rayon_collision"])
if "dist_seuil_m" in SIM:
    DIST_SEUIL = float(SIM["dist_seuil_m"])
    DIST_SEUIL_UA = DIST_SEUIL / UA if UA > 0 else 0.0
else:
    DIST_SEUIL_UA = float(SIM.get("dist_seuil_ua", 0.01978))
    DIST_SEUIL = DIST_SEUIL_UA * UA

# Checkpoint / reprise
CHECKPOINT_EVERY_ANS = float(SIM.get("checkpoint_every_ans", 10.0))
CHECKPOINT_DIR = str(SIM.get("checkpoint_dir", "checkpoints"))

# Corrections relativistes
ENABLE_PN1 = bool(SIM.get("enable_pn1", False))
C_LIGHT = float(SIM.get("c_light", 299792458.0))
NUM_THREADS = int(SIM.get("num_threads", 0))  # 0 = réglage automatique
FLUSH_EVERY = int(SIM.get("flush_every", 50000))
WEB_PORT = int(SIM.get("web_port", 5050))
OUTPUT_NAME = str(SIM.get("output_name", "")).strip()

NBODIES = len(BODIES)
BODY_NAMES = []
BODY_MASSES = []
BODY_POSITIONS = []
BODY_VELOCITIES = []
BODY_J2R2 = []
BODY_MDOT = []
BODY_SPIN = []    # vecteur moment angulaire propre S = I·ω·ŝ (kg·m²/s)
BODY_K2   = []    # nombre de Love de degré 2 (0 = corps rigide)
BODY_RADII = []   # rayon physique de chaque corps (m)

_RESOLVED_POSITIONS = resolve_body_positions(BODIES)

_specs = []
for i, body in enumerate(BODIES):
    name = str(body.get("name", f"Corps {i}"))
    mass = float(body.get("mass", 1.0))
    pos = [float(v) for v in _RESOLVED_POSITIONS[i]]
    use_auto = bool(body.get("use_auto_vel", False))
    vel_manual = [float(v) for v in body.get("vel_manual", body.get("vel", [0.0, 0.0, 0.0]))]
    incl_deg = float(body.get("incl_deg", 0.0))
    sens = int(body.get("sens", 1))
    j2r2 = float(body.get("j2r2", 0.0))
    mdot = _resolve_mdot(body)
    k2 = float(body.get("k2", 0.0))
    rayon = float(body.get("rayon", 1e3))
    pi = body.get("parent_index")
    parent_index = int(pi) if pi is not None and int(pi) >= 0 else None

    spin_rate = float(body.get("spin_rate", 0.0))
    spin_axis_raw = body.get("spin_axis", [0.0, 0.0, 1.0])
    spin_axis = [float(v) for v in spin_axis_raw]
    ax_norm = (spin_axis[0] ** 2 + spin_axis[1] ** 2 + spin_axis[2] ** 2) ** 0.5
    if ax_norm > 1e-30:
        spin_axis = [v / ax_norm for v in spin_axis]
    else:
        spin_axis = [0.0, 0.0, 1.0]

    inertia_factor = float(body.get("inertia_factor", 0.4))
    if j2r2 > 0.0 and spin_rate != 0.0:
        R_approx = (j2r2 / 1e-3) ** 0.5
        I = inertia_factor * mass * R_approx ** 2
        S = I * spin_rate
    else:
        S = 0.0
    spin_vec = [S * spin_axis[0], S * spin_axis[1], S * spin_axis[2]]

    _specs.append({
        "name": name,
        "mass": mass,
        "pos": pos,
        "use_auto": use_auto if i > 0 else False,
        "vel_manual": vel_manual,
        "incl_deg": incl_deg,
        "sens": sens,
        "j2r2": j2r2,
        "mdot": mdot,
        "k2": k2,
        "rayon": rayon,
        "spin_vec": spin_vec,
        "parent_index": parent_index,
        "vel0": [float(v) for v in body.get("vel", [0.0, 0.0, 0.0])] if i == 0 else None,
    })

_velocities = resolve_body_velocities(BODIES, _RESOLVED_POSITIONS, G)

for spec, vel in zip(_specs, _velocities):
    BODY_NAMES.append(spec["name"])
    BODY_MASSES.append(spec["mass"])
    BODY_POSITIONS.append(spec["pos"])
    BODY_VELOCITIES.append([float(v) for v in vel])
    BODY_J2R2.append(spec["j2r2"])
    BODY_MDOT.append(spec["mdot"])
    BODY_SPIN.append(spec["spin_vec"])
    BODY_K2.append(spec["k2"])
    BODY_RADII.append(spec["rayon"])

# Vitesses déjà corrigées barycentriques dans resolve_body_velocities().
