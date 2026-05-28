""" dashboard_orbite.py — Astralis · Dashboard d'analyse et de contrôle de simulation N-corps
==============================================================================
Lancement :
    python dashboard_orbite.py
    python dashboard_orbite.py --file chemin/vers/simulation.parquet

Onglets :
  1. Orbites 2D/3D  — trajectoires des corps
  2. Distances       — distances inter-corps au cours du temps
  3. Énergie         — dérive d'énergie et moment cinétique
  4. Simulation      — paramétrage et lancement de moteur_astralis.py

Dépendances : PyQt5, matplotlib, numpy, pandas, pyarrow
""" 
import sys
import os
import json
import glob
import re
import subprocess
import threading
import argparse
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit,
    QGroupBox, QSplitter, QTextEdit, QProgressBar,
    QSlider, QScrollArea, QMessageBox, QSizePolicy,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────
#  Python portable (WinPython sur clé USB)
# ─────────────────────────────────────────────────────────

# Fonctionne en script .py ET compile en .exe (PyInstaller)
if getattr(sys, 'frozen', False):
    # Mode .exe : tout est a cote de Astralis.exe
    APP_DIR      = os.path.dirname(sys.executable)
    PROJECT_ROOT = APP_DIR
    _BUNDLE_DIR  = sys._MEIPASS   # ressources embarquees (logo, etc.)
else:
    # Mode script normal
    APP_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR)
    _BUNDLE_DIR  = APP_DIR

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def _default_moteur_script() -> str:
    """Chemin par défaut du moteur (routé par Astralis.exe en mode installé)."""
    return os.path.join(APP_DIR, "moteur_astralis.py")


def _moteur_script_ok(script: str) -> bool:
    """True si le script moteur est utilisable (fichier ou routage .exe)."""
    script = (script or "").strip()
    if not script:
        return False
    if getattr(sys, "frozen", False):
        return os.path.basename(script).lower() == "moteur_astralis.py"
    return os.path.isfile(script)


def _configs_dir():
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    return CONFIGS_DIR


def _outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    return OUTPUTS_DIR


def _sanitize_output_basename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:120]


def _preview_parquet_filename(output_name: str, output_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = _sanitize_output_basename(output_name)
    fname = f"{label}_{ts}.parquet" if label else f"simulation_{ts}.parquet"
    base = output_dir if os.path.isabs(output_dir) else os.path.join(PROJECT_ROOT, output_dir)
    return os.path.join(base, fname)


def _search_roots(base_dir: str):
    """Dossiers où chercher une installation portable (app, racine projet, parent clé USB)."""
    roots = []
    for candidate in (
        base_dir,
        PROJECT_ROOT,
        os.path.dirname(PROJECT_ROOT),
        APP_DIR,
    ):
        candidate = os.path.abspath(candidate)
        if candidate and candidate not in roots:
            roots.append(candidate)
    return roots


def _is_portable_dir_name(name: str) -> bool:
    """True si le nom de dossier ressemble à WinPython / WPy portable."""
    lower = name.lower()
    return (
        lower == "winpython"
        or lower.startswith("wpy")
        or lower.startswith("winpython")
    )


def _iter_portable_dirs(base_dir: str):
    """Parcourt WinPython/, WPy64-..., etc. (insensible à la casse)."""
    seen = set()
    for root in _search_roots(base_dir):
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            path = os.path.abspath(os.path.join(root, name))
            if path in seen or not os.path.isdir(path):
                continue
            if _is_portable_dir_name(name):
                seen.add(path)
                yield path


def _collect_python_candidates(portable_root: str):
    """Liste les python.exe sous une installation portable."""
    primary, fallback = [], []
    for root, _dirs, files in os.walk(portable_root):
        if "python.exe" not in files:
            continue
        exe = os.path.join(root, "python.exe")
        norm = root.replace("\\", "/").lower()
        if norm.endswith("/scripts") or "/scripts/" in norm:
            fallback.append(exe)
        else:
            primary.append(exe)
    return primary or fallback


def resolve_python_executable(base_dir: str) -> str:
    """
    Si une installation WinPython/WPy est présente près du projet (clé USB),
    retourne le chemin absolu vers python.exe ; sinon sys.executable.
    """
    candidates = []
    for portable_root in _iter_portable_dirs(base_dir):
        candidates.extend(_collect_python_candidates(portable_root))

    if candidates:
        candidates.sort(
            key=lambda p: (
                "python-" not in os.path.basename(os.path.dirname(p)).lower(),
                "scripts" in p.replace("\\", "/").lower(),
                p,
            )
        )
        return os.path.abspath(candidates[0])
    return sys.executable


def _exe_looks_portable(python_exe: str) -> bool:
    """Heuristique : le chemin ressemble à une install WinPython/WPy."""
    lower = os.path.abspath(python_exe).replace("\\", "/").lower()
    markers = ("/winpython/", "/wpy", "\\winpython\\", "\\wpy")
    return any(m.replace("/", "\\") in lower or m in lower for m in markers)


def is_portable_python_launch(base_dir: str, python_exe: str) -> bool:
    """True si python_exe provient d'une installation portable détectée."""
    exe_abs = os.path.abspath(python_exe)
    for portable_root in _iter_portable_dirs(base_dir):
        root_abs = os.path.abspath(portable_root)
        try:
            if os.path.commonpath([exe_abs, root_abs]) == root_abs:
                return True
        except ValueError:
            continue
    return _exe_looks_portable(python_exe)


def describe_python_environment(base_dir: str) -> dict:
    """Résumé Python + CPU pour l'onglet Simulation."""
    base_dir = base_dir or APP_DIR
    exe = resolve_python_executable(base_dir)
    portable = is_portable_python_launch(base_dir, exe)
    cores = os.cpu_count() or 1
    portable_dirs = list(_iter_portable_dirs(base_dir))
    return {
        "exe": exe,
        "portable": portable,
        "label": "Clé USB (WinPython)" if portable else "PC local",
        "cores": cores,
        "auto_threads": default_num_threads(cores),
        "portable_dirs": portable_dirs,
        "searched_roots": _search_roots(base_dir),
    }


def default_num_threads(cores: int = None) -> int:
    """Nombre de threads Numba recommandé (règle auto du simulateur)."""
    if cores is None:
        cores = os.cpu_count() or 1
    if cores <= 2:
        return 1
    if cores <= 4:
        return cores - 1
    return cores - 2


def load_last_run_parquet(project_dir: str = ""):
    """Chemin du dernier .parquet produit (outputs/last_run.json)."""
    path = os.path.join(OUTPUTS_DIR, "last_run.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("parquet", "")
        return p if p and os.path.isfile(p) else None
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────
UA = 1.496e11
G  = 6.674e-11

# Plage des champs position (mètres)
_POS_SPIN_MIN = -1e14
_POS_SPIN_MAX = 1e14


def _fmt_distance_m(m: float) -> str:
    """Affichage lisible d'une distance en mètres."""
    a = abs(float(m))
    if a >= 1e9:
        return f"{m / 1e9:.4f}×10⁹ m"
    if a >= 1e6:
        return f"{m / 1e6:.4f}×10⁶ m"
    if a >= 1e3:
        return f"{m / 1e3:.2f} km"
    return f"{m:.2f} m"

PALETTE = [
    "#FFD700",  # or — corps central / étoile
    "#4FC3F7",  # bleu ciel
    "#EF5350",  # rouge
    "#66BB6A",  # vert
    "#AB47BC",  # violet
    "#FF7043",  # orange
    "#26C6DA",  # cyan
    "#EC407A",  # rose
]

# ── Palette Control Room SpaceX / NASA ───────────────────────────────────────
DARK_BG   = "#0B0F19"   # le vide — fond principal
PANEL_BG  = "#111827"   # panneaux / onglets
CARD_BG   = "#1A233A"   # cartes / inputs interactifs
ACCENT    = "#00E5FF"   # néon cyan — éléments actifs / primaire
ACCENT2   = "#38BDF8"   # bleu ciel — survols
BORDER    = "#2A3B5C"   # bordures fines discrètes
TEXT_COL  = "#F8FAFC"   # texte principal
MUTED     = "#94A3B8"   # texte secondaire
HIGHLIGHT = ACCENT2     # alias hover
FIG_VOID  = DARK_BG     # graphiques sur fond vide
FIG_PANEL = PANEL_BG    # graphiques dans un panneau

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_COL};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}
/* ── Onglets ── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {PANEL_BG};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {CARD_BG};
    color: {MUTED};
    padding: 10px 18px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.5px;
    min-width: 90px;
    border: 1px solid transparent;
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {TEXT_COL};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background: {PANEL_BG};
    color: {ACCENT2};
    border-color: {BORDER};
}}
/* ── Boutons ── */
QPushButton {{
    background-color: {CARD_BG};
    color: {ACCENT2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 16px;
    font-weight: 500;
    letter-spacing: 0.5px;
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {BORDER};
    color: {ACCENT2};
}}
QPushButton:pressed {{
    background-color: {PANEL_BG};
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton#primary {{
    background-color: {DARK_BG};
    border: 1px solid {ACCENT};
    color: {ACCENT};
    font-size: 14px;
    padding: 10px 24px;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {PANEL_BG};
    border-color: {ACCENT2};
    color: {ACCENT2};
}}
QPushButton#danger {{
    background-color: #140810;
    border: 1px solid #5c2030;
    color: #f87171;
}}
QPushButton#danger:hover {{
    background-color: #1f0a14;
    border-color: #ef4444;
    color: #fca5a5;
}}
/* ── GroupBox ── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    top: -8px;
    color: {ACCENT2};
    letter-spacing: 1px;
}}
/* ── Labels ── */
QLabel {{
    color: {TEXT_COL};
}}
QLabel#muted {{
    color: {MUTED};
    font-size: 11px;
}}
/* ── Inputs ── */
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
    background: {CARD_BG};
    color: {TEXT_COL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 8px;
    selection-background-color: {BORDER};
    selection-color: {ACCENT};
}}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border-color: {ACCENT};
    background-color: {PANEL_BG};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {CARD_BG};
    color: {TEXT_COL};
    border: 1px solid {BORDER};
    selection-background-color: {BORDER};
    selection-color: {ACCENT};
}}
/* ── Console ── */
QTextEdit {{
    background: {DARK_BG};
    color: {ACCENT2};
    border: 1px solid {BORDER};
    border-radius: 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 6px;
    border-radius: 3px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT2};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {DARK_BG};
    height: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 3px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT2};
}}
/* ── ProgressBar ── */
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {CARD_BG};
    text-align: center;
    color: {ACCENT};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT2}, stop:1 {ACCENT});
    border-radius: 2px;
}}
/* ── Slider ── */
QSlider::groove:horizontal {{
    background: {CARD_BG};
    height: 4px;
    border-radius: 2px;
    border: 1px solid {BORDER};
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
    border: 1px solid {ACCENT2};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT2};
    width: 14px;
    height: 14px;
    margin: -5px 0;
}}
/* ── Table ── */
QTableWidget {{
    background: {CARD_BG};
    gridline-color: {BORDER};
    color: {TEXT_COL};
    border: none;
}}
QTableWidget::item:selected {{
    background: {BORDER};
    color: {ACCENT};
}}
QHeaderView::section {{
    background: {PANEL_BG};
    color: {ACCENT2};
    padding: 5px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    letter-spacing: 0.5px;
    font-size: 11px;
}}
/* ── Checkbox ── */
QCheckBox {{
    spacing: 6px;
    color: {TEXT_COL};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 2px;
    background: {CARD_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT2};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT2};
}}
/* ── Séparateurs ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {BORDER};
}}
/* ── StatusBar ── */
QStatusBar {{
    background: {PANEL_BG};
    color: {MUTED};
    border-top: 1px solid {BORDER};
    font-size: 11px;
}}
"""
# ─────────────────────────────────────────────────────────
#  Utilitaires matplotlib (thème sombre)
# ─────────────────────────────────────────────────────────
def apply_dark_axes(ax, bg=FIG_PANEL, fg=TEXT_COL, grid_alpha=0.4):
    ax.set_facecolor(bg)
    ax.figure.patch.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=fg, labelsize=9)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    if hasattr(ax, 'zaxis'):
        ax.zaxis.label.set_color(fg)
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.tick_params(axis='z', colors=fg)
    ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5, color=BORDER)


def make_figure(nrows=1, ncols=1, **kwargs):
    fig = Figure(facecolor=FIG_PANEL, **kwargs)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.12)
    return fig


# ─────────────────────────────────────────────────────────
#  Chargement données
# ─────────────────────────────────────────────────────────
class SimData:
    """Conteneur pour les données d'une simulation."""
    def __init__(self):
        self.df: pd.DataFrame = None
        self.metrics: dict = {}
        self.parquet_path: str = ""
        self.n_bodies: int = 0
        self.body_names: list = []
        self.t_jours: np.ndarray = None

    MAX_ROWS = 50_000  # Limite affichage — les métriques exactes viennent du pkl

    @classmethod
    def load(cls, parquet_path: str) -> "SimData":
        sd = cls()
        sd.parquet_path = parquet_path

        # Lire d'abord uniquement les métadonnées (nbre de lignes) sans charger les données
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        n_total = pf.metadata.num_rows

        if n_total > cls.MAX_ROWS:
            # Lecture par batch : ne charge qu'un batch à la fois en mémoire
            # puis ne garde qu'une ligne sur `step` — pic mémoire = 1 batch
            step = max(1, n_total // cls.MAX_ROWS)
            BATCH = 20_000
            chunks = []
            cursor = 0
            for batch in pf.iter_batches(batch_size=BATCH):
                n_b = len(batch)
                local_idx = [i for i in range(n_b) if (cursor + i) % step == 0]
                if local_idx:
                    chunks.append(batch.to_pandas().iloc[local_idx])
                cursor += n_b
            sd.df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            sd.df = pd.read_parquet(parquet_path)

        sd.t_jours = sd.df["Temps (jours)"].values

        # Détecter le nombre de corps
        x_cols = [c for c in sd.df.columns if c.startswith("X") and c[1:].isdigit()]
        sd.n_bodies = len(x_cols)

        # Charger le sidecar métriques s'il existe (JSON sécurisé)
        from core.metrics_sidecar import load_sidecar, is_sidecar_fresh
        if is_sidecar_fresh(parquet_path):
            raw = load_sidecar(parquet_path)
            if raw:
                sd.metrics = raw.get("summary", raw)

        # Calculer les métriques de base depuis le DataFrame si absentes
        sd._fill_metrics_from_df()

        # Noms par défaut
        sd.body_names = sd.metrics.get(
            "body_names",
            [f"Corps {i}" for i in range(sd.n_bodies)]
        )
        if len(sd.body_names) < sd.n_bodies:
            sd.body_names += [f"Corps {i}" for i in range(len(sd.body_names), sd.n_bodies)]

        return sd

    def _fill_metrics_from_df(self):
        """Calcule les métriques manquantes directement depuis le DataFrame."""
        m = self.metrics
        df = self.df

        # Dérive énergie
        if "derive_e" not in m and "E_totale" in df.columns:
            E = df["E_totale"].values
            E0 = E[0]
            if E0 != 0:
                derive_e = abs((E[-1] - E0) / E0 * 100.0)
                duree = self.duree_ans()
                m["derive_e"] = derive_e
                m["derive_e_mean"] = derive_e / duree if duree > 0 else float("nan")

        # Dérive moment cinétique
        if "derive_l" not in m and "L_total" in df.columns:
            L = df["L_total"].values
            L0 = L[0]
            if L0 != 0:
                m["derive_l"] = abs((L[-1] - L0) / L0 * 100.0)

    def positions(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne (X, Y, Z) en UA pour le corps i.
        Le parquet stocke les positions en UA directement."""
        x = self.df[f"X{i}"].values.astype(float)
        y = self.df[f"Y{i}"].values.astype(float)
        z = self.df[f"Z{i}"].values.astype(float)
        return x, y, z

    def body_parents_map(self) -> dict[int, int]:
        """Index du parent par corps (0 = référentiel absolu / centre)."""
        out = {i: 0 for i in range(self.n_bodies)}
        raw = self.metrics.get("body_parents", {})
        for k, v in raw.items():
            try:
                out[int(k)] = int(v)
            except (TypeError, ValueError):
                pass
        if not any(out.get(i, 0) not in (0, i) for i in range(self.n_bodies)):
            try:
                import config as _cfg
                for i, body in enumerate(_cfg.BODIES):
                    if i >= self.n_bodies:
                        break
                    pi = body.get("parent_index")
                    if pi is not None and int(pi) >= 0:
                        out[i] = int(pi)
            except Exception:
                pass
        return out

    def reference_parent_index(self):
        """Corps de référence pour le repère parent (ex. Terre pour la Lune)."""
        for i in range(self.n_bodies):
            p = self.parent_index(i)
            if p is not None:
                return p
        return None

    def parent_index(self, i: int):
        """Parent orbital si satellite, sinon None."""
        p = self.body_parents_map().get(i, 0)
        if 0 < p < self.n_bodies and p != i:
            return p
        return None

    def is_satellite(self, i: int) -> bool:
        return self.parent_index(i) is not None

    def has_satellites(self) -> bool:
        return any(self.is_satellite(i) for i in range(self.n_bodies))

    def positions_display(
        self, i: int, frame: str = "absolute", ref_parent: int = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trajectoire en UA.
        Absolu : repère simulation (Terre orbite le Soleil, etc.).
        Parent : uniquement satellites (rel. parent) et parent fixé à l'origine.
        """
        x, y, z = self.positions(i)
        if frame != "parent":
            return x, y, z
        if ref_parent is None:
            ref_parent = self.reference_parent_index()
        if ref_parent is not None and i == ref_parent:
            return np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        p = self.parent_index(i)
        if p is not None:
            xp, yp, zp = self.positions(p)
            return x - xp, y - yp, z - zp
        return x, y, z

    def display_scale(self, unit: str) -> float:
        """Facteur multiplicatif pour convertir des UA en l'unité choisie."""
        return {"UA": 1.0, "km": UA / 1e3, "m": UA}.get(unit, 1.0)

    def distance_pair(self, i: int, j: int) -> np.ndarray:
        xi, yi, zi = self.positions(i)
        xj, yj, zj = self.positions(j)
        return np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)

    def energy_drift(self) -> np.ndarray:
        E = self.df["E_totale"].values
        E0 = E[0]
        if E0 == 0:
            return np.zeros_like(E)
        return (E - E0) / abs(E0) * 100.0

    def angular_momentum(self) -> np.ndarray:
        return self.df["L_total"].values

    def duree_ans(self) -> float:
        return self.t_jours[-1] / 365.25 if self.t_jours is not None and len(self.t_jours) else 0


# ─────────────────────────────────────────────────────────
#  Worker thread simulation
# ─────────────────────────────────────────────────────────
class SimulationWorker(QThread):
    log_line  = pyqtSignal(str)
    finished  = pyqtSignal(bool, str)  # success, output_file

    def __init__(self, script_path, args_list, cwd, python_exe=None):
        super().__init__()
        self.script_path = script_path
        self.args_list   = args_list
        self.cwd         = cwd
        self.python_exe  = python_exe or sys.executable
        self._proc       = None
        self._stop       = False

    def run(self):
        cmd = [self.python_exe, self.script_path] + self.args_list
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            for line in self._proc.stdout:
                self.log_line.emit(line.rstrip())
                if self._stop:
                    self._proc.terminate()
                    break
            self._proc.wait()
            ok = (self._proc.returncode == 0)
            self.finished.emit(ok, "")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()


# ─────────────────────────────────────────────────────────
#  Onglet 1 — Orbites (4 vues simultanées)
# ─────────────────────────────────────────────────────────
class OrbiteTab(QWidget):

    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Barre d'outils
        toolbar = QHBoxLayout()

        self.cb_layout = QComboBox()
        self.cb_layout.addItems([
            "4 vues (XY / XZ / YZ / 3D)",
            "2 vues (XY / 3D)",
            "Vue unique XY",
            "Vue unique XZ",
            "Vue unique YZ",
            "Vue unique 3D",
        ])
        self.cb_layout.currentIndexChanged.connect(self.refresh)

        self.cb_subsample = QComboBox()
        self.cb_subsample.addItems(["Tous les points", "× 2", "× 5", "× 10", "× 50"])
        self.cb_subsample.currentIndexChanged.connect(self.refresh)

        self.cb_trail_mode = QComboBox()
        self.cb_trail_mode.addItems([
            "Ligne complète",
            "Fenêtre glissante",
            "Gradient temporel",
            "Densité (nuage)",
            "Position seule",
        ])
        self.cb_trail_mode.setCurrentIndex(0)
        self.cb_trail_mode.setToolTip(
            "Ligne complète    — trace toute la trajectoire (illisible >500 ans)\n"
            "Fenêtre glissante — affiche seulement les N dernières années autour du curseur\n"
            "Gradient temporel — opacité croissante du passé vers le présent\n"
            "Densité (nuage)   — tous les points en scatter très transparent\n"
            "Position seule    — seulement le point courant, aucune trajectoire"
        )
        self.cb_trail_mode.currentIndexChanged.connect(self._on_trail_mode_changed)
        self.cb_trail_mode.currentIndexChanged.connect(self.refresh)

        self.sp_window = QDoubleSpinBox()
        self.sp_window.setRange(1.0, 100000.0)
        self.sp_window.setValue(200.0)
        self.sp_window.setSuffix(" ans")
        self.sp_window.setDecimals(0)
        self.sp_window.setSingleStep(50)
        self.sp_window.setToolTip("Durée de la fenêtre glissante ou du gradient (en années simulées)")
        self.sp_window.valueChanged.connect(self.refresh)
        self.lbl_window = QLabel("Fenêtre :")

        self.chk_center = QCheckBox("Centrer sur corps 0")
        self.chk_center.setChecked(False)
        self.chk_center.setToolTip(
            "Décale la vue pour garder le corps 0 (Soleil) au centre de l'écran.\n"
            "En mode « Orbite satellite », le parent est déjà à l'origine."
        )
        self.chk_center.stateChanged.connect(self.refresh)

        self.cb_frame = QComboBox()
        self.cb_frame.addItems([
            "Système solaire (absolu)",
            "Orbite satellite (parent fixe)",
        ])
        self.cb_frame.setToolTip(
            "Système solaire : Soleil au centre, Terre en orbite, Lune en orbite "
            "autour de la Terre (simulation réelle).\n"
            "Orbite satellite : zoom sur l'orbite locale — parent fixe à l'origine, "
            "satellite(s) autour. Le Soleil n'est pas affiché (vue détaillée)."
        )
        self.cb_frame.currentIndexChanged.connect(self.refresh)

        self.cb_unit = QComboBox()
        self.cb_unit.addItems(["UA", "km", "m"])
        self.cb_unit.setToolTip("Unité des axes pour les graphiques d'orbite.")
        self.cb_unit.currentIndexChanged.connect(self.refresh)

        # Slider temps
        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(1000)
        self.slider_t.setValue(1000)
        self.slider_t.valueChanged.connect(self.refresh)
        self.lbl_t = QLabel("t = --- jours")
        self.lbl_t.setMinimumWidth(150)

        toolbar.addWidget(QLabel("Disposition :"))
        toolbar.addWidget(self.cb_layout)
        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Sous-ech. :"))
        toolbar.addWidget(self.cb_subsample)
        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Tracé :"))
        toolbar.addWidget(self.cb_trail_mode)
        toolbar.addSpacing(6)
        toolbar.addWidget(self.lbl_window)
        toolbar.addWidget(self.sp_window)
        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Repère :"))
        toolbar.addWidget(self.cb_frame)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Unité :"))
        toolbar.addWidget(self.cb_unit)
        toolbar.addSpacing(16)
        toolbar.addWidget(self.chk_center)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Jusqu'à :"))
        toolbar.addWidget(self.slider_t)
        toolbar.addWidget(self.lbl_t)

        layout.addLayout(toolbar)

        # ── Bandeau de sélection des corps ───────────────────────────────────
        self._body_bar = QWidget()
        self._body_bar.setStyleSheet(
            f"background:{CARD_BG}; border-radius:4px; padding:2px 6px;"
        )
        self._body_bar_layout = QHBoxLayout(self._body_bar)
        self._body_bar_layout.setContentsMargins(6, 3, 6, 3)
        self._body_bar_layout.setSpacing(14)

        lbl_corps = QLabel("Afficher :")
        lbl_corps.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:bold;")
        self._body_bar_layout.addWidget(lbl_corps)

        btn_all  = QPushButton("Tout")
        btn_none = QPushButton("Aucun")
        for btn in (btn_all, btn_none):
            btn.setFixedHeight(20)
            btn.setStyleSheet("font-size:10px; padding:1px 6px;")
        btn_all .clicked.connect(self._select_all_bodies)
        btn_none.clicked.connect(self._select_no_bodies)
        self._body_bar_layout.addWidget(btn_all)
        self._body_bar_layout.addWidget(btn_none)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{BORDER};")
        self._body_bar_layout.addWidget(sep)

        self._body_bar_layout.addStretch()
        self._body_checks: list = []

        layout.addWidget(self._body_bar)

        # Canvas
        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

        # Stats
        self.stats_label = QLabel("")
        self.stats_label.setObjectName("muted")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)

    def _on_trail_mode_changed(self, idx):
        """Affiche le spinner fenêtre seulement pour les modes qui en ont besoin."""
        needs_window = idx in (1, 2)
        self.lbl_window.setVisible(needs_window)
        self.sp_window .setVisible(needs_window)

    def _rebuild_body_checks(self):
        """Recrée une case à cocher par corps selon les données chargées."""
        for chk in self._body_checks:
            self._body_bar_layout.removeWidget(chk)
            chk.deleteLater()
        self._body_checks.clear()
        if self.data is None:
            return
        stretch_idx = self._body_bar_layout.count() - 1
        for i in range(self.data.n_bodies):
            name  = (self.data.body_names[i]
                     if i < len(self.data.body_names) else f"Corps {i}")
            color = PALETTE[i % len(PALETTE)]
            chk   = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-size: 11px; font-weight: bold; }}"
                f"QCheckBox::indicator {{ width: 13px; height: 13px; }}"
            )
            chk.stateChanged.connect(self.refresh)
            self._body_checks.insert(i, chk)
            self._body_bar_layout.insertWidget(stretch_idx + i, chk)

    def _select_all_bodies(self):
        for chk in self._body_checks:
            chk.setChecked(True)

    def _select_no_bodies(self):
        for chk in self._body_checks:
            chk.setChecked(False)

    def _visible_bodies(self) -> set:
        if not self._body_checks:
            return set(range(self.data.n_bodies if self.data else 0))
        return {i for i, chk in enumerate(self._body_checks) if chk.isChecked()}

    def load_data(self, data: SimData):
        self.data = data
        n = len(data.t_jours)
        self.slider_t.setMaximum(max(1, n - 1))
        self.slider_t.setValue(n - 1)
        self._rebuild_body_checks()
        # Suggérer automatiquement un mode lisible selon la durée
        duree_ans = data.t_jours[-1] / 365.25 if n > 0 else 0
        if duree_ans > 500:
            self.cb_trail_mode.setCurrentIndex(1)   # Fenêtre glissante
            self.sp_window.setValue(min(200.0, duree_ans / 10))
        else:
            self.cb_trail_mode.setCurrentIndex(0)   # Ligne complète
        self.cb_frame.setCurrentIndex(0)
        self.cb_unit.setCurrentText("UA")
        self.refresh()

    def _orbit_frame_mode(self) -> str:
        return "parent" if self.cb_frame.currentIndex() == 1 else "absolute"

    def _show_body_in_frame(self, d: SimData, i: int, frame: str) -> bool:
        """En mode satellite, n'afficher que le parent (fixe) et ses satellites."""
        if frame != "parent":
            return True
        if d.is_satellite(i):
            return True
        ref = d.reference_parent_index()
        return ref is not None and i == ref

    def _orbit_unit_scale(self, d: SimData) -> tuple[str, float]:
        unit = self.cb_unit.currentText()
        return unit, d.display_scale(unit)

    def _body_legend_name(self, d: SimData, i: int, frame: str) -> str:
        name = d.body_names[i] if i < len(d.body_names) else f"Corps {i}"
        if frame == "parent" and d.is_satellite(i):
            p = d.parent_index(i)
            pname = d.body_names[p] if p is not None and p < len(d.body_names) else f"Corps {p}"
            return f"{name} (orbite {pname})"
        if frame == "parent" and i == d.reference_parent_index():
            return f"{name} (fixe)"
        return name

    def _draw_2d(self, ax, d, idx_slice, idx_max, cx0, cy0, cz0, plane, trail_mode,
                 window_pts, frame, scale, unit, ref_parent=None):
        """Dessine une vue 2D — 5 modes de tracé."""
        ax.set_aspect("equal", adjustable="datalim")
        plane_map = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
        ia, ib = plane_map[plane]
        axis_names = ["X", "Y", "Z"]
        xl = f"{axis_names[ia]} ({unit})"
        yl = f"{axis_names[ib]} ({unit})"
        offsets = [cx0 * scale, cy0 * scale, cz0 * scale]
        ax.set_ylabel(yl, fontsize=8)
        title_suffix = " — orbite locale" if frame == "parent" else ""
        ref_name = ""
        if frame == "parent" and ref_parent is not None:
            pname = d.body_names[ref_parent] if ref_parent < len(d.body_names) else f"Corps {ref_parent}"
            ref_name = f" ({pname} fixe, satellite en orbite)"
        ax.set_title(f"Plan {plane}{title_suffix}{ref_name}", color=TEXT_COL, fontsize=9, pad=4)
        ax.set_xlabel(xl, fontsize=8)

        visible = self._visible_bodies() if hasattr(self, "_body_checks") and self._body_checks else set(range(d.n_bodies))
        for i in range(d.n_bodies):
            if i not in visible:
                continue
            if not self._show_body_in_frame(d, i, frame):
                continue
            x, y, z = d.positions_display(i, frame, ref_parent)
            x, y, z = x * scale, y * scale, z * scale
            coords   = [x, y, z]
            color    = PALETTE[i % len(PALETTE)]
            name     = self._body_legend_name(d, i, frame)
            ha_full  = coords[ia][:idx_max + 1] - offsets[ia]
            hb_full  = coords[ib][:idx_max + 1] - offsets[ib]
            ha_c     = coords[ia][idx_max] - offsets[ia]
            hb_c     = coords[ib][idx_max] - offsets[ib]

            if trail_mode == 0:          # Ligne complète
                ha = ha_full[idx_slice]; hb = hb_full[idx_slice]
                ax.plot(ha, hb, color=color, lw=0.7, alpha=0.55)

            elif trail_mode == 1:        # Fenêtre glissante
                i0 = max(0, idx_max - window_pts)
                ax.plot(ha_full[i0:], hb_full[i0:], color=color, lw=0.9, alpha=0.80)

            elif trail_mode == 2:        # Gradient temporel
                i0  = max(0, idx_max - window_pts)
                haw = ha_full[i0:]; hbw = hb_full[i0:]
                n_seg = len(haw) - 1
                if n_seg > 1:
                    import matplotlib.colors as mc
                    base    = mc.to_rgb(color)
                    n_ch    = min(n_seg, 80)
                    chunk   = max(1, n_seg // n_ch)
                    for k in range(n_ch):
                        s = k * chunk; e = min(s + chunk + 1, n_seg + 1)
                        alpha = 0.05 + 0.90 * (k / max(n_ch - 1, 1))
                        ax.plot(haw[s:e], hbw[s:e], color=base, lw=1.1, alpha=alpha)

            elif trail_mode == 3:        # Densité / nuage de points
                ha = ha_full[idx_slice]; hb = hb_full[idx_slice]
                ax.scatter(ha, hb, color=color, s=0.4, alpha=0.12, linewidths=0)

            # trail_mode == 4 → Position seule, rien à tracer ici

            ax.scatter([ha_c], [hb_c], color=color, s=55,
                       zorder=5, label=name, edgecolors="white", linewidths=0.4)

    def _draw_3d(self, ax, d, idx_slice, idx_max, cx0, cy0, cz0, trail_mode, window_pts,
                 frame, scale, unit, ref_parent=None):
        """Dessine la vue 3D — 5 modes de tracé."""
        title_suffix = " — orbite locale" if frame == "parent" else ""
        ref_name = ""
        if frame == "parent" and ref_parent is not None:
            pname = d.body_names[ref_parent] if ref_parent < len(d.body_names) else f"Corps {ref_parent}"
            ref_name = f" ({pname} fixe, satellite en orbite)"
        ax.set_title(f"Vue 3D{title_suffix}{ref_name}", color=TEXT_COL, fontsize=9, pad=4)
        visible = self._visible_bodies() if hasattr(self, "_body_checks") and self._body_checks else set(range(d.n_bodies))
        for i in range(d.n_bodies):
            if i not in visible:
                continue
            if not self._show_body_in_frame(d, i, frame):
                continue
            x, y, z = d.positions_display(i, frame, ref_parent)
            x, y, z = x * scale, y * scale, z * scale
            color  = PALETTE[i % len(PALETTE)]
            name   = self._body_legend_name(d, i, frame)
            xi_f   = x[:idx_max + 1] - cx0 * scale
            yi_f   = y[:idx_max + 1] - cy0 * scale
            zi_f   = z[:idx_max + 1] - cz0 * scale
            xi_c   = x[idx_max] - cx0 * scale
            yi_c   = y[idx_max] - cy0 * scale
            zi_c   = z[idx_max] - cz0 * scale

            if trail_mode == 0:
                ax.plot(xi_f[idx_slice], yi_f[idx_slice], zi_f[idx_slice],
                        color=color, lw=0.6, alpha=0.45)
            elif trail_mode == 1:
                i0 = max(0, idx_max - window_pts)
                ax.plot(xi_f[i0:], yi_f[i0:], zi_f[i0:], color=color, lw=0.9, alpha=0.80)
            elif trail_mode == 2:
                i0  = max(0, idx_max - window_pts)
                xw  = xi_f[i0:]; yw = yi_f[i0:]; zw = zi_f[i0:]
                n_seg = len(xw) - 1
                if n_seg > 1:
                    import matplotlib.colors as mc
                    base = mc.to_rgb(color)
                    n_ch = min(n_seg, 60); chunk = max(1, n_seg // n_ch)
                    for k in range(n_ch):
                        s = k * chunk; e = min(s + chunk + 1, n_seg + 1)
                        alpha = 0.05 + 0.90 * (k / max(n_ch - 1, 1))
                        ax.plot(xw[s:e], yw[s:e], zw[s:e], color=base, lw=1.0, alpha=alpha)
            elif trail_mode == 3:
                ax.scatter(xi_f[idx_slice], yi_f[idx_slice], zi_f[idx_slice],
                           color=color, s=0.4, alpha=0.12, linewidths=0)

            ax.scatter([xi_c], [yi_c], [zi_c], color=color, s=55, zorder=5, label=name)

    def refresh(self):
        if self.data is None:
            return
        d        = self.data
        n_tot    = len(d.t_jours)
        idx_max  = min(self.slider_t.value(), n_tot - 1)
        t_cur    = d.t_jours[idx_max]
        self.lbl_t.setText(f"t = {t_cur:.0f} j  ({t_cur/365.25:.1f} ans)")

        sub_map  = {"Tous les points": 1, "× 2": 2, "× 5": 5, "× 10": 10, "× 50": 50}
        step     = sub_map[self.cb_subsample.currentText()]
        trail_mode  = self.cb_trail_mode.currentIndex()
        window_ans  = self.sp_window.value()
        # Convertir la fenêtre en nombre de points (approximation)
        dt_moy      = d.t_jours[-1] / max(n_tot - 1, 1)  # jours/point
        window_pts  = max(10, int(window_ans * 365.25 / dt_moy))
        center      = self.chk_center.isChecked()
        layout_mode = self.cb_layout.currentText()
        idx_slice   = slice(0, idx_max + 1, step)
        frame       = self._orbit_frame_mode()
        unit, scale = self._orbit_unit_scale(d)
        ref_parent  = d.reference_parent_index() if frame == "parent" else None

        cx0, cy0, cz0 = 0.0, 0.0, 0.0
        if center and frame != "parent":
            x0, y0, z0 = d.positions(0)
            cx0, cy0, cz0 = x0[idx_max], y0[idx_max], z0[idx_max]
        elif center and frame == "parent" and ref_parent is not None:
            cx0, cy0, cz0 = 0.0, 0.0, 0.0

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)

        # Définir la grille de sous-graphes selon la disposition choisie
        if "4 vues" in layout_mode:
            axes_2d = []
            ax_xy = self.figure.add_subplot(2, 2, 1)
            ax_xz = self.figure.add_subplot(2, 2, 2)
            ax_yz = self.figure.add_subplot(2, 2, 3)
            ax_3d = self.figure.add_subplot(2, 2, 4, projection="3d")
            for ax, plane in [(ax_xy, "XY"), (ax_xz, "XZ"), (ax_yz, "YZ")]:
                apply_dark_axes(ax)
                self._draw_2d(ax, d, idx_slice, idx_max, cx0, cy0, cz0, plane,
                              trail_mode, window_pts, frame, scale, unit, ref_parent)
            apply_dark_axes(ax_3d)
            self._draw_3d(ax_3d, d, idx_slice, idx_max, cx0, cy0, cz0,
                          trail_mode, window_pts, frame, scale, unit, ref_parent)
            legend_ax = ax_xy

        elif "2 vues" in layout_mode:
            ax_xy = self.figure.add_subplot(1, 2, 1)
            ax_3d = self.figure.add_subplot(1, 2, 2, projection="3d")
            apply_dark_axes(ax_xy)
            apply_dark_axes(ax_3d)
            self._draw_2d(ax_xy, d, idx_slice, idx_max, cx0, cy0, cz0, "XY",
                          trail_mode, window_pts, frame, scale, unit, ref_parent)
            self._draw_3d(ax_3d, d, idx_slice, idx_max, cx0, cy0, cz0,
                          trail_mode, window_pts, frame, scale, unit, ref_parent)
            legend_ax = ax_xy

        else:
            plane_map = {
                "Vue unique XY": ("XY", False),
                "Vue unique XZ": ("XZ", False),
                "Vue unique YZ": ("YZ", False),
                "Vue unique 3D": (None, True),
            }
            plane, is3d = plane_map.get(layout_mode, ("XY", False))
            if is3d:
                ax = self.figure.add_subplot(1, 1, 1, projection="3d")
                apply_dark_axes(ax)
                self._draw_3d(ax, d, idx_slice, idx_max, cx0, cy0, cz0,
                              trail_mode, window_pts, frame, scale, unit, ref_parent)
            else:
                ax = self.figure.add_subplot(1, 1, 1)
                apply_dark_axes(ax)
                self._draw_2d(ax, d, idx_slice, idx_max, cx0, cy0, cz0, plane,
                              trail_mode, window_pts, frame, scale, unit, ref_parent)
            legend_ax = ax

        # Légende sur le premier axe
        handles, labels = legend_ax.get_legend_handles_labels()
        if handles:
            legend_ax.legend(handles, labels, loc="upper right", fontsize=8,
                             facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)

        frame_lbl = " — zoom satellite" if frame == "parent" else ""
        title = (
            f"Orbites — {d.n_bodies} corps{frame_lbl}  |  "
            f"t = {t_cur/365.25:.1f} ans  |  unité : {unit}"
        )
        self.figure.suptitle(title, color=TEXT_COL, fontsize=10, y=0.99)
        self.figure.tight_layout(rect=[0, 0, 1, 0.97])

        duree = d.t_jours[-1] / 365.25
        self.stats_label.setText(
            f"Durée totale : {duree:.1f} ans  |  Points : {n_tot:,}  |  "
            f"Fichier : {os.path.basename(d.parquet_path)}"
        )
        self.canvas.draw()


# ─────────────────────────────────────────────────────────
#  Fenêtre plein écran pour un graphe de distance
# ─────────────────────────────────────────────────────────
class DistanceFullScreen(QMainWindow):
    def __init__(self, fig, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plein écran — {title}")
        self.resize(1100, 680)
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet(f"background:{FIG_VOID};")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        canvas = FigureCanvas(fig)
        nav = NavigationToolbar(canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(canvas)
        canvas.draw()


# ─────────────────────────────────────────────────────────
#  Onglet 2 — Distances (un graphique par paire + plein écran)
# ─────────────────────────────────────────────────────────
class DistanceTab(QWidget):

    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._fullscreen_wins = []   # garder les références vivantes
        self._pairs_data = []        # (lbl, dist_array) calculés
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Sous-onglets : courbes temporelles | histogramme
        self._subtabs = QTabWidget()
        self._subtabs.setTabPosition(QTabWidget.North)
        layout.addWidget(self._subtabs)

        # ── Sous-onglet 1 : Courbes temporelles ──────────────────────────────
        tab_courbes = QWidget()
        lay_c = QVBoxLayout(tab_courbes)
        lay_c.setContentsMargins(4, 4, 4, 4)

        ctrl = QHBoxLayout()
        self.cb_unit = QComboBox()
        self.cb_unit.addItems(["UA", "km", "m"])
        self.cb_unit.currentIndexChanged.connect(self.refresh)

        self.cb_cols = QComboBox()
        self.cb_cols.addItems(["1 colonne", "2 colonnes", "3 colonnes"])
        self.cb_cols.setCurrentIndex(1)
        self.cb_cols.currentIndexChanged.connect(self.refresh)

        self.chk_log = QCheckBox("Echelle log Y")
        self.chk_log.stateChanged.connect(self.refresh)

        self.chk_shared_y = QCheckBox("Axe Y commun")
        self.chk_shared_y.setChecked(False)
        self.chk_shared_y.stateChanged.connect(self.refresh)

        self.cb_fullscreen = QComboBox()
        self.cb_fullscreen.addItem("-- Plein ecran --")
        self.cb_fullscreen.setMinimumWidth(200)
        btn_fs = QPushButton("Ouvrir plein ecran")
        btn_fs.clicked.connect(self._open_fullscreen)

        btn_png_c = QPushButton("💾 PNG")
        btn_png_c.setToolTip("Exporter le graphe courant en image PNG")
        btn_png_c.clicked.connect(lambda: self._export_png(self.figure))
        btn_csv_c = QPushButton("📄 CSV")
        btn_csv_c.setToolTip("Exporter les distances en CSV")
        btn_csv_c.clicked.connect(self._export_csv_distances)

        ctrl.addWidget(QLabel("Unite :"));   ctrl.addWidget(self.cb_unit)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Colonnes :")); ctrl.addWidget(self.cb_cols)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.chk_log);        ctrl.addWidget(self.chk_shared_y)
        ctrl.addStretch()
        ctrl.addWidget(btn_png_c);           ctrl.addWidget(btn_csv_c)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.cb_fullscreen);  ctrl.addWidget(btn_fs)
        lay_c.addLayout(ctrl)

        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        lay_c.addWidget(nav)
        lay_c.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setMaximumHeight(130)
        lay_c.addWidget(self.table)

        self._subtabs.addTab(tab_courbes, "Courbes temporelles")

        # ── Sous-onglet 2 : Histogramme ──────────────────────────────────────
        tab_histo = QWidget()
        lay_h = QVBoxLayout(tab_histo)
        lay_h.setContentsMargins(4, 4, 4, 4)

        ctrl_h = QHBoxLayout()
        self.cb_histo_pair = QComboBox()
        self.cb_histo_pair.setMinimumWidth(220)
        self.cb_histo_pair.currentIndexChanged.connect(self._refresh_histo)

        self.cb_histo_unit = QComboBox()
        self.cb_histo_unit.addItems(["UA", "km", "m"])
        self.cb_histo_unit.currentIndexChanged.connect(self._refresh_histo)

        self.sp_histo_bins = QSpinBox()
        self.sp_histo_bins.setRange(10, 500); self.sp_histo_bins.setValue(80)
        self.sp_histo_bins.setSuffix(" bins")
        self.sp_histo_bins.setToolTip("Nombre de barres de l'histogramme")
        self.sp_histo_bins.valueChanged.connect(self._refresh_histo)

        btn_png_h = QPushButton("💾 PNG")
        btn_png_h.setToolTip("Exporter l'histogramme en PNG")
        btn_png_h.clicked.connect(lambda: self._export_png(self.figure_histo))
        btn_csv_h = QPushButton("📄 CSV")
        btn_csv_h.setToolTip("Exporter les données de la paire sélectionnée en CSV")
        btn_csv_h.clicked.connect(self._export_csv_histo)

        ctrl_h.addWidget(QLabel("Paire :"));  ctrl_h.addWidget(self.cb_histo_pair)
        ctrl_h.addSpacing(12)
        ctrl_h.addWidget(QLabel("Unité :"));  ctrl_h.addWidget(self.cb_histo_unit)
        ctrl_h.addSpacing(12)
        ctrl_h.addWidget(self.sp_histo_bins)
        ctrl_h.addStretch()
        ctrl_h.addWidget(btn_png_h); ctrl_h.addWidget(btn_csv_h)
        lay_h.addLayout(ctrl_h)

        self.figure_histo = Figure(facecolor=FIG_PANEL)
        self.canvas_histo = FigureCanvas(self.figure_histo)
        self.canvas_histo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav_h = NavigationToolbar(self.canvas_histo, self)
        nav_h.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        lay_h.addWidget(nav_h)
        lay_h.addWidget(self.canvas_histo)

        self.lbl_histo_stats = QLabel("")
        self.lbl_histo_stats.setObjectName("muted")
        self.lbl_histo_stats.setAlignment(Qt.AlignCenter)
        lay_h.addWidget(self.lbl_histo_stats)

        self._subtabs.addTab(tab_histo, "Histogramme des distances")

    def load_data(self, data: SimData):
        self.data = data
        # Peupler le sélecteur de paires pour l'histogramme
        self.cb_histo_pair.blockSignals(True)
        self.cb_histo_pair.clear()
        for i in range(data.n_bodies):
            for j in range(i + 1, data.n_bodies):
                ni = data.body_names[i] if i < len(data.body_names) else f"C{i}"
                nj = data.body_names[j] if j < len(data.body_names) else f"C{j}"
                self.cb_histo_pair.addItem(f"{ni} — {nj}", (i, j))
        self.cb_histo_pair.blockSignals(False)
        self.refresh()
        self._refresh_histo()

    def _compute_pairs(self):
        """Retourne la liste des paires avec distances calculées."""
        d = self.data
        unit = self.cb_unit.currentText()
        scale = {"UA": 1.0, "km": UA / 1e3, "m": UA}[unit]
        pairs = []
        for i in range(d.n_bodies):
            for j in range(i + 1, d.n_bodies):
                ni = d.body_names[i] if i < len(d.body_names) else f"C{i}"
                nj = d.body_names[j] if j < len(d.body_names) else f"C{j}"
                lbl = f"{ni} — {nj}"
                dist = d.distance_pair(i, j) * scale
                pairs.append((i, j, lbl, dist))
        return pairs

    def _draw_pair_on_ax(self, ax, t_ans, i, j, lbl, dist, unit, log_y, shared_y,
                         y_min_global, y_max_global, show_xlabel=True):
        """Dessine une paire sur un axe donné."""
        color = PALETTE[(i + j) % len(PALETTE)]
        ax.plot(t_ans, dist, color=color, lw=1.0, alpha=0.9)
        ax.fill_between(t_ans, dist, alpha=0.08, color=color)

        idx_min = int(np.argmin(dist))
        idx_max_d = int(np.argmax(dist))
        ax.axhline(dist[idx_min],   color=color, lw=0.5, linestyle=":", alpha=0.6)
        ax.axhline(dist[idx_max_d], color=color, lw=0.5, linestyle=":", alpha=0.6)

        ax.set_title(lbl, color=TEXT_COL, fontsize=9, pad=3)
        if show_xlabel:
            ax.set_xlabel("Temps (ans)", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(f"Dist. ({unit})", fontsize=8)
        ax.tick_params(labelsize=8)

        if log_y:
            ax.set_yscale("log")
        elif shared_y:
            ax.set_ylim(y_min_global * 0.95, y_max_global * 1.05)

        ax.annotate(
            f"min={dist[idx_min]:.3f}",
            xy=(t_ans[idx_min], dist[idx_min]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7, color=color, alpha=0.85,
        )

    def refresh(self):
        if self.data is None:
            return
        d = self.data
        unit = self.cb_unit.currentText()
        t_ans = d.t_jours / 365.25
        log_y = self.chk_log.isChecked()
        shared_y = self.chk_shared_y.isChecked()
        n_cols = int(self.cb_cols.currentText()[0])

        pairs = self._compute_pairs()
        self._pairs_data = pairs   # garder pour plein écran

        # Mettre à jour le sélecteur plein écran
        current_fs = self.cb_fullscreen.currentText()
        self.cb_fullscreen.blockSignals(True)
        self.cb_fullscreen.clear()
        self.cb_fullscreen.addItem("-- Plein ecran --")
        for _, _, lbl, _ in pairs:
            self.cb_fullscreen.addItem(lbl)
        idx = self.cb_fullscreen.findText(current_fs)
        self.cb_fullscreen.setCurrentIndex(max(0, idx))
        self.cb_fullscreen.blockSignals(False)

        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        n_rows = math.ceil(n_pairs / n_cols)
        all_dists = [p[3] for p in pairs]
        y_min_global = min(d.min() for d in all_dists)
        y_max_global = max(d.max() for d in all_dists)

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)

        table_data = []
        for k, (i, j, lbl, dist) in enumerate(pairs):
            row_idx = k // n_cols
            ax = self.figure.add_subplot(n_rows, n_cols, k + 1)
            apply_dark_axes(ax)
            show_x = (row_idx == n_rows - 1) or (k + n_cols >= n_pairs)
            self._draw_pair_on_ax(ax, t_ans, i, j, lbl, dist, unit,
                                  log_y, shared_y, y_min_global, y_max_global, show_x)
            table_data.append((lbl, dist.min(), dist.max(), dist.mean()))

        self.figure.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)
        self.canvas.draw()

        # Table résumé
        self.table.clear()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Paire", f"Min ({unit})", f"Max ({unit})", f"Moy ({unit})"])
        self.table.setRowCount(len(table_data))
        for r, (lbl, mn, mx, mean) in enumerate(table_data):
            self.table.setItem(r, 0, QTableWidgetItem(lbl))
            self.table.setItem(r, 1, QTableWidgetItem(f"{mn:.4f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{mx:.4f}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{mean:.4f}"))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

    def _refresh_histo(self, *_):
        """Dessine l'histogramme de la paire sélectionnée."""
        if self.data is None or self.cb_histo_pair.count() == 0:
            return
        pair = self.cb_histo_pair.currentData()
        if pair is None:
            return
        i, j = pair
        unit = self.cb_histo_unit.currentText()
        scale = {"UA": 1.0, "km": UA / 1e3, "m": UA}[unit]
        dist  = self.data.distance_pair(i, j) * scale
        bins  = self.sp_histo_bins.value()
        lbl   = self.cb_histo_pair.currentText()

        color = PALETTE[(i + j) % len(PALETTE)]
        self.figure_histo.clear()
        self.figure_histo.patch.set_facecolor(FIG_PANEL)
        ax = self.figure_histo.add_subplot(111)
        apply_dark_axes(ax)
        n, edges, patches = ax.hist(dist, bins=bins, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(float(np.mean(dist)),   color="white",   lw=1.2, linestyle="--",
                   label=f"Moyenne : {np.mean(dist):.3f} {unit}")
        ax.axvline(float(np.median(dist)), color="#FFD700", lw=1.2, linestyle=":",
                   label=f"Médiane : {np.median(dist):.3f} {unit}")
        ax.set_xlabel(f"Distance ({unit})", fontsize=9)
        ax.set_ylabel("Occurrences", fontsize=9)
        ax.set_title(f"Distribution des distances — {lbl}", color=TEXT_COL, fontsize=10)
        ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=8)
        self.figure_histo.tight_layout(pad=1.3)
        self.canvas_histo.draw()

        self.lbl_histo_stats.setText(
            f"Min : {dist.min():.4f}  |  Max : {dist.max():.4f}  |  "
            f"Écart-type : {dist.std():.4f}  ({unit})"
        )

    @staticmethod
    def _export_png(figure):
        """Ouvre un dialogue et sauvegarde la figure en PNG."""
        path, _ = QFileDialog.getSaveFileName(
            None, "Exporter en PNG", "graphe.png", "Images PNG (*.png)")
        if path:
            figure.savefig(path, dpi=150, bbox_inches="tight",
                           facecolor=figure.get_facecolor())

    def _export_csv_distances(self):
        """Exporte toutes les paires de distances en CSV."""
        if not self.data or not self._pairs_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter distances CSV", "distances.csv", "CSV (*.csv)")
        if not path:
            return
        unit = self.cb_unit.currentText()
        scale = {"UA": 1.0, "km": UA / 1e3, "m": UA}[unit]
        t_ans = self.data.t_jours / 365.25
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            headers = ["temps_ans"] + [lbl for _, _, lbl, _ in self._pairs_data]
            w.writerow(headers)
            for k in range(len(t_ans)):
                row = [f"{t_ans[k]:.6f}"] + [f"{dist[k]:.6f}" for _, _, _, dist in self._pairs_data]
                w.writerow(row)

    def _export_csv_histo(self):
        """Exporte la série de distances de la paire sélectionnée en CSV."""
        if self.data is None or self.cb_histo_pair.count() == 0:
            return
        pair = self.cb_histo_pair.currentData()
        if pair is None:
            return
        i, j = pair
        lbl  = self.cb_histo_pair.currentText().replace(" — ", "_vs_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter CSV", f"distances_{lbl}.csv", "CSV (*.csv)")
        if not path:
            return
        unit  = self.cb_histo_unit.currentText()
        scale = {"UA": 1.0, "km": UA / 1e3, "m": UA}[unit]
        dist  = self.data.distance_pair(i, j) * scale
        t_ans = self.data.t_jours / 365.25
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"temps_ans", f"distance_{unit}"])
            for t, v in zip(t_ans, dist):
                w.writerow([f"{t:.6f}", f"{v:.6f}"])

    def _open_fullscreen(self):
        """Ouvre une paire sélectionnée en plein écran dans sa propre fenêtre."""
        if not self._pairs_data:
            return
        sel = self.cb_fullscreen.currentText()
        if sel.startswith("--"):
            QMessageBox.information(self, "Info",
                "Selectionne une paire dans le menu deroulant puis clique sur Ouvrir plein ecran.")
            return

        # Trouver la paire
        target = next(((i, j, lbl, dist) for i, j, lbl, dist in self._pairs_data
                       if lbl == sel), None)
        if target is None:
            return

        i, j, lbl, dist = target
        d = self.data
        unit = self.cb_unit.currentText()
        t_ans = d.t_jours / 365.25
        log_y = self.chk_log.isChecked()

        fig = Figure(facecolor=FIG_PANEL, figsize=(10, 5))
        ax = fig.add_subplot(111)
        apply_dark_axes(ax)
        self._draw_pair_on_ax(ax, t_ans, i, j, lbl, dist, unit,
                              log_y, False, dist.min(), dist.max(), True)

        # Stats sur le graphe
        ax.annotate(
            f"max={dist.max():.3f} {unit}  |  moy={dist.mean():.3f} {unit}",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9, color=TEXT_COL,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, alpha=0.8)
        )
        fig.tight_layout(pad=1.5)

        win = DistanceFullScreen(fig, lbl, self)
        win.setStyleSheet(STYLESHEET)
        win.show()
        self._fullscreen_wins.append(win)  # éviter garbage collection


# ─────────────────────────────────────────────────────────
#  Onglet 3 — Énergie
# ─────────────────────────────────────────────────────────
class EnergieTab(QWidget):

    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Métriques synthèse
        self.cards = QHBoxLayout()
        self.card_derive_e  = self._make_card("Dérive énergie (totale)", "—")
        self.card_derive_l  = self._make_card("Dérive moment L", "—")
        self.card_derive_em = self._make_card("Dérive énergie (moy/an)", "—")
        self.card_duree     = self._make_card("Durée simulée", "—")
        self.cards.addWidget(self.card_derive_e[0])
        self.cards.addWidget(self.card_derive_l[0])
        self.cards.addWidget(self.card_derive_em[0])
        self.cards.addWidget(self.card_duree[0])
        layout.addLayout(self.cards)

        # Options
        ctrl = QHBoxLayout()
        self.cb_subplot = QComboBox()
        self.cb_subplot.addItems(["Énergie + Moment L", "Énergie seule", "Moment L seul", "Pas de temps (dt)"])
        self.cb_subplot.currentIndexChanged.connect(self.refresh)
        self.chk_log = QCheckBox("Éch. log |dérive|")
        self.chk_log.stateChanged.connect(self.refresh)
        ctrl.addWidget(QLabel("Graphe :"))
        ctrl.addWidget(self.cb_subplot)
        ctrl.addSpacing(16)
        ctrl.addWidget(self.chk_log)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.figure = make_figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

    def _make_card(self, title, value):
        frame = QFrame()
        frame.setStyleSheet(f"background: {CARD_BG}; border-radius: 8px; padding: 4px;")
        vl = QVBoxLayout(frame)
        lbl_t = QLabel(title)
        lbl_t.setObjectName("muted")
        lbl_t.setAlignment(Qt.AlignCenter)
        lbl_v = QLabel(value)
        lbl_v.setAlignment(Qt.AlignCenter)
        lbl_v.setFont(QFont("Segoe UI", 15, QFont.Bold))
        lbl_v.setStyleSheet(f"color:{MUTED};")
        vl.addWidget(lbl_t)
        vl.addWidget(lbl_v)
        return frame, lbl_v

    def _update_card(self, card_tuple, text):
        card_tuple[1].setText(text)

    def load_data(self, data: SimData):
        self.data = data
        m = data.metrics
        self._update_card(self.card_derive_e,  f"{m.get('derive_e', float('nan')):.2e} %")
        self._update_card(self.card_derive_l,  f"{m.get('derive_l', float('nan')):.2e} %")
        self._update_card(self.card_derive_em, f"{m.get('derive_e_mean', float('nan')):.2e} %/an")
        self._update_card(self.card_duree,     f"{data.duree_ans():.1f} ans")
        self.refresh()

    def refresh(self):
        if self.data is None:
            return
        d = self.data
        t_ans = d.t_jours / 365.25
        mode = self.cb_subplot.currentText()
        log_y = self.chk_log.isChecked()

        self.figure.clear()

        if "+" in mode:
            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)
        else:
            ax1 = self.figure.add_subplot(111)
            ax2 = None

        apply_dark_axes(ax1)
        if ax2:
            apply_dark_axes(ax2)

        drift = d.energy_drift()

        if "Énergie" in mode or mode == "Énergie + Moment L":
            y = np.abs(drift) if log_y else drift
            ax1.plot(t_ans, y, color=ACCENT, lw=0.8, label="Dérive ΔE/E₀ (%)")
            if log_y:
                ax1.set_yscale("log")
            ax1.set_ylabel("ΔE/E₀ (%)" if not log_y else "|ΔE/E₀| (%, log)")
            ax1.set_title("Dérive d'énergie", color=TEXT_COL, fontsize=10)
            ax1.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)
            if ax2 is None:
                ax1.set_xlabel("Temps (ans)")

        if "Moment L" in mode:
            L = d.angular_momentum()
            L0 = L[0]
            dL = (L - L0) / abs(L0) * 100 if L0 != 0 else L - L0
            target = ax2 if ax2 else ax1
            target.plot(t_ans, np.abs(dL) if log_y else dL,
                        color="#66BB6A", lw=0.8, label="Dérive ΔL/L₀ (%)")
            if log_y:
                target.set_yscale("log")
            target.set_ylabel("ΔL/L₀ (%)" if not log_y else "|ΔL/L₀| (%, log)")
            target.set_title("Dérive moment cinétique", color=TEXT_COL, fontsize=10)
            target.set_xlabel("Temps (ans)")
            target.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)

        if "Pas de temps" in mode:
            if "dt" in d.df.columns:
                ax1.plot(t_ans, d.df["dt"].values, color="#4FC3F7", lw=0.7)
                ax1.set_ylabel("dt (s)")
                ax1.set_xlabel("Temps (ans)")
                ax1.set_title("Pas de temps adaptatif", color=TEXT_COL, fontsize=10)

        if ax2:
            ax1.set_xlabel("")

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()


# ─────────────────────────────────────────────────────────
#  Onglet 4 — Vitesses
# ─────────────────────────────────────────────────────────
class VitesseTab(QWidget):
    """ Affiche la vitesse scalaire de chaque corps au fil du temps,
    ainsi que les composantes Vx, Vy, Vz séparément.
    Utile pour repérer des accélérations brutales (approches proches)
    ou vérifier que les vitesses initiales sont cohérentes.
    """
    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Info explicative
        info = QLabel(
            "Ce graphe montre la vitesse de chaque corps en km/s. Une vitesse constante = orbite stable. Un pic = approche proche ou perturbation."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Vitesse scalaire", "Vx", "Vy", "Vz", "Toutes composantes"])
        self.cb_mode.currentIndexChanged.connect(self.refresh)

        self.cb_corps = QComboBox()
        self.cb_corps.addItem("Tous les corps")
        self.cb_corps.currentIndexChanged.connect(self.refresh)

        self.chk_log = QCheckBox("Echelle log")
        self.chk_log.stateChanged.connect(self.refresh)

        ctrl.addWidget(QLabel("Afficher :"))
        ctrl.addWidget(self.cb_mode)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Corps :"))
        ctrl.addWidget(self.cb_corps)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.chk_log)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setMaximumHeight(120)
        layout.addWidget(self.table)

    def load_data(self, data: SimData):
        self.data = data
        self.cb_corps.blockSignals(True)
        self.cb_corps.clear()
        self.cb_corps.addItem("Tous les corps")
        for name in data.body_names:
            self.cb_corps.addItem(name)
        self.cb_corps.blockSignals(False)
        self.refresh()

    def _get_vel(self, d: SimData, i: int):
        """Retourne (vx,vy,vz,v_scalar) en km/s pour le corps i."""
        vx = d.df[f"Vx{i}"].values / 1e3
        vy = d.df[f"Vy{i}"].values / 1e3
        vz = d.df[f"Vz{i}"].values / 1e3 if f"Vz{i}" in d.df.columns else np.zeros(len(vx))
        v  = np.sqrt(vx**2 + vy**2 + vz**2)
        return vx, vy, vz, v

    def refresh(self):
        if self.data is None:
            return
        d = self.data
        # Vérifier que les colonnes vitesse existent
        if "Vx0" not in d.df.columns:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            apply_dark_axes(ax)
            ax.text(0.5, 0.5, "Colonnes Vx/Vy non disponibles dans ce fichier",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        t_ans = d.t_jours / 365.25
        mode = self.cb_mode.currentText()
        corps_sel = self.cb_corps.currentText()
        log_y = self.chk_log.isChecked()

        indices = list(range(d.n_bodies))
        if corps_sel != "Tous les corps":
            idx = self.cb_corps.currentIndex() - 1
            if 0 <= idx < d.n_bodies:
                indices = [idx]

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)

        if mode == "Toutes composantes" and len(indices) == 1:
            axes = [self.figure.add_subplot(3, 1, k+1) for k in range(3)]
            labels = ["Vx (km/s)", "Vy (km/s)", "Vz (km/s)"]
            for ax in axes: apply_dark_axes(ax)
            i = indices[0]
            vx, vy, vz, _ = self._get_vel(d, i)
            color = PALETTE[i % len(PALETTE)]
            for ax, data_v, lbl in zip(axes, [vx, vy, vz], labels):
                ax.plot(t_ans, data_v, color=color, lw=0.8)
                ax.set_ylabel(lbl, fontsize=8)
                if log_y: ax.set_yscale("symlog")
            axes[-1].set_xlabel("Temps (ans)")
            self.figure.suptitle(
                f"Composantes vitesse — {d.body_names[i]}",
                color=TEXT_COL, fontsize=10)
        else:
            ax = self.figure.add_subplot(111)
            apply_dark_axes(ax)
            comp_map = {"Vitesse scalaire": 3, "Vx": 0, "Vy": 1, "Vz": 2}
            comp_idx = comp_map.get(mode, 3)
            table_data = []
            for i in indices:
                vx, vy, vz, v = self._get_vel(d, i)
                comps = [vx, vy, vz, v]
                y = comps[comp_idx]
                color = PALETTE[i % len(PALETTE)]
                name = d.body_names[i] if i < len(d.body_names) else f"C{i}"
                ax.plot(t_ans, y, color=color, lw=0.8, label=name, alpha=0.85)
                table_data.append((name, float(y.min()), float(y.max()), float(y.mean())))
            if log_y: ax.set_yscale("symlog")
            ax.set_xlabel("Temps (ans)")
            ax.set_ylabel(f"{mode} (km/s)")
            ax.set_title(mode, color=TEXT_COL, fontsize=10)
            ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=9)

            self.table.clear()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["Corps","Min (km/s)","Max (km/s)","Moy (km/s)"])
            self.table.setRowCount(len(table_data))
            for r,(n,mn,mx,me) in enumerate(table_data):
                for c,v in enumerate([n,f"{mn:.2f}",f"{mx:.2f}",f"{me:.2f}"]):
                    self.table.setItem(r,c,QTableWidgetItem(v))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)

        self.figure.tight_layout(pad=1.3)
        self.canvas.draw()


# ─────────────────────────────────────────────────────────
#  Onglet 5 — Stabilité orbitale
# ─────────────────────────────────────────────────────────
class StabiliteTab(QWidget):
    """ Analyse la stabilité de chaque orbite via plusieurs indicateurs :
    - Excentricité instantanée (comment l'ellipse se déforme au cours du temps)
    - Demi-grand axe (comment la taille de l'orbite évolue)
    - Inclinaison (si le plan orbital change)
    Ces paramètres sont calculés à partir des positions et vitesses
    à chaque instant, en approximation képlérienne à 2 corps.
    """
    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "Stabilite orbitale : ces courbes montrent comment les parametres de chaque orbite evoluent. "
            "Excentricite : 0 = cercle parfait, 1 = parabole (evasion). "
            "Demi-grand axe : taille de l'orbite en UA. "
            "Inclinaison : angle du plan orbital en degres. "
            "Si ces valeurs varient peu = systeme stable. Si elles derivent ou oscillent fortement = instabilite."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.cb_param = QComboBox()
        self.cb_param.addItems([
            "Excentricite",
            "Demi-grand axe (UA)",
            "Inclinaison (deg)",
            "Longitude périhélie ω (deg)",
            "Nœud ascendant Ω (deg)",
            "Les trois",
            "Angles orbitaux (ω + Ω)",
        ])
        self.cb_param.currentIndexChanged.connect(self.refresh)
        self.cb_ref = QComboBox()
        self.cb_ref.addItem("Corps 0 (central)")
        self.cb_ref.currentIndexChanged.connect(self.refresh)

        btn_png = QPushButton("💾 PNG")
        btn_png.setToolTip("Exporter le graphe en PNG")
        btn_png.clicked.connect(self._export_png)
        btn_csv = QPushButton("📄 CSV")
        btn_csv.setToolTip("Exporter les éléments orbitaux en CSV")
        btn_csv.clicked.connect(self._export_csv)

        ctrl.addWidget(QLabel("Parametre :"))
        ctrl.addWidget(self.cb_param)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Reference :"))
        ctrl.addWidget(self.cb_ref)
        ctrl.addStretch()
        ctrl.addWidget(btn_png)
        ctrl.addWidget(btn_csv)
        layout.addLayout(ctrl)

        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{FIG_PANEL}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setMaximumHeight(120)
        layout.addWidget(self.table)

    def load_data(self, data: SimData):
        self.data = data
        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        for i, name in enumerate(data.body_names):
            self.cb_ref.addItem(f"Corps {i} — {name}")
        self.cb_ref.blockSignals(False)
        self.refresh()

    def _orbital_elements(self, d: SimData, i: int, ref: int):
        """ Calcule excentricité, demi-grand axe, inclinaison, ω et Ω
        de manière instantanée pour corps i autour du corps ref.
        Approximation keplérienne à 2 corps.
        Retourne : (e, a, incl, omega, Omega) — tous en arrays numpy.
        """
        G_SI = 6.674e-11
        m0 = float(d.df.get("M0", pd.Series([1.989e30])).iloc[0]) if "M0" in d.df.columns else 1.989e30
        xi, yi, zi = (d.df[f"X{i}"].values, d.df[f"Y{i}"].values,
                      d.df[f"Z{i}"].values if f"Z{i}" in d.df.columns else np.zeros(len(d.t_jours)))
        xr, yr, zr = (d.df[f"X{ref}"].values, d.df[f"Y{ref}"].values,
                      d.df[f"Z{ref}"].values if f"Z{ref}" in d.df.columns else np.zeros(len(d.t_jours)))
        rx, ry, rz = xi - xr, yi - yr, zi - zr
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        if "Vx0" not in d.df.columns:
            return None, None, None, None, None

        vxi, vyi = d.df[f"Vx{i}"].values, d.df[f"Vy{i}"].values
        vzi = d.df[f"Vz{i}"].values if f"Vz{i}" in d.df.columns else np.zeros(len(d.t_jours))
        vxr, vyr = d.df[f"Vx{ref}"].values, d.df[f"Vy{ref}"].values
        vzr = d.df[f"Vz{ref}"].values if f"Vz{ref}" in d.df.columns else np.zeros(len(d.t_jours))
        dvx, dvy, dvz = vxi - vxr, vyi - vyr, vzi - vzr
        v2 = dvx**2 + dvy**2 + dvz**2

        mu = G_SI * m0

        # Énergie spécifique → demi-grand axe
        eps = v2 / 2 - mu / np.maximum(r, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.where(eps < 0, -mu / (2 * eps), np.nan) / UA

        # Moment cinétique h = r × v
        hx = ry * dvz - rz * dvy
        hy = rz * dvx - rx * dvz
        hz = rx * dvy - ry * dvx
        h2 = hx**2 + hy**2 + hz**2
        h  = np.sqrt(h2)

        # Excentricité
        with np.errstate(invalid='ignore'):
            e2 = 1 + 2 * eps * h2 / (mu**2)
            e  = np.sqrt(np.clip(e2, 0, None))

        # Inclinaison
        with np.errstate(invalid='ignore'):
            incl = np.degrees(np.arccos(np.clip(hz / np.maximum(h, 1e-30), -1, 1)))

        # Vecteur nœud N = Z × h
        nx = -hy
        ny =  hx
        nz =  np.zeros_like(hx)
        n_mag = np.sqrt(nx**2 + ny**2)

        # Nœud ascendant Ω (longitude du nœud ascendant)
        with np.errstate(invalid='ignore'):
            Omega = np.degrees(np.arctan2(ny, nx)) % 360.0

        # Vecteur excentricité e_vec = (v×h)/mu - r_hat
        ex = (dvy * hz - dvz * hy) / mu - rx / np.maximum(r, 1e-30)
        ey = (dvz * hx - dvx * hz) / mu - ry / np.maximum(r, 1e-30)
        ez = (dvx * hy - dvy * hx) / mu - rz / np.maximum(r, 1e-30)

        # Longitude du périhélie ω (argument du périapsis + Ω)
        with np.errstate(invalid='ignore'):
            omega = np.degrees(np.arctan2(ey, ex)) % 360.0

        return e, a, incl, omega, Omega

    def refresh(self):
        if self.data is None:
            return
        d = self.data
        t_ans = d.t_jours / 365.25
        param = self.cb_param.currentText()
        ref_idx = self.cb_ref.currentIndex()

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)

        angles_mode   = param == "Angles orbitaux (ω + Ω)"
        trois_mode    = param == "Les trois"
        n_plots = 2 if angles_mode else (3 if trois_mode else 1)
        axes = [self.figure.add_subplot(n_plots, 1, k+1) for k in range(n_plots)]
        for ax in axes:
            apply_dark_axes(ax)

        table_data = []
        self._last_elements = []   # pour export CSV
        for i in range(d.n_bodies):
            if i == ref_idx:
                continue
            e, a, incl, omega, Omega = self._orbital_elements(d, i, ref_idx)
            if e is None:
                continue
            color = PALETTE[i % len(PALETTE)]
            name = d.body_names[i] if i < len(d.body_names) else f"C{i}"
            self._last_elements.append((name, t_ans, e, a, incl, omega, Omega))

            if param == "Excentricite":
                axes[0].plot(t_ans, e, color=color, lw=0.8, label=name)
                axes[0].set_ylabel("Excentricité")
                axes[0].axhline(1.0, color="#EF5350", lw=0.8, linestyle="--", alpha=0.6)
                axes[0].set_xlabel("Temps (ans)")
            elif param == "Demi-grand axe (UA)":
                axes[0].plot(t_ans, a, color=color, lw=0.8, label=name)
                axes[0].set_ylabel("Demi-grand axe (UA)")
                axes[0].set_xlabel("Temps (ans)")
            elif param == "Inclinaison (deg)":
                axes[0].plot(t_ans, incl, color=color, lw=0.8, label=name)
                axes[0].set_ylabel("Inclinaison (°)")
                axes[0].set_xlabel("Temps (ans)")
            elif param == "Longitude périhélie ω (deg)":
                axes[0].plot(t_ans, omega, color=color, lw=0.8, label=name)
                axes[0].set_ylabel("ω — Longitude périhélie (°)")
                axes[0].set_xlabel("Temps (ans)")
            elif param == "Nœud ascendant Ω (deg)":
                axes[0].plot(t_ans, Omega, color=color, lw=0.8, label=name)
                axes[0].set_ylabel("Ω — Nœud ascendant (°)")
                axes[0].set_xlabel("Temps (ans)")
            elif angles_mode:
                axes[0].plot(t_ans, omega, color=color, lw=0.7, label=name)
                axes[1].plot(t_ans, Omega, color=color, lw=0.7, label=name)
                axes[0].set_ylabel("ω — Longitude périhélie (°)")
                axes[1].set_ylabel("Ω — Nœud ascendant (°)")
                axes[1].set_xlabel("Temps (ans)")
            elif trois_mode:
                axes[0].plot(t_ans, e,    color=color, lw=0.7, label=name)
                axes[1].plot(t_ans, a,    color=color, lw=0.7, label=name)
                axes[2].plot(t_ans, incl, color=color, lw=0.7, label=name)
                axes[0].set_ylabel("Excentricité")
                axes[0].axhline(1.0, color="#EF5350", lw=0.8, linestyle="--", alpha=0.5)
                axes[1].set_ylabel("Demi-grand axe (UA)")
                axes[2].set_ylabel("Inclinaison (°)")
                axes[2].set_xlabel("Temps (ans)")

            e_mean     = float(np.nanmean(e))
            a_mean     = float(np.nanmean(a)) if a is not None else float('nan')
            omega_mean = float(np.nanmean(omega))
            Omega_mean = float(np.nanmean(Omega))
            table_data.append((name,
                                f"{e_mean:.4f}",
                                f"{a_mean:.4f}",
                                f"{float(np.nanmean(incl)):.2f}",
                                f"{omega_mean:.2f}",
                                f"{Omega_mean:.2f}"))

        for ax in axes:
            ax.legend(facecolor=CARD_BG, edgecolor=BORDER,
                      labelcolor=TEXT_COL, fontsize=8)

        self.figure.suptitle(
            f"Stabilité orbitale — {param}  (réf: {d.body_names[ref_idx]})",
            color=TEXT_COL, fontsize=10)
        self.figure.tight_layout(pad=1.3, rect=[0, 0, 1, 0.96])
        self.canvas.draw()

        self.table.clear()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Corps", "e moy", "a moy (UA)", "i moy (°)", "ω moy (°)", "Ω moy (°)"])
        self.table.setRowCount(len(table_data))
        for r, row in enumerate(table_data):
            for c, v in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en PNG", "stabilite.png", "Images PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches="tight",
                                facecolor=self.figure.get_facecolor())

    def _export_csv(self):
        if not hasattr(self, "_last_elements") or not self._last_elements:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter CSV", "elements_orbitaux.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["corps", "temps_ans", "e", "a_UA", "incl_deg", "omega_deg", "Omega_deg"])
            for name, t_ans, e, a, incl, omega, Omega in self._last_elements:
                for k in range(len(t_ans)):
                    w.writerow([name,
                                f"{t_ans[k]:.6f}",
                                f"{e[k]:.6f}",
                                f"{a[k]:.6f}" if a is not None and not np.isnan(a[k]) else "",
                                f"{incl[k]:.4f}",
                                f"{omega[k]:.4f}",
                                f"{Omega[k]:.4f}"])


# ─────────────────────────────────────────────────────────
#  Onglet 6 — Résumé de simulation
# ─────────────────────────────────────────────────────────
class ResumeTab(QWidget):
    """ Vue de synthèse : toutes les métriques importantes de la simulation
    sur une seule page, avec des explications pour chaque valeur.
    """
    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        self.layout_main = QVBoxLayout(container)
        self.layout_main.setContentsMargins(16, 16, 16, 16)
        self.layout_main.setSpacing(12)

        self.lbl_title = QLabel("Chargez un fichier de simulation pour voir le résumé.")
        self.lbl_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_title.setStyleSheet(f"color:{ACCENT2}; font-weight:600;")
        self.layout_main.addWidget(self.lbl_title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(10)
        self.layout_main.addWidget(self.content_widget)
        self.layout_main.addStretch()

    def load_data(self, data: SimData):
        self.data = data
        self._rebuild()

    def _card(self, title: str, value: str, explication: str, good: bool = None):
        """Crée une carte métrique avec explication."""
        frame = QFrame()
        frame.setStyleSheet(f""" QFrame {{ background:{CARD_BG}; border-radius:8px; border:1px solid {BORDER}; }}
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)

        # Indicateur coloré
        dot = QLabel("●")
        dot_color = "#66BB6A" if good is True else ("#EF5350" if good is False else MUTED)
        dot.setStyleSheet(f"color:{dot_color}; font-size:18px;")
        dot.setFixedWidth(24)
        layout.addWidget(dot)

        # Titre + explication
        txt = QVBoxLayout()
        lbl_t = QLabel(title)
        lbl_t.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_ex = QLabel(explication)
        lbl_ex.setObjectName("muted")
        lbl_ex.setWordWrap(True)
        txt.addWidget(lbl_t)
        txt.addWidget(lbl_ex)
        layout.addLayout(txt)
        layout.addStretch()

        # Valeur
        lbl_v = QLabel(value)
        lbl_v.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_v.setStyleSheet(f"color:{MUTED};")
        lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_v.setMinimumWidth(140)
        layout.addWidget(lbl_v)
        return frame

    def _section(self, title: str):
        lbl = QLabel(title)
        lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl.setStyleSheet(f"color:{TEXT_COL}; border-bottom:1px solid {BORDER}; padding-bottom:4px;")
        return lbl

    def _rebuild(self):
        # Vider
        for i in reversed(range(self.content_layout.count())):
            w = self.content_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        d = self.data
        m = d.metrics
        duree = d.duree_ans()
        n_pts = len(d.t_jours)

        self.lbl_title.setText(f"Résumé — {os.path.basename(d.parquet_path)}")

        # ── Section Simulation ──
        self.content_layout.addWidget(self._section("Simulation"))

        self.content_layout.addWidget(self._card(
            "Durée simulée", f"{duree:.2f} ans",
            "Durée totale de la simulation en années terrestres."
        ))
        self.content_layout.addWidget(self._card(
            "Nombre de corps", f"{d.n_bodies}",
            "Nombre de corps massifs pris en compte dans le calcul gravitationnel."
        ))
        self.content_layout.addWidget(self._card(
            "Points enregistrés", f"{n_pts:,}",
            "Nombre de lignes dans le fichier. Plus il y en a, plus la résolution temporelle est fine."
        ))
        dt_moy = duree * 365.25 * 86400 / max(n_pts, 1)
        self.content_layout.addWidget(self._card(
            "Pas de temps moyen", f"{dt_moy:.1f} s",
            "Durée moyenne entre deux points enregistrés. Ce n'est pas le pas interne du simulateur."
        ))

        # ── Section Précision numérique ──
        self.content_layout.addWidget(self._section("Precision numerique"))

        derive_e = m.get("derive_e", float("nan"))
        good_e = abs(derive_e) < 0.01 if not math.isnan(derive_e) else None
        self.content_layout.addWidget(self._card(
            "Derive d'energie totale", f"{derive_e:.2e} %",
            "Ecart relatif de l'energie entre le debut et la fin. "
            "Vert si < 0.01% (tres bien). Rouge si > 1% (simulation peu fiable).",
            good=good_e
        ))

        derive_l = m.get("derive_l", float("nan"))
        good_l = abs(derive_l) < 0.01 if not math.isnan(derive_l) else None
        self.content_layout.addWidget(self._card(
            "Derive du moment cinetique", f"{derive_l:.2e} %",
            "Le moment cinetique devrait etre conserve. "
            "Une derive importante indique des erreurs d'integration.",
            good=good_l
        ))

        derive_em = m.get("derive_e_mean", float("nan"))
        self.content_layout.addWidget(self._card(
            "Derive energie moyenne / an", f"{derive_em:.2e} %/an",
            "Taux de derive par annee. Permet d'extrapoler la fiabilite sur de longues simulations."
        ))

        from core.periods import (
            compute_body_periods,
            format_period_seconds,
            periods_days_to_seconds,
            kepler_periods_from_dataframe,
        )

        # Périodes de rotation depuis settings.json (spin_rate)
        rotation_by_name = {}
        settings_path = os.path.join(APP_DIR, "settings.json")
        if os.path.isfile(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                G_cfg = float(cfg.get("physics", {}).get("G", 6.674e-11))
                for entry in compute_body_periods(cfg.get("bodies", []), G_cfg):
                    rotation_by_name[entry["name"]] = entry
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        kepler_sec = m.get("periods_kepler_sec")
        if not kepler_sec and d.df is not None and len(d.df) > 0:
            masses = {i: 1.989e30 if i == 0 else 5.972e24 for i in range(d.n_bodies)}
            try:
                import config as _cfg
                masses = {i: float(_cfg.BODY_MASSES[i])
                          for i in range(min(d.n_bodies, len(_cfg.BODY_MASSES)))}
                G_k = float(_cfg.G)
                ua_k = float(_cfg.UA)
            except Exception:
                G_k, ua_k = 6.674e-11, UA
            body_idx = list(range(d.n_bodies))
            kepler_sec = kepler_periods_from_dataframe(
                d.df, body_idx, masses, central_idx=0, G=G_k, ua_m=ua_k)

        periods_meas = m.get("periods", {})

        # ── Section Corps ──
        self.content_layout.addWidget(self._section("Corps simulés"))
        for i, name in enumerate(d.body_names):
            x, y, z = d.positions(i)
            r_min = float(np.sqrt(x**2 + y**2 + z**2).min())
            r_max = float(np.sqrt(x**2 + y**2 + z**2).max())

            if i > 0:
                dist = d.distance_pair(0, i)
                extra = (
                    f"  |  dist au corps central : "
                    f"{_fmt_distance_m(dist.min() * UA)} – {_fmt_distance_m(dist.max() * UA)}"
                )
            else:
                extra = "  (corps de référence)"

            self.content_layout.addWidget(self._card(
                name,
                f"r : {_fmt_distance_m(r_min * UA)} – {_fmt_distance_m(r_max * UA)}",
                f"Distance à l'origine (min–max).{extra}"
            ))

        # ── Section Périodes ──
        self.content_layout.addWidget(self._section("Periodes par corps"))
        central_name = d.body_names[0] if d.body_names else "corps 0"
        for i, name in enumerate(d.body_names):
            lines = []
            hints = []

            if i == 0:
                lines.append("Orbite : — (référence)")
            else:
                T_k = None
                if isinstance(kepler_sec, dict):
                    T_k = kepler_sec.get(i)
                if T_k is not None:
                    lines.append(
                        f"Orbite (Kepler, départ) : {format_period_seconds(T_k)}"
                    )
                    hints.append(
                        f"Tour complet autour de « {central_name} » "
                        "d'après l'état initial enregistré."
                    )
                pk = periods_meas.get(i, periods_meas.get(str(i)))
                T_m = periods_days_to_seconds(pk)
                if T_m is not None:
                    lines.append(
                        f"Orbite (mesurée) : {format_period_seconds(T_m)}"
                    )
                    hints.append(
                        "Période observée dans la simulation "
                        "(passages répétés du plan XY)."
                    )
                elif i > 0 and T_k is None:
                    lines.append("Orbite : indéterminée")

            rot = rotation_by_name.get(name, {})
            rot_lbl = rot.get("rotation_label", "—")
            lines.append(f"Rotation propre : {rot_lbl}")
            if rot.get("rotation_hint"):
                hints.append(rot["rotation_hint"])

            self.content_layout.addWidget(self._card(
                name,
                "\n".join(lines) if lines else "—",
                " ".join(hints) if hints else "Périodes orbitale et de rotation.",
            ))

        # ── Section Excentricités ──
        eccs_parent = m.get("eccentricities_parent", {})
        eccs_helio = m.get("eccentricities_helio", m.get("eccentricities", {}))
        parents = m.get("body_parents", {})
        if eccs_parent or eccs_helio:
            self.content_layout.addWidget(self._section("Excentricites"))
            for i, name in enumerate(d.body_names):
                e_par = eccs_parent.get(i, eccs_parent.get(str(i)))
                e_hel = eccs_helio.get(i, eccs_helio.get(str(i)))
                lines = []
                hints = []
                if e_par is not None and not (isinstance(e_par, float) and math.isnan(e_par)):
                    pidx = parents.get(i, parents.get(str(i), 0))
                    pname = d.body_names[int(pidx)] if int(pidx) < d.n_bodies else "parent"
                    lines.append(f"Autour de {pname} : {float(e_par):.4f}")
                    hints.append(
                        "Variation de la distance au parent (pertinent pour la Lune). "
                        "0 = orbite quasi circulaire."
                    )
                if e_hel is not None and not (isinstance(e_hel, float) and math.isnan(e_hel)):
                    lines.append(f"Vers le Soleil (rayon) : {float(e_hel):.4f}")
                    hints.append(
                        "Distance au centre (0) : inclut l'orbite du parent ; "
                        "souvent élevée pour un satellite même si son orbite locale est stable."
                    )
                if not lines:
                    continue
                ecc_for_good = None
                if e_par is not None:
                    try:
                        ep = float(e_par)
                        if not math.isnan(ep):
                            ecc_for_good = ep
                    except (TypeError, ValueError):
                        pass
                good_ecc = ecc_for_good < 0.1 if ecc_for_good is not None else None
                self.content_layout.addWidget(self._card(
                    name,
                    "\n".join(lines),
                    " ".join(hints),
                    good=good_ecc,
                ))


# ─────────────────────────────────────────────────────────
#  Onglet 4 — Simulation (éditeur visuel complet)
# ─────────────────────────────────────────────────────────

# Valeurs par défaut pour un nouveau corps
BODY_DEFAULTS = {
    "name": "Nouveau corps",
    "mass": 5.972e24,
    "pos": [UA, 0.0, 0.0],
    "use_auto_vel": True,
    "vel_manual": [0.0, 0.0, 0.0],
    "incl_deg": 0.0,
    "sens": 1,
    "j2r2": 0.0,
    "mdot": 0.0,
}

# Orbite type Lune / Terre par défaut pour un nouveau satellite
DEFAULT_SATELLITE_ORBIT_M = 384400e3
SATELLITE_BODY_DEFAULTS = {
    "mass": 7.34767309e22,
    "rayon": 1.7374e6,
    "use_auto_vel": True,
    "incl_deg": 0.0,
    "sens": 1,
    "j2r2": 0.0,
    "mdot": 0.0,
    "spin_rate": 0.0,
    "spin_axis": [0.0, 0.0, 1.0],
    "k2": 0.0,
    "inertia_factor": 0.4,
}

def _deep_merge_settings(base: dict, override: dict) -> dict:
    """Fusion profonde pour ne pas écraser des clés absentes du dashboard."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_settings(out[key], value)
        else:
            out[key] = value
    return out


TOOLTIPS = {
    "duree_ans":     "Durée totale de la simulation en années.",
    "save_every":    "Sauvegarder une ligne de données toutes les N frames.\nPlus petit = plus de précision mais fichier plus lourd.",
    "dt_max":        "Pas de temps maximum en secondes.\n1 jour = 86400 s, 1 heure = 3600 s.",
    "dt_min":        "Pas de temps minimum en secondes (précision max).",
    "softening":     "Adoucissement gravitationnel en mètres.\nÉvite les divergences quand deux corps sont très proches.",
    "dist_seuil_m": "Distance (m) sous laquelle le pas de temps est réduit.\n"
                    "Pour la Lune (~3,84×10⁸ m), mettre un seuil < cette distance (ex. 2×10⁸ m).",
    "alpha":         "Agressivité de l'adaptation du pas de temps.\n0.01 = très prudent, 0.5 = agressif.",
    "mass":          "Masse du corps en kg.\nSoleil = 1.989e30, Terre = 5.972e24, Mars = 6.418e23.",
    "pos_x":         "Position X en mètres (référentiel absolu, ou relatif au parent si satellite).",
    "pos_rel":       "Position du satellite par rapport au corps parent (mètres).\n"
                     "Ex. Lune : X ≈ 3,844×10⁸ m depuis la Terre.",
    "incl_deg":      "Inclinaison orbitale en degrés.\n"
                     "Satellite : inclinaison par rapport au plan orbital du parent "
                     "(ex. plan Terre–Soleil pour la Lune).\n"
                     "Autres corps : inclinaison par rapport au plan XY global.\n"
                     "Avec vitesse auto : tangent 3D correct autour du parent.",
    "use_auto_vel":  "Calcule automatiquement la vitesse circulaire\npour une orbite stable autour du corps central.",
    "sens":               "Sens de rotation : +1 = prograde, -1 = rétrograde.",
    "checkpoint_every_ans": "Fréquence de sauvegarde des checkpoints en années simulées.\n"
                            "Ex : 10 = un checkpoint tous les 10 ans.\n"
                            "Permet de reprendre la simulation après une interruption.",
    "monitor_every":      "Calculer l'énergie et le moment cinétique toutes les N frames.\n"
                          "Plus petit = suivi plus fin de la dérive d'énergie, mais plus lent.\n"
                          "Doit être un multiple de save_every pour cohérence optimale.",
    "realtime_every":     "Rafraîchir le viewer temps réel toutes les N frames.\n"
                          "50 = mise à jour toutes les 50 itérations.\n"
                          "Réduire si l'animation est saccadée, augmenter pour accélérer le calcul.",
    "rayon_collision":    "Paramètre legacy — remplacé par le rayon physique de chaque corps.\n"
                          "La collision est maintenant détectée si dist < rayon_i + rayon_j.",
    "rayon":              "Rayon physique du corps (mètres). Utilisé pour la détection de collision.\n"
                          "Exemples : Soleil 6.957e8 m, Terre 6.371e6 m, Mars 3.389e6 m.",
    "output_dir":         "Dossier de sortie pour les fichiers .parquet et métriques.\n"
                          "Chemin relatif au script, ou absolu.",
    "output_name":        "Préfixe du nom de fichier Parquet (ex: orbite_3_corps).\n"
                          "Le fichier sera : nom_AAAA-MM-JJ_HH-MM-SS.parquet\n"
                          "Laissez vide pour simulation_AAAA-MM-JJ_HH-MM-SS.parquet",
    "checkpoint_dir":     "Sous-dossier (dans output_dir) pour les fichiers de reprise.\n"
                          "Chemin relatif au dossier de sortie, ou absolu.",
    "num_threads":        "Nombre de threads Numba pour le calcul parallèle des forces.\n"
                          "0 ou « Auto » = réglage automatique (laisse 1–2 cœurs au système).\n"
                          "1 = séquentiel. Valeur max = nombre de cœurs de la machine.",
    "flush_every":        "Nombre de lignes de données gardées en RAM avant écriture sur disque.\n"
                          "Plus petit = moins de RAM, plus d'écritures disque.\n"
                          "50000 est un bon compromis pour les longues simulations.",
    "web_port":           "Port du serveur web Three.js (--web).\nDéfaut : 5050.",
    "G_const":            "Constante gravitationnelle G en m³·kg⁻¹·s⁻².\n"
                          "Valeur CODATA : 6.674×10⁻¹¹.",
    "UA_const":           "Unité astronomique en mètres.\nDéfaut : 1.496×10¹¹ m.",
    "c_light":            "Vitesse de la lumière en m/s (utilisée pour les corrections PN1).\n"
                          "Valeur physique exacte : 299 792 458 m/s.\n"
                          "Ne modifier que pour des expériences numériques.",
    "j2r2":               "Coefficient d'aplatissement J2 × rayon² du corps (m²).\n"
                          "Modélise l'effet du renflement équatorial sur l'orbite.\n"
                          "0 = sphère parfaite. Terre : J2=1.083e-3, R=6.371e6 m → J2R2 ≈ 4.4e10.",
    "mdot":               "Amplitude de dM/dt en kg/s (valeur toujours positive ici).\n"
                          "Perte → mdot négatif dans le JSON. Gain → mdot positif.\n"
                          "0 = masse constante. Soleil (vent) : ~2e9 kg/s en perte.",
    "pos_y":              "Position initiale Y en mètres (plan orbital, perpendiculaire à X).",
    "pos_z":              "Position initiale Z en mètres (hors du plan orbital).",
    "vel_x":              "Vitesse initiale selon X en m/s.",
    "vel_z":              "Vitesse initiale selon Z en m/s.",
}


class BodyWidget(QGroupBox):
    """Panneau d'édition pour un seul corps."""

    removed = pyqtSignal(object)  # self

    def __init__(self, index: int, data: dict = None, period_host=None):
        super().__init__()
        self.index = index
        self._period_host = period_host
        self._parent_index = None
        self._build(data or dict(BODY_DEFAULTS))

    def _build(self, d: dict):
        pi = d.get("parent_index")
        self._parent_index = int(pi) if pi is not None and int(pi) >= 0 else None
        if self._parent_index is not None and self._parent_index >= self.index:
            self._parent_index = None
        self._apply_satellite_style()
        self._update_title(d.get("name", f"Corps {self.index}"))
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        fl = QFormLayout()
        fl.setLabelAlignment(Qt.AlignRight)
        fl.setSpacing(5)

        # Nom
        self.le_name = QLineEdit(str(d.get("name", f"Corps {self.index}")))
        self.le_name.textChanged.connect(self._on_name_changed)
        fl.addRow("Nom :", self.le_name)

        # Masse
        self.sp_mass = QDoubleSpinBox()
        self.sp_mass.setRange(1.0, 1e35)
        self.sp_mass.setDecimals(3)
        self.sp_mass.setValue(float(d.get("mass", 5.972e24)))
        self.sp_mass.setSuffix(" kg")
        self.sp_mass.setToolTip(TOOLTIPS["mass"])
        # Notation scientifique
        self.sp_mass.setSingleStep(1e23)
        fl.addRow("Masse :", self.sp_mass)

        # Raccourcis masses
        shortcuts = QHBoxLayout()
        for label, val in [("Soleil", 1.989e30), ("Terre", 5.972e24),
                           ("Mars", 6.418e23), ("Jupiter", 1.898e27)]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.setStyleSheet("font-size:10px; padding:1px 4px;")
            btn.clicked.connect(lambda _, v=val: self.sp_mass.setValue(v))
            shortcuts.addWidget(btn)
        fl.addRow("", shortcuts)

        # Position : relative au parent si satellite, sinon absolue (m)
        pos_ui = self._initial_ui_position(d)
        self._lbl_pos_x = QLabel()
        self._lbl_pos_y = QLabel()
        self._lbl_pos_z = QLabel()
        self.sp_pos_x = QDoubleSpinBox()
        self.sp_pos_x.setRange(_POS_SPIN_MIN, _POS_SPIN_MAX)
        self.sp_pos_x.setDecimals(0)
        self.sp_pos_x.setValue(float(pos_ui[0]))
        self.sp_pos_x.setSuffix(" m")
        self.sp_pos_y = QDoubleSpinBox()
        self.sp_pos_y.setRange(_POS_SPIN_MIN, _POS_SPIN_MAX)
        self.sp_pos_y.setDecimals(0)
        self.sp_pos_y.setValue(float(pos_ui[1]))
        self.sp_pos_y.setSuffix(" m")
        self.sp_pos_z = QDoubleSpinBox()
        self.sp_pos_z.setRange(_POS_SPIN_MIN, _POS_SPIN_MAX)
        self.sp_pos_z.setDecimals(0)
        self.sp_pos_z.setValue(float(pos_ui[2]))
        self.sp_pos_z.setSuffix(" m")
        self._update_position_labels()
        pos_tip = (TOOLTIPS["pos_rel"] if self._parent_index is not None
                   else TOOLTIPS["pos_x"])
        self.sp_pos_x.setToolTip(pos_tip)
        self.sp_pos_y.setToolTip(pos_tip)
        self.sp_pos_z.setToolTip(pos_tip)
        fl.addRow(self._lbl_pos_x, self.sp_pos_x)
        fl.addRow(self._lbl_pos_y, self.sp_pos_y)
        fl.addRow(self._lbl_pos_z, self.sp_pos_z)
        self.lbl_pos_abs = QLabel("")
        self.lbl_pos_abs.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        self.lbl_pos_abs.setWordWrap(True)
        fl.addRow("", self.lbl_pos_abs)
        for sp in (self.sp_pos_x, self.sp_pos_y, self.sp_pos_z):
            sp.valueChanged.connect(self._on_position_ui_changed)

        # Inclinaison
        self.sp_incl = QDoubleSpinBox()
        self.sp_incl.setRange(-180.0, 180.0)
        self.sp_incl.setDecimals(2)
        self.sp_incl.setValue(float(d.get("incl_deg", 0.0)))
        self.sp_incl.setSuffix(" °")
        self.sp_incl.setToolTip(TOOLTIPS["incl_deg"])
        fl.addRow("Inclinaison :", self.sp_incl)

        # Sens
        self.cb_sens = QComboBox()
        self.cb_sens.addItems(["Prograde (+1)", "Retrograde (-1)"])
        self.cb_sens.setCurrentIndex(0 if int(d.get("sens", 1)) >= 0 else 1)
        self.cb_sens.setToolTip(TOOLTIPS["sens"])
        fl.addRow("Sens :", self.cb_sens)

        # Vitesse auto
        self.chk_auto = QCheckBox("Vitesse circulaire automatique")
        self.chk_auto.setChecked(bool(d.get("use_auto_vel", True)))
        tip_auto = TOOLTIPS["use_auto_vel"]
        if self._parent_index is not None:
            tip_auto += (
                "\n\nSatellite : la vitesse circulaire est calculée "
                f"autour de « {self._parent_display_name()} ».")
        self.chk_auto.setToolTip(tip_auto)
        self.chk_auto.stateChanged.connect(self._toggle_vel)
        fl.addRow("", self.chk_auto)

        # Vitesse manuelle (masquée si auto)
        vel = d.get("vel_manual", d.get("vel", [0.0, 0.0, 0.0]))

        self.sp_vx = QDoubleSpinBox()
        self.sp_vx.setRange(-1e8, 1e8); self.sp_vx.setDecimals(2)
        self.sp_vx.setValue(float(vel[0]) if len(vel) > 0 else 0.0)
        self.sp_vx.setSuffix(" m/s")
        self.sp_vx.setToolTip(TOOLTIPS.get("vel_x", "Vitesse initiale selon X en m/s"))
        self.lbl_vx = QLabel("Vitesse Vx :")
        fl.addRow(self.lbl_vx, self.sp_vx)

        self.sp_vy = QDoubleSpinBox()
        self.sp_vy.setRange(-1e8, 1e8); self.sp_vy.setDecimals(2)
        self.sp_vy.setValue(float(vel[1]) if len(vel) > 1 else 0.0)
        self.sp_vy.setSuffix(" m/s")
        self.lbl_vy = QLabel("Vitesse Vy :")
        fl.addRow(self.lbl_vy, self.sp_vy)

        self.sp_vz = QDoubleSpinBox()
        self.sp_vz.setRange(-1e8, 1e8); self.sp_vz.setDecimals(2)
        self.sp_vz.setValue(float(vel[2]) if len(vel) > 2 else 0.0)
        self.sp_vz.setSuffix(" m/s")
        self.sp_vz.setToolTip(TOOLTIPS.get("vel_z", "Vitesse initiale selon Z en m/s"))
        self.lbl_vz = QLabel("Vitesse Vz :")
        fl.addRow(self.lbl_vz, self.sp_vz)

        # ── Paramètres avancés par corps ─────────────────────────────────────
        fl.addRow(QLabel(""))  # séparateur visuel

        self.sp_j2r2 = QDoubleSpinBox()
        self.sp_j2r2.setRange(0.0, 1e18); self.sp_j2r2.setDecimals(3)
        self.sp_j2r2.setSingleStep(1e8)
        self.sp_j2r2.setValue(float(d.get("j2r2", 0.0)))
        self.sp_j2r2.setSuffix(" m²")
        self.sp_j2r2.setToolTip(TOOLTIPS.get("j2r2",
            "Coefficient J2 × rayon² (m²). 0 = sphère parfaite.\n"
            "Terre ≈ 4.4e10  |  Jupiter ≈ 1.5e15"))
        fl.addRow("J2·R² (aplatiss.) :", self.sp_j2r2)

        self.sp_mdot = QDoubleSpinBox()
        self.sp_mdot.setRange(0.0, 1e15); self.sp_mdot.setDecimals(3)
        self.sp_mdot.setSingleStep(1e8)
        mdot_raw = float(d.get("mdot", 0.0))
        if "mdot_loss" in d:
            mdot_loss = bool(d["mdot_loss"])
            mdot_mag = abs(mdot_raw)
        elif mdot_raw < 0.0:
            mdot_loss = True
            mdot_mag = -mdot_raw
        elif mdot_raw > 0.0:
            mdot_loss = True
            mdot_mag = mdot_raw
        else:
            mdot_loss = True
            mdot_mag = 0.0
        self.sp_mdot.setValue(mdot_mag)
        self.sp_mdot.setSuffix(" kg/s")
        self.sp_mdot.setToolTip(TOOLTIPS["mdot"])
        fl.addRow("|ṁ| (kg/s) :", self.sp_mdot)

        mdot_mode_row = QHBoxLayout()
        self.chk_mdot_loss = QCheckBox("Perte de masse")
        self.chk_mdot_gain = QCheckBox("Gain de masse")
        self.chk_mdot_loss.setToolTip(
            "dM/dt < 0 : la masse diminue (vent stellaire, évaporation…).")
        self.chk_mdot_gain.setToolTip(
            "dM/dt > 0 : la masse augmente (accrétion, comète…).")
        self.chk_mdot_loss.setChecked(mdot_loss)
        self.chk_mdot_gain.setChecked(not mdot_loss and mdot_mag > 0.0)
        self.chk_mdot_loss.stateChanged.connect(self._on_mdot_loss_toggled)
        self.chk_mdot_gain.stateChanged.connect(self._on_mdot_gain_toggled)
        mdot_mode_row.addWidget(self.chk_mdot_loss)
        mdot_mode_row.addWidget(self.chk_mdot_gain)
        mdot_mode_row.addStretch()
        mdot_mode_w = QWidget()
        mdot_mode_w.setLayout(mdot_mode_row)
        fl.addRow("", mdot_mode_w)

        # ── Rayon physique ────────────────────────────────────────────────────
        self.sp_rayon = QDoubleSpinBox()
        self.sp_rayon.setRange(1.0, 1e12)
        self.sp_rayon.setDecimals(0)
        self.sp_rayon.setSingleStep(1e5)
        self.sp_rayon.setValue(float(d.get("rayon", 1e3)))
        self.sp_rayon.setSuffix(" m")
        self.sp_rayon.setToolTip(
            "Rayon physique du corps (mètres).\n"
            "Utilisé pour la détection de collision : collision si dist < rayon_i + rayon_j.\n"
            "Exemples :\n"
            "  Soleil   : 6.957e8 m  (696 700 km)\n"
            "  Terre    : 6.371e6 m  (6 371 km)\n"
            "  Mars     : 3.389e6 m  (3 389 km)\n"
            "  Jupiter  : 7.149e7 m  (71 490 km)\n"
            "  Astéroïde: ~500e3 m   (500 km)")
        fl.addRow("Rayon physique :", self.sp_rayon)

        # ── Rotation propre (spin) ────────────────────────────────────────────
        fl.addRow(QLabel(""))
        sep_spin = QLabel("── Rotation propre ──")
        sep_spin.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        fl.addRow(sep_spin)

        self.sp_spin_rate = QDoubleSpinBox()
        self.sp_spin_rate.setRange(0.0, 1e4)
        self.sp_spin_rate.setDecimals(6)
        self.sp_spin_rate.setValue(float(d.get("spin_rate", 0.0)))
        self.sp_spin_rate.setSuffix(" rad/s")
        self.sp_spin_rate.setSingleStep(1e-5)
        self.sp_spin_rate.setToolTip(
            "Vitesse angulaire propre ω du corps (rad/s).\n"
            "0 = pas de rotation. Exemples :\n"
            "  Terre    : 7.27×10⁻⁵ rad/s  (1 tour/24 h)\n"
            "  Jupiter  : 1.76×10⁻⁴ rad/s  (1 tour/10 h)\n"
            "  Soleil   : 2.87×10⁻⁶ rad/s  (1 tour/25 j)\n"
            "  Mercure  : 1.24×10⁻⁶ rad/s  (1 tour/59 j)\n"
            "  Vénus    : −2.99×10⁻⁷ rad/s (rétrograde)\n\n"
            "Effet physique : définit la norme de S = I·ω·ŝ."
        )
        fl.addRow("Vitesse spin ω :", self.sp_spin_rate)

        spin_axis = d.get("spin_axis", [0.0, 0.0, 1.0])
        self.sp_spin_ax = QDoubleSpinBox()
        self.sp_spin_ax.setRange(-1.0, 1.0); self.sp_spin_ax.setDecimals(4)
        self.sp_spin_ax.setValue(float(spin_axis[0]) if len(spin_axis) > 0 else 0.0)
        self.sp_spin_ax.setToolTip(
            "Composante X de l'axe de rotation ŝ (normalisé automatiquement).\n"
            "Ex: ŝ = (0,0,1) = axe Z (plan orbital).")
        self.sp_spin_ay = QDoubleSpinBox()
        self.sp_spin_ay.setRange(-1.0, 1.0); self.sp_spin_ay.setDecimals(4)
        self.sp_spin_ay.setValue(float(spin_axis[1]) if len(spin_axis) > 1 else 0.0)
        self.sp_spin_ay.setToolTip("Composante Y de l'axe de rotation ŝ.")
        self.sp_spin_az = QDoubleSpinBox()
        self.sp_spin_az.setRange(-1.0, 1.0); self.sp_spin_az.setDecimals(4)
        self.sp_spin_az.setValue(float(spin_axis[2]) if len(spin_axis) > 2 else 1.0)
        self.sp_spin_az.setToolTip(
            "Composante Z de l'axe de rotation ŝ.\n"
            "Exemples d'obliquité :\n"
            "  Terre    : ŝ ≈ (sin(23.4°), 0, cos(23.4°)) = (0.397, 0, 0.918)\n"
            "  Uranus   : ŝ ≈ (0.998, 0, 0.063)  — couché sur le côté\n"
            "  Vénus    : ŝ ≈ (0, 0, -1)  — rotation rétrograde")

        axis_row = QHBoxLayout()
        for lbl, sp in [("X", self.sp_spin_ax), ("Y", self.sp_spin_ay), ("Z", self.sp_spin_az)]:
            axis_row.addWidget(QLabel(lbl))
            axis_row.addWidget(sp)
        axis_widget = QWidget(); axis_widget.setLayout(axis_row)
        fl.addRow("Axe spin ŝ :", axis_widget)

        self.sp_k2 = QDoubleSpinBox()
        self.sp_k2.setRange(0.0, 2.0); self.sp_k2.setDecimals(4)
        self.sp_k2.setSingleStep(0.05)
        self.sp_k2.setValue(float(d.get("k2", 0.0)))
        self.sp_k2.setToolTip(
            "Nombre de Love de degré 2 — mesure la déformabilité du corps.\n"
            "0     = corps rigide, pas de couplage spin-orbite.\n"
            "0.03  = planète rocheuse dense (Mercure)\n"
            "0.17  = Mars\n"
            "0.30  = Terre\n"
            "0.49  = Jupiter (géante gazeuse)\n"
            "0.75  = étoile très déformable\n\n"
            "Avec k2>0 et j2r2>0, l'axe de spin précesse sous les couples\n"
            "de marée exercés par les autres corps.\n"
            "Ex: précession des équinoxes terrestres ≈ 26 000 ans.")
        fl.addRow("k₂ (Love) :", self.sp_k2)

        self.sp_inertia = QDoubleSpinBox()
        self.sp_inertia.setRange(0.1, 0.9); self.sp_inertia.setDecimals(3)
        self.sp_inertia.setSingleStep(0.01)
        self.sp_inertia.setValue(float(d.get("inertia_factor", 0.4)))
        self.sp_inertia.setToolTip(
            "Facteur d'inertie α tel que I = α·M·R².\n"
            "Définit comment la masse est distribuée dans le corps.\n"
            "0.4  = sphère homogène (défaut)\n"
            "0.33 = Terre (noyau dense)\n"
            "0.26 = Jupiter (très concentré au centre)\n"
            "0.5  = coque creuse")
        fl.addRow("Facteur inertie α :", self.sp_inertia)

        fl.addRow(QLabel(""))
        sep_per = QLabel("── Périodes estimées ──")
        sep_per.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        fl.addRow(sep_per)

        self.lbl_period_orbit = QLabel("—")
        self.lbl_period_orbit.setWordWrap(True)
        self.lbl_period_orbit.setStyleSheet(f"color:#c8cdd8; font-weight:bold;")
        self.lbl_period_orbit.setToolTip(
            "Temps pour un tour complet autour du corps 0 (Kepler), "
            "à partir de la position et de la vitesse initiales.")
        fl.addRow("Période orbitale :", self.lbl_period_orbit)

        self.lbl_period_spin = QLabel("—")
        self.lbl_period_spin.setWordWrap(True)
        self.lbl_period_spin.setStyleSheet(f"color:#c8cdd8; font-weight:bold;")
        self.lbl_period_spin.setToolTip("Temps pour un tour sur lui-même : T = 2π / |ω|.")
        fl.addRow("Période rotation :", self.lbl_period_spin)

        layout.addLayout(fl)
        self._bind_period_refresh()

        btn_add_sat = QPushButton("+ Ajouter un satellite")
        btn_add_sat.setToolTip(
            "Ajoute un corps satellite en orbite autour de ce corps "
            "(même panneau de paramètres, orbite autour du parent).")
        btn_add_sat.clicked.connect(self._on_add_satellite_clicked)
        layout.addWidget(btn_add_sat)

        # Bouton supprimer (sauf corps 0)
        if self.index > 0:
            btn_del = QPushButton("Supprimer ce corps")
            btn_del.setObjectName("danger")
            btn_del.setFixedHeight(24)
            btn_del.clicked.connect(lambda: self.removed.emit(self))
            layout.addWidget(btn_del)

        self._toggle_vel(self.chk_auto.checkState())
        self._update_abs_pos_hint()

    def _apply_satellite_style(self):
        if self._parent_index is not None:
            self.setStyleSheet(
                "QGroupBox { margin-left: 20px; border: 1px solid #3a5080; "
                "border-radius: 6px; padding-top: 14px; }"
                "QGroupBox::title { color: #8ab4f8; }"
            )
        else:
            self.setStyleSheet("")

    def _parent_display_name(self) -> str:
        if self._parent_index is None or self._period_host is None:
            return ""
        for w in self._period_host._body_widgets:
            if w.index == self._parent_index:
                return w.le_name.text().strip() or f"Corps {self._parent_index}"
        return f"Corps {self._parent_index}"

    def _on_add_satellite_clicked(self, checked=False):
        if self._period_host is not None:
            self._period_host._add_satellite(self)

    def _initial_ui_position(self, d: dict) -> list:
        """Coordonnées affichées dans les champs (relatives si satellite)."""
        if self._parent_index is not None and self._period_host is not None:
            if "pos_rel" in d:
                pr = d["pos_rel"]
                return [float(pr[0]), float(pr[1]) if len(pr) > 1 else 0.0,
                        float(pr[2]) if len(pr) > 2 else 0.0]
            abs_pos = d.get("pos", [0.0, 0.0, 0.0])
            ppos = self._period_host.absolute_pos_for_index(self._parent_index)
            return [float(abs_pos[k]) - float(ppos[k]) for k in range(3)]
        pos = d.get("pos", [UA, 0.0, 0.0])
        return [float(pos[0]) if len(pos) > 0 else 0.0,
                float(pos[1]) if len(pos) > 1 else 0.0,
                float(pos[2]) if len(pos) > 2 else 0.0]

    def _ui_position(self) -> list:
        return [self.sp_pos_x.value(), self.sp_pos_y.value(), self.sp_pos_z.value()]

    def absolute_position(self) -> list:
        """Position absolue (m) pour la simulation / JSON pos."""
        rel = self._ui_position()
        if self._parent_index is None or self._period_host is None:
            return rel
        ppos = self._period_host.absolute_pos_for_index(self._parent_index)
        return [rel[k] + ppos[k] for k in range(3)]

    def _update_position_labels(self):
        pname = self._parent_display_name()
        if self._parent_index is not None:
            self._lbl_pos_x.setText(f"X rel. {pname} :")
            self._lbl_pos_y.setText(f"Y rel. {pname} :")
            self._lbl_pos_z.setText(f"Z rel. {pname} :")
        else:
            self._lbl_pos_x.setText("Position X :")
            self._lbl_pos_y.setText("Position Y :")
            self._lbl_pos_z.setText("Position Z :")

    def _update_abs_pos_hint(self):
        if self._parent_index is None:
            self.lbl_pos_abs.setText("")
            return
        ap = self.absolute_position()
        self.lbl_pos_abs.setText(
            f"Position absolue : ({_fmt_distance_m(ap[0])}, "
            f"{_fmt_distance_m(ap[1])}, {_fmt_distance_m(ap[2])})"
        )

    def _on_position_ui_changed(self, *_):
        self._update_abs_pos_hint()
        self._request_period_refresh()
        if self._period_host is not None and self._parent_index is None:
            self._period_host.refresh_satellite_position_hints()

    def _on_name_changed(self, text: str):
        self._update_title(text)
        if self._period_host is None:
            return
        for w in self._period_host._body_widgets:
            if w._parent_index == self.index:
                w._update_title(w.le_name.text())
                w._update_position_labels()
                w._update_abs_pos_hint()

    def _update_title(self, name: str):
        label = name.strip() if name.strip() else f"Corps {self.index}"
        if self._parent_index is not None:
            self.setTitle(f"🛰 Satellite de {self._parent_display_name()} — {label}")
        else:
            self.setTitle(f"Corps {self.index} — {label}")

    def _toggle_vel(self, state):
        auto = (state == Qt.Checked)
        for w in (self.sp_vx, self.lbl_vx,
                  self.sp_vy, self.lbl_vy,
                  self.sp_vz, self.lbl_vz):
            w.setVisible(not auto)
        self._request_period_refresh()

    def _on_mdot_loss_toggled(self, state):
        if state == Qt.Checked:
            self.chk_mdot_gain.blockSignals(True)
            self.chk_mdot_gain.setChecked(False)
            self.chk_mdot_gain.blockSignals(False)
        elif not self.chk_mdot_gain.isChecked():
            self.chk_mdot_gain.setChecked(True)

    def _on_mdot_gain_toggled(self, state):
        if state == Qt.Checked:
            self.chk_mdot_loss.blockSignals(True)
            self.chk_mdot_loss.setChecked(False)
            self.chk_mdot_loss.blockSignals(False)
        elif not self.chk_mdot_loss.isChecked():
            self.chk_mdot_loss.setChecked(True)

    def _bind_period_refresh(self):
        """Recalcule les périodes affichées quand un paramètre change."""
        widgets = (
            self.le_name,
            self.sp_mass, self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_incl, self.cb_sens, self.chk_auto,
            self.sp_vx, self.sp_vy, self.sp_vz,
            self.sp_spin_rate,
        )
        for w in widgets:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._request_period_refresh)
            elif hasattr(w, "textChanged"):
                w.textChanged.connect(self._request_period_refresh)
            elif hasattr(w, "stateChanged"):
                w.stateChanged.connect(self._request_period_refresh)
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self._request_period_refresh)

    def _request_period_refresh(self, *_):
        if self._period_host is not None:
            self._period_host._update_all_body_periods()

    def set_period_labels(self, orbital: str, rotation: str,
                          orbital_hint: str = "", rotation_hint: str = ""):
        self.lbl_period_orbit.setText(orbital)
        self.lbl_period_spin.setText(rotation)
        if orbital_hint:
            self.lbl_period_orbit.setToolTip(orbital_hint)
        if rotation_hint:
            self.lbl_period_spin.setToolTip(rotation_hint)

    def to_dict(self) -> dict:
        sens = 1 if self.cb_sens.currentIndex() == 0 else -1
        mdot_mag = self.sp_mdot.value()
        mdot_loss = self.chk_mdot_loss.isChecked()
        if not mdot_loss and not self.chk_mdot_gain.isChecked():
            mdot_loss = True
        if mdot_mag == 0.0:
            mdot_signed = 0.0
        elif mdot_loss:
            mdot_signed = -mdot_mag
        else:
            mdot_signed = mdot_mag
        abs_pos = self.absolute_position()
        rel_pos = self._ui_position()
        d = {
            "name":          self.le_name.text(),
            "mass":          self.sp_mass.value(),
            "pos":           abs_pos,
            "use_auto_vel":  self.chk_auto.isChecked(),
            "vel_manual":    [self.sp_vx.value(),
                              self.sp_vy.value(),
                              self.sp_vz.value()],
            "incl_deg":      self.sp_incl.value(),
            "sens":          sens,
            "j2r2":          self.sp_j2r2.value(),
            "mdot":          mdot_signed,
            "mdot_loss":     mdot_loss,
            "spin_rate":     self.sp_spin_rate.value(),
            "spin_axis":     [self.sp_spin_ax.value(),
                              self.sp_spin_ay.value(),
                              self.sp_spin_az.value()],
            "k2":            self.sp_k2.value(),
            "inertia_factor": self.sp_inertia.value(),
            "rayon":          self.sp_rayon.value(),
        }
        if self._parent_index is not None:
            d["parent_index"] = self._parent_index
            d["pos_rel"] = rel_pos
        return d


class SimulationTab(QWidget):

    def __init__(self):
        super().__init__()
        self.worker: SimulationWorker = None
        self._body_widgets: list[BodyWidget] = []
        self._progress_file = ""
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._poll_sim_progress)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Script row ───────────────────────────────────
        script_row = QHBoxLayout()
        self.le_script = QLineEdit()
        self.le_script.setPlaceholderText("Chemin vers moteur_astralis.py  (obligatoire)")
        btn_browse = QPushButton("Parcourir…")
        btn_browse.clicked.connect(self._browse_script)
        self.le_cwd = QLineEdit()
        self.le_cwd.setPlaceholderText("Dossier de travail (auto)")
        self.le_cwd.setMaximumWidth(220)
        script_row.addWidget(QLabel("Script :"))
        script_row.addWidget(self.le_script)
        script_row.addWidget(btn_browse)
        script_row.addSpacing(12)
        script_row.addWidget(QLabel("Dossier :"))
        script_row.addWidget(self.le_cwd)
        root.addLayout(script_row)

        _default_script = _default_moteur_script()
        if _moteur_script_ok(_default_script):
            self.le_script.setText(_default_script)
            self.le_cwd.setText(APP_DIR)
        self.le_script.textChanged.connect(lambda _: self._update_runtime_info())
        self.le_cwd.textChanged.connect(lambda _: self._update_runtime_info())

        # ── Barre presets + chargement ──────────────────
        preset_row = QHBoxLayout()
        lbl_preset = QLabel("Scenario rapide :")
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("-- Choisir un preset --")
        self.cb_preset.addItems([
            "Soleil - Terre - Mars",
            "Soleil - Terre - Lune",
            "Soleil - Terre - Mars - Jupiter",
            "Etoile binaire",
            "Systeme compact 3 corps",
        ])
        self.cb_preset.currentTextChanged.connect(
            lambda t: self._apply_preset(t) if not t.startswith("--") else None)
        self.cb_saved_configs = QComboBox()
        self.cb_saved_configs.setMinimumWidth(200)
        self.cb_saved_configs.currentIndexChanged.connect(self._on_saved_config_selected)
        btn_load_config = QPushButton("Charger une config…")
        btn_load_config.clicked.connect(self._load_settings_file)
        btn_save_config = QPushButton("Sauvegarder config…")
        btn_save_config.clicked.connect(self._save_settings_preset)
        preset_row.addWidget(lbl_preset)
        preset_row.addWidget(self.cb_preset)
        preset_row.addSpacing(12)
        preset_row.addWidget(QLabel("Mes configs :"))
        preset_row.addWidget(self.cb_saved_configs)
        preset_row.addWidget(btn_load_config)
        preset_row.addWidget(btn_save_config)
        preset_row.addStretch()
        root.addLayout(preset_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); root.addWidget(sep)

        # ── Zone principale : sous-onglets + console ─────
        main_split = QHBoxLayout()
        root.addLayout(main_split)

        # Sous-onglets paramètres
        self.subtabs = QTabWidget()
        self.subtabs.setMinimumWidth(420)
        main_split.addWidget(self.subtabs)

        def spin(mn, mx, val, dec=2, suffix="", tip="", step=None):
            s = QDoubleSpinBox()
            s.setRange(mn, mx); s.setDecimals(dec); s.setValue(val)
            if suffix: s.setSuffix(f" {suffix}")
            if tip: s.setToolTip(tip)
            if step: s.setSingleStep(step)
            return s

        def row_form(layout, label, widget, tip=""):
            lbl = QLabel(label)
            if tip:
                lbl.setToolTip(tip)
            layout.addRow(lbl, widget)

        def hint(layout, text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            layout.addRow("", lbl)

        # ── Onglet Général ───────────────────────────────────────────────
        tab_general = QWidget()
        fl_gen = QFormLayout(tab_general)
        fl_gen.setSpacing(10)
        fl_gen.setContentsMargins(12, 12, 12, 12)

        self.sp_duree = spin(0.001, 1e9, 100.0, 3, "ans", TOOLTIPS["duree_ans"])
        self.sp_save = QSpinBox()
        self.sp_save.setRange(1, 100000)
        self.sp_save.setValue(100)
        self.sp_save.setToolTip(TOOLTIPS["save_every"])
        self.sp_monitor = QSpinBox()
        self.sp_monitor.setRange(1, 100000)
        self.sp_monitor.setValue(100)
        self.sp_monitor.setToolTip(TOOLTIPS["monitor_every"])
        self.sp_flush = QSpinBox()
        self.sp_flush.setRange(1000, 500000)
        self.sp_flush.setValue(50000)
        self.sp_flush.setSingleStep(5000)
        self.sp_flush.setToolTip(TOOLTIPS["flush_every"])
        self.le_output_dir = QLineEdit("outputs")
        self.le_output_dir.setToolTip(TOOLTIPS["output_dir"])
        self.le_output_name = QLineEdit()
        self.le_output_name.setPlaceholderText("ex: orbite_3_corps (vide = nom automatique)")
        self.le_output_name.setToolTip(TOOLTIPS["output_name"])

        row_form(fl_gen, "Durée totale :", self.sp_duree, TOOLTIPS["duree_ans"])
        hint(fl_gen, "  Durée simulée en années.")
        row_form(fl_gen, "Sauvegarde (frames) :", self.sp_save, TOOLTIPS["save_every"])
        hint(fl_gen, "  Une ligne Parquet toutes les N frames.")
        row_form(fl_gen, "Monitor (frames) :", self.sp_monitor, TOOLTIPS["monitor_every"])
        hint(fl_gen, "  Calcul énergie / moment cinétique toutes les N frames.")
        row_form(fl_gen, "Flush RAM (lignes) :", self.sp_flush, TOOLTIPS["flush_every"])
        hint(fl_gen, "  Écriture disque par morceaux pour limiter la RAM.")
        fl_gen.addRow(self._sep_label("Dossiers et sauvegarde"))
        row_form(fl_gen, "Nom de sauvegarde :", self.le_output_name, TOOLTIPS["output_name"])
        hint(fl_gen, "  Ex: ma_sim → outputs/ma_sim_2026-05-21_12-00-00.parquet")
        row_form(fl_gen, "Dossier sorties :", self.le_output_dir, TOOLTIPS["output_dir"])
        hint(fl_gen, "  Dossier parent des fichiers .parquet.")
        self.subtabs.addTab(tab_general, "Général")

        # ── Onglet Intégrateur ─────────────────────────────────────────────
        tab_int = QWidget()
        fl_int = QFormLayout(tab_int)
        fl_int.setSpacing(10)
        fl_int.setContentsMargins(12, 12, 12, 12)

        self.sp_dt_max = spin(1.0, 1e8, 7180.2, 1, "s", TOOLTIPS["dt_max"])
        self.sp_dt_min = spin(0.001, 1e6, 9.677, 3, "s", TOOLTIPS["dt_min"])
        self.sp_soft = spin(1.0, 1e12, 45568.0, 0, "m", TOOLTIPS["softening"])
        self.sp_seuil = spin(1.0, 1e15, 0.005 * UA, 0, "m", TOOLTIPS["dist_seuil_m"])
        self.sp_alpha = spin(0.001, 1.0, 0.1, 3, "", TOOLTIPS["alpha"])

        row_form(fl_int, "dt maximum :", self.sp_dt_max, TOOLTIPS["dt_max"])
        row_form(fl_int, "dt minimum :", self.sp_dt_min, TOOLTIPS["dt_min"])
        row_form(fl_int, "Seuil distance (dt) :", self.sp_seuil, TOOLTIPS["dist_seuil_m"])
        row_form(fl_int, "Alpha (adaptation) :", self.sp_alpha, TOOLTIPS["alpha"])
        fl_int.addRow(self._sep_label("Stabilité numérique"))
        row_form(fl_int, "Softening :", self.sp_soft, TOOLTIPS["softening"])
        fl_int.addRow(self._sep_label("Collisions"))
        hint(fl_int,
             "  Collision si distance < rayon_i + rayon_j.\n"
             "  Rayons : onglet « Corps ».")
        self.subtabs.addTab(tab_int, "Intégrateur")

        # ── Onglet Physique ────────────────────────────────────────────────
        tab_phys = QWidget()
        fl_phys = QFormLayout(tab_phys)
        fl_phys.setSpacing(10)
        fl_phys.setContentsMargins(12, 12, 12, 12)

        self.sp_G = QDoubleSpinBox()
        self.sp_G.setRange(1e-15, 1e-5)
        self.sp_G.setDecimals(15)
        self.sp_G.setValue(6.674e-11)
        self.sp_G.setToolTip(TOOLTIPS["G_const"])
        self.sp_G.valueChanged.connect(lambda _: self._update_all_body_periods())
        self.sp_UA = QDoubleSpinBox()
        self.sp_UA.setRange(1e9, 1e13)
        self.sp_UA.setDecimals(3)
        self.sp_UA.setValue(1.496e11)
        self.sp_UA.setSuffix(" m")
        self.sp_UA.setToolTip(TOOLTIPS["UA_const"])
        self.chk_pn1 = QCheckBox("Corrections relativistes PN1")
        self.sp_c_light = spin(1.0, 1e12, 299792458.0, 0, "m/s", TOOLTIPS["c_light"])
        self.sp_c_light.setSingleStep(1e6)

        row_form(fl_phys, "Constante G :", self.sp_G, TOOLTIPS["G_const"])
        row_form(fl_phys, "Unité UA :", self.sp_UA, TOOLTIPS["UA_const"])
        fl_phys.addRow(self._sep_label("Relativité"))
        fl_phys.addRow("", self.chk_pn1)
        row_form(fl_phys, "Vitesse lumière c :", self.sp_c_light, TOOLTIPS["c_light"])
        self.subtabs.addTab(tab_phys, "Physique")

        # ── Onglet Reprise ─────────────────────────────────────────────────
        tab_reprise = QWidget()
        fl_rep = QFormLayout(tab_reprise)
        fl_rep.setSpacing(10)
        fl_rep.setContentsMargins(12, 12, 12, 12)

        self.sp_ckpt = spin(0.001, 1e6, 10.0, 3, "ans", TOOLTIPS["checkpoint_every_ans"])
        self.le_ckpt_dir = QLineEdit("checkpoints")
        self.le_ckpt_dir.setToolTip(TOOLTIPS["checkpoint_dir"])
        self.chk_resume = QCheckBox("Reprendre depuis checkpoint (--resume)")
        self.chk_nockpt = QCheckBox("Désactiver les checkpoints (--no-checkpoint)")

        row_form(fl_rep, "Checkpoint (fréquence) :", self.sp_ckpt, TOOLTIPS["checkpoint_every_ans"])
        row_form(fl_rep, "Dossier checkpoints :", self.le_ckpt_dir, TOOLTIPS["checkpoint_dir"])
        fl_rep.addRow(self._sep_label("Lancement"))
        fl_rep.addRow("", self.chk_resume)
        fl_rep.addRow("", self.chk_nockpt)
        self.subtabs.addTab(tab_reprise, "Reprise")

        # ── Onglet Viewer ──────────────────────────────────────────────────
        tab_viewer = QWidget()
        fl_view = QFormLayout(tab_viewer)
        fl_view.setSpacing(10)
        fl_view.setContentsMargins(12, 12, 12, 12)

        self.sp_rt_every = QSpinBox()
        self.sp_rt_every.setRange(1, 10000)
        self.sp_rt_every.setValue(50)
        self.sp_rt_every.setToolTip(TOOLTIPS["realtime_every"])
        self.sp_web_port = QSpinBox()
        self.sp_web_port.setRange(1024, 65535)
        self.sp_web_port.setValue(5050)
        self.sp_web_port.setToolTip(TOOLTIPS["web_port"])
        self.chk_rt = QCheckBox("Visualisation matplotlib (--realtime)")
        self.chk_web = QCheckBox("Viewer web Three.js (--web)")

        row_form(fl_view, "Rafraîchissement (frames) :", self.sp_rt_every, TOOLTIPS["realtime_every"])
        row_form(fl_view, "Port web :", self.sp_web_port, TOOLTIPS["web_port"])
        fl_view.addRow(self._sep_label("Modes"))
        fl_view.addRow("", self.chk_rt)
        fl_view.addRow("", self.chk_web)
        hint(fl_view, "  Les viewers ralentissent la simulation.")
        self.subtabs.addTab(tab_viewer, "Viewer")

        # ── Onglet Système (Python + CPU) ──────────────────────────────────
        tab_systeme = QWidget()
        fl_sys = QFormLayout(tab_systeme)
        fl_sys.setSpacing(10)
        fl_sys.setContentsMargins(12, 12, 12, 12)

        def row_sys(label, widget, tip=""):
            lbl = QLabel(label)
            if tip:
                lbl.setToolTip(tip)
            fl_sys.addRow(lbl, widget)

        fl_sys.addRow(self._sep_label("Interpréteur Python"))

        self.lbl_python_env = QLabel()
        self.lbl_python_env.setWordWrap(True)
        self.lbl_python_env.setTextFormat(Qt.RichText)
        self.lbl_python_env.setStyleSheet(f"color:{TEXT_COL}; font-size:11px;")
        fl_sys.addRow("", self.lbl_python_env)

        lbl_py_hint = QLabel(
            "  Détecte WinPython/, winpython/, WPy64-.../ à côté du projet "
            "ou à la racine parente de la clé USB.\n"
            "  Vérifiez aussi le champ « Dossier » (dossier de travail) en haut de l'onglet."
        )
        lbl_py_hint.setWordWrap(True)
        lbl_py_hint.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        fl_sys.addRow("", lbl_py_hint)

        fl_sys.addRow(self._sep_label("Processeur et parallélisme Numba"))

        self.lbl_cpu_info = QLabel()
        self.lbl_cpu_info.setWordWrap(True)
        self.lbl_cpu_info.setStyleSheet(f"color:{TEXT_COL}; font-size:11px;")
        fl_sys.addRow("", self.lbl_cpu_info)

        self.chk_threads_auto = QCheckBox("Réglage automatique des threads")
        self.chk_threads_auto.setChecked(True)
        self.chk_threads_auto.setToolTip(TOOLTIPS["num_threads"])
        fl_sys.addRow("", self.chk_threads_auto)

        self.sp_num_threads = QSpinBox()
        _cores = os.cpu_count() or 1
        self.sp_num_threads.setRange(1, max(1, _cores))
        self.sp_num_threads.setValue(default_num_threads(_cores))
        self.sp_num_threads.setEnabled(False)
        self.sp_num_threads.setToolTip(TOOLTIPS["num_threads"])
        row_sys("Threads Numba :", self.sp_num_threads, TOOLTIPS["num_threads"])

        self.lbl_threads_hint = QLabel(
            f"  Machine : {_cores} cœur(s) détecté(s). "
            f"Auto → {default_num_threads(_cores)} thread(s) pour laisser Windows respirer."
        )
        self.lbl_threads_hint.setWordWrap(True)
        self.lbl_threads_hint.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        fl_sys.addRow("", self.lbl_threads_hint)

        self.chk_threads_auto.toggled.connect(self._on_threads_auto_toggled)
        self.sp_num_threads.valueChanged.connect(lambda _: self._update_runtime_info())

        self.subtabs.addTab(tab_systeme, "Système")


        tab_bodies = QWidget()
        bodies_layout = QVBoxLayout(tab_bodies)
        bodies_layout.setContentsMargins(4, 4, 4, 4)

        btn_add = QPushButton("+ Ajouter un corps")
        btn_add.clicked.connect(self._add_body)
        bodies_layout.addWidget(btn_add)

        self.bodies_scroll = QScrollArea()
        self.bodies_scroll.setWidgetResizable(True)
        self.bodies_container = QWidget()
        self.bodies_vbox = QVBoxLayout(self.bodies_container)
        self.bodies_vbox.setSpacing(6)
        self.bodies_vbox.addStretch()
        self.bodies_scroll.setWidget(self.bodies_container)
        bodies_layout.addWidget(self.bodies_scroll)

        self.subtabs.addTab(tab_bodies, "Corps")

        # Ajouter corps par défaut (Central + 2 planètes)
        self._init_default_bodies()

        # ── Console droite ───────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        console_hdr = QHBoxLayout()
        console_hdr.addWidget(QLabel("Console"))
        btn_clear = QPushButton("Effacer")
        btn_clear.clicked.connect(lambda: self.console.clear())
        console_hdr.addStretch()
        console_hdr.addWidget(btn_clear)
        right_layout.addLayout(console_hdr)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        right_layout.addWidget(self.console)

        self.lbl_status = QLabel("En attente…")
        self.lbl_status.setObjectName("muted")
        right_layout.addWidget(self.lbl_status)

        main_split.addWidget(right)
        main_split.setStretch(0, 7)
        main_split.setStretch(1, 3)

        # ── Boutons lancement ────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color:{BORDER};"); root.addWidget(sep2)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Lancer la simulation")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("■  Arreter")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(220)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.progress)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._refresh_saved_configs_list()
        self._load_initial_settings()
        self._update_runtime_info()

    def _project_dir(self) -> str:
        script = self.le_script.text().strip()
        cwd = self.le_cwd.text().strip()
        if cwd:
            return cwd
        if script:
            return os.path.dirname(script)
        return APP_DIR

    def _on_threads_auto_toggled(self, checked: bool):
        self.sp_num_threads.setEnabled(not checked)
        self._update_runtime_info()

    def _effective_num_threads(self) -> int:
        info = describe_python_environment(self._project_dir())
        if self.chk_threads_auto.isChecked():
            return info["auto_threads"]
        return self.sp_num_threads.value()

    def _active_settings_path(self) -> str:
        """Chemin unique lu par config.py / moteur_astralis.py."""
        from config import SETTINGS_PATH
        return str(SETTINGS_PATH)

    def _update_runtime_info(self):
        info = describe_python_environment(self._project_dir())
        cores = info["cores"]
        self.sp_num_threads.setMaximum(max(1, cores))
        if self.chk_threads_auto.isChecked():
            threads = info["auto_threads"]
            mode_threads = f"auto → {threads} thread(s)"
        else:
            threads = self.sp_num_threads.value()
            mode_threads = f"manuel → {threads} thread(s)"

        icon_py = "🔌" if info["portable"] else "💻"
        extra = ""
        if not info["portable"]:
            searched = "<br>".join(info["searched_roots"])
            if info["portable_dirs"]:
                extra = (
                    "<br><span style='color:#f0ad4e'>Dossier portable trouvé, "
                    "mais python.exe introuvable dedans.</span>"
                    f"<br><span style='font-size:10px'>{info['portable_dirs'][0]}</span>"
                )
            else:
                extra = (
                    "<br><span style='color:#f0ad4e'>Aucun dossier WinPython / WPy* détecté.</span>"
                    f"<br><span style='font-size:10px'>Dossiers inspectés :<br>{searched}</span>"
                )
        self.lbl_python_env.setText(
            f"{icon_py}  Python : <b>{info['label']}</b><br>"
            f"    <span style='font-size:10px'>{info['exe']}</span>"
            f"{extra}"
        )
        self.lbl_cpu_info.setText(
            f"⚙️  Processeur : {cores} cœur(s) — {mode_threads} pour Numba"
        )
        self.lbl_threads_hint.setText(
            f"  Machine : {cores} cœur(s) détecté(s). "
            f"Auto → {info['auto_threads']} thread(s) pour laisser Windows respirer."
        )

    def _sep_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{MUTED}; font-size:11px; margin-top:6px;")
        return lbl

    def _init_default_bodies(self):
        """Crée les 3 corps par défaut (étoile + 2 planètes)."""
        defaults = [
            {"name": "Etoile centrale", "mass": 1.989e30,
             "pos": [0.0, 0.0, 0.0], "use_auto_vel": False,
             "vel_manual": [0.0, 0.0, 0.0], "incl_deg": 0.0, "sens": 1},
            {"name": "Planete 1", "mass": 5.972e24,
             "pos": [1.0 * UA, 0.0, 0.0], "use_auto_vel": True,
             "vel_manual": [0.0, 0.0, 0.0], "incl_deg": 23.4, "sens": 1},
            {"name": "Planete 2", "mass": 6.418e23,
             "pos": [1.52 * UA, 0.0, 0.0], "use_auto_vel": True,
             "vel_manual": [0.0, 0.0, 0.0], "incl_deg": 25.2, "sens": 1},
        ]
        for i, d in enumerate(defaults):
            self._add_body(d, i)
        self._update_all_body_periods()

    def absolute_pos_for_index(self, index: int) -> list:
        """Position absolue (m) du corps à l'index donné selon l'UI actuelle."""
        if 0 <= index < len(self._body_widgets):
            return self._body_widgets[index].absolute_position()
        return [0.0, 0.0, 0.0]

    def refresh_satellite_position_hints(self):
        for w in self._body_widgets:
            if w._parent_index is not None:
                w._update_abs_pos_hint()

    def _prepare_body_data_for_ui(self, data: dict, body_index: int,
                                  abs_positions: list = None) -> dict:
        """Convertit pos absolue → relative pour l'affichage des satellites."""
        d = dict(data)
        pi = d.get("parent_index")
        if pi is None:
            return d
        pi = int(pi)
        if pi < 0 or pi >= body_index:
            d.pop("parent_index", None)
            return d
        if "pos_rel" in d:
            d["pos"] = [float(v) for v in d["pos_rel"]]
            return d
        abs_pos = d.get("pos", [0.0, 0.0, 0.0])
        if abs_positions is not None and pi < len(abs_positions):
            ppos = abs_positions[pi]
        else:
            ppos = [0.0, 0.0, 0.0]
        d["pos"] = [float(abs_pos[k]) - float(ppos[k]) for k in range(3)]
        return d

    def _update_all_body_periods(self):
        """Recalcule et affiche les périodes orbitales / rotation pour chaque corps."""
        from core.periods import compute_body_periods
        if not self._body_widgets:
            return
        bodies = [w.to_dict() for w in self._body_widgets]
        try:
            entries = compute_body_periods(bodies, self.sp_G.value())
        except (TypeError, ValueError):
            return
        for w, entry in zip(self._body_widgets, entries):
            w.set_period_labels(
                entry.get("orbital_label", "—"),
                entry.get("rotation_label", "—"),
                entry.get("orbital_hint", ""),
                entry.get("rotation_hint", ""),
            )

    def _insert_index_after_parent_children(self, parent_index: int) -> int:
        """Position dans la liste pour insérer un satellite après ses frères."""
        insert_at = parent_index + 1
        while insert_at < len(self._body_widgets):
            if self._body_widgets[insert_at]._parent_index == parent_index:
                insert_at += 1
            else:
                break
        return insert_at

    def _renumber_bodies(self):
        old_to_new = {w.index: i for i, w in enumerate(self._body_widgets)}
        for i, w in enumerate(self._body_widgets):
            w.index = i
            if w._parent_index is not None:
                new_p = old_to_new.get(w._parent_index)
                w._parent_index = new_p if new_p is not None and new_p < w.index else None
            w._apply_satellite_style()
            w._update_title(w.le_name.text())
            w._update_position_labels()
            w._update_abs_pos_hint()
        self.refresh_satellite_position_hints()

    def _collect_descendants(self, parent_index: int) -> list:
        """Satellites (et sous-satellites) d'un corps."""
        found = []
        for w in self._body_widgets:
            if w._parent_index == parent_index:
                found.append(w)
                found.extend(self._collect_descendants(w.index))
        return found

    def _add_body(self, data: dict = None, index: int = None, insert_at: int = None,
                  ui_ready: bool = False):
        # Sécurité PyQt : ignore le signal du bouton clicked()
        if isinstance(data, bool):
            data = None
        if insert_at is None:
            if index is None:
                index = len(self._body_widgets)
            insert_at = index
        ui_data = data or dict(BODY_DEFAULTS)
        if data is not None and not ui_ready:
            ui_data = self._prepare_body_data_for_ui(data, insert_at)
        bw = BodyWidget(insert_at, ui_data, period_host=self)
        bw.removed.connect(self._remove_body)
        self._body_widgets.insert(insert_at, bw)
        vbox_idx = min(insert_at, max(0, self.bodies_vbox.count() - 1))
        self.bodies_vbox.insertWidget(vbox_idx, bw)
        self._renumber_bodies()
        self._update_all_body_periods()

    def _add_satellite(self, parent_bw: BodyWidget):
        """Ajoute un satellite personnalisable en orbite autour de parent_bw."""
        parent_idx = parent_bw.index
        n_sat = sum(
            1 for w in self._body_widgets if w._parent_index == parent_idx
        ) + 1
        pname = parent_bw.le_name.text().strip() or f"Corps {parent_idx}"
        orbit_m = DEFAULT_SATELLITE_ORBIT_M
        rel = [orbit_m, 0.0, 0.0]
        ppos = parent_bw.absolute_position()
        data = dict(SATELLITE_BODY_DEFAULTS)
        data.update({
            "name": f"Satellite {n_sat} ({pname})",
            "pos_rel": rel,
            "pos": [ppos[k] + rel[k] for k in range(3)],
            "parent_index": parent_idx,
            "use_auto_vel": True,
            "vel_manual": [0.0, 0.0, 0.0],
        })
        insert_at = self._insert_index_after_parent_children(parent_idx)
        self._add_body(data, insert_at=insert_at)
        if hasattr(self, "console"):
            self.console.append(
                f"Satellite ajoute autour de « {pname} » "
                f"(orbite ~{_fmt_distance_m(orbit_m)})."
            )

    def _remove_body(self, bw: BodyWidget):
        to_remove = [bw] + self._collect_descendants(bw.index)
        remaining = len(self._body_widgets) - len(to_remove)
        if remaining < 2:
            QMessageBox.warning(
                self, "Minimum",
                "Il faut au moins 2 corps (suppression des satellites incluse).",
            )
            return
        for w in to_remove:
            if w in self._body_widgets:
                self._body_widgets.remove(w)
                w.setParent(None)
                w.deleteLater()
        self._renumber_bodies()
        self._update_all_body_periods()

    def _refresh_saved_configs_list(self, select_path: str = ""):
        """Remplit la liste des JSON dans configs/."""
        self.cb_saved_configs.blockSignals(True)
        self.cb_saved_configs.clear()
        self.cb_saved_configs.addItem("-- Choisir une config --", "")
        for path in sorted(glob.glob(os.path.join(_configs_dir(), "*.json"))):
            self.cb_saved_configs.addItem(os.path.basename(path), path)
        if select_path:
            idx = self.cb_saved_configs.findData(select_path)
            if idx >= 0:
                self.cb_saved_configs.setCurrentIndex(idx)
        self.cb_saved_configs.blockSignals(False)

    def _load_settings_from_path(self, path: str, sync_active: bool = True):
        """Charge un fichier JSON dans l'UI et optionnellement dans app/settings.json."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self._apply_settings(s)
            self.console.append(f"Parametres charges depuis {path}")
            if sync_active:
                active = self._active_settings_path()
                self._write_settings(active)
                self.console.append(f"Config active → {active}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{e}")

    def _load_settings_file(self):
        """Ouvre le dossier configs/ pour charger une configuration."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une configuration",
            _configs_dir(), "JSON (*.json);;Tous (*.*)")
        if not path:
            return
        self._load_settings_from_path(path)
        self._refresh_saved_configs_list(select_path=path)

    def _save_settings_preset(self):
        """Enregistre la configuration courante dans configs/."""
        default_name = self.le_output_name.text().strip() or "ma_configuration"
        default_name = _sanitize_output_basename(default_name) or "ma_configuration"
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la configuration",
            os.path.join(_configs_dir(), default_name + ".json"),
            "JSON (*.json);;Tous (*.*)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self._write_settings(path)
            self.console.append(f"Configuration sauvegardee : {path}")
            self._refresh_saved_configs_list(select_path=path)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer :\n{e}")

    def _on_saved_config_selected(self, index: int):
        if index <= 0:
            return
        path = self.cb_saved_configs.currentData()
        if path and os.path.isfile(path):
            self._load_settings_from_path(path)

    def _apply_preset(self, name: str):
        """Applique un preset de scénario."""
        presets = {
            "Soleil - Terre - Mars": {
                "simulation": {"duree_ans": 10.0, "dt_max": 7180.2, "dt_min": 9.677,
                               "softening": 45568.0, "alpha": 0.1},
                "bodies": [
                    {"name": "Soleil",  "mass": 1.989e30, "pos": [0,0,0],
                     "use_auto_vel": False, "vel_manual": [0,0,0], "incl_deg": 0, "sens": 1},
                    {"name": "Terre",   "mass": 5.972e24, "pos": [1.0*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 23.4, "sens": 1},
                    {"name": "Mars",    "mass": 6.418e23, "pos": [1.52*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 25.2, "sens": 1},
                ],
            },
            "Soleil - Terre - Lune": {
                # dt_max = 900 s (15 min) : adapté à l'orbite lunaire (~384 400 km).
                # dist_seuil_m = 2e8 m : réduit le pas quand Lune < 200 000 km de la Terre.
                "simulation": {"duree_ans": 2.0, "dt_max": 900.0, "dt_min": 1.0,
                               "softening": 1000.0, "alpha": 0.05,
                               "dist_seuil_m": 2.0e8},
                "bodies": [
                    {"name": "Soleil",  "mass": 1.989e30, "rayon": 6.957e8,
                     "pos": [0,0,0],
                     "use_auto_vel": False, "vel_manual": [0,0,0], "incl_deg": 0, "sens": 1,
                     "j2r2": 0.0, "mdot": 0.0, "spin_rate": 0.0,
                     "spin_axis": [0,0,1], "k2": 0.0, "inertia_factor": 0.4},
                    {"name": "Terre",   "mass": 5.972e24, "rayon": 6.371e6,
                     "pos": [1.0*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 23.4, "sens": 1,
                     "j2r2": 0.0, "mdot": 0.0, "spin_rate": 0.0,
                     "spin_axis": [0,0,1], "k2": 0.0, "inertia_factor": 0.4},
                    {"name": "Lune",    "mass": 7.342e22, "rayon": 1.7374e6,
                     "pos": [1.0*UA + 384400e3, 0, 0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 5.145, "sens": 1,
                     "parent_index": 1,
                     "pos_rel": [384400e3, 0, 0],
                     "j2r2": 0.0, "mdot": 0.0, "spin_rate": 0.0,
                     "spin_axis": [0,0,1], "k2": 0.0, "inertia_factor": 0.4},
                ],
            },
            "Soleil - Terre - Mars - Jupiter": {
                "simulation": {"duree_ans": 50.0, "dt_max": 7180.2, "dt_min": 9.677,
                               "softening": 45568.0, "alpha": 0.1},
                "bodies": [
                    {"name": "Soleil",  "mass": 1.989e30, "pos": [0,0,0],
                     "use_auto_vel": False, "vel_manual": [0,0,0], "incl_deg": 0, "sens": 1},
                    {"name": "Terre",   "mass": 5.972e24, "pos": [1.0*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 23.4, "sens": 1},
                    {"name": "Mars",    "mass": 6.418e23, "pos": [1.52*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 25.2, "sens": 1},
                    {"name": "Jupiter", "mass": 1.898e27, "pos": [5.20*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 1.3, "sens": 1},
                ],
            },
            "Etoile binaire": {
                "simulation": {"duree_ans": 5.0, "dt_max": 3600.0, "dt_min": 1.0,
                               "softening": 1e6, "alpha": 0.05},
                "bodies": [
                    {"name": "Etoile A", "mass": 1.989e30, "pos": [0.5*UA,0,0],
                     "use_auto_vel": False, "vel_manual": [0, 15000, 0], "incl_deg": 0, "sens": 1},
                    {"name": "Etoile B", "mass": 1.5e30,   "pos": [-0.5*UA,0,0],
                     "use_auto_vel": False, "vel_manual": [0, -20000, 0], "incl_deg": 0, "sens": 1},
                ],
            },
            "Systeme compact 3 corps": {
                "simulation": {"duree_ans": 2.0, "dt_max": 3600.0, "dt_min": 1.0,
                               "softening": 1e5, "alpha": 0.05},
                "bodies": [
                    {"name": "Central", "mass": 1.989e30, "pos": [0,0,0],
                     "use_auto_vel": False, "vel_manual": [0,0,0], "incl_deg": 0, "sens": 1},
                    {"name": "Corps A", "mass": 5e24, "pos": [0.5*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 0, "sens": 1},
                    {"name": "Corps B", "mass": 5e24, "pos": [0.8*UA,0,0],
                     "use_auto_vel": True, "vel_manual": [0,0,0], "incl_deg": 45.0, "sens": -1},
                ],
            },
        }
        if name not in presets:
            return
        self._apply_settings(presets[name])
        self.console.append(f"Preset charge : {name}")

    def _apply_settings(self, s: dict):
        """Applique un dict settings à tous les widgets."""
        physics = s.get("physics", {})
        if "G" in physics:
            self.sp_G.setValue(float(physics["G"]))
        if "UA" in physics:
            self.sp_UA.setValue(float(physics["UA"]))

        sim = s.get("simulation", {})
        if "duree_ans" in sim:
            self.sp_duree.setValue(float(sim["duree_ans"]))
        if "save_every" in sim:
            self.sp_save.setValue(int(sim["save_every"]))
        if "monitor_every" in sim:
            self.sp_monitor.setValue(int(sim["monitor_every"]))
        if "realtime_every" in sim:
            self.sp_rt_every.setValue(int(sim["realtime_every"]))
        if "flush_every" in sim:
            self.sp_flush.setValue(int(sim["flush_every"]))
        if "web_port" in sim:
            self.sp_web_port.setValue(int(sim["web_port"]))
        if "output_dir" in sim:
            self.le_output_dir.setText(str(sim["output_dir"]))
        if "checkpoint_dir" in sim:
            self.le_ckpt_dir.setText(str(sim["checkpoint_dir"]))
        if "dt_max" in sim:
            self.sp_dt_max.setValue(float(sim["dt_max"]))
        if "dt_min" in sim:
            self.sp_dt_min.setValue(float(sim["dt_min"]))
        if "softening" in sim:
            self.sp_soft.setValue(float(sim["softening"]))
        if "dist_seuil_m" in sim:
            self.sp_seuil.setValue(float(sim["dist_seuil_m"]))
        elif "dist_seuil_ua" in sim:
            ua = float(physics.get("UA", self.sp_UA.value()))
            self.sp_seuil.setValue(float(sim["dist_seuil_ua"]) * ua)
        if "alpha" in sim:
            self.sp_alpha.setValue(float(sim["alpha"]))
        if "enable_pn1" in sim:
            self.chk_pn1.setChecked(bool(sim["enable_pn1"]))
        if "c_light" in sim:
            self.sp_c_light.setValue(float(sim["c_light"]))
        if "checkpoint_every_ans" in sim:
            self.sp_ckpt.setValue(float(sim["checkpoint_every_ans"]))
        if "output_name" in sim:
            self.le_output_name.setText(str(sim["output_name"]))
        if "num_threads" in sim:
            nt = int(sim["num_threads"])
            if nt <= 0:
                self.chk_threads_auto.setChecked(True)
            else:
                self.chk_threads_auto.setChecked(False)
                self.sp_num_threads.setValue(min(nt, self.sp_num_threads.maximum()))
        self._update_runtime_info()

        bodies = s.get("bodies", [])
        if bodies:
            from core.body_positions import resolve_body_positions
            abs_positions = resolve_body_positions(bodies)
            for bw in list(self._body_widgets):
                bw.setParent(None)
                bw.deleteLater()
            self._body_widgets.clear()
            for i, bd in enumerate(bodies):
                ui_bd = self._prepare_body_data_for_ui(bd, i, abs_positions)
                self._add_body(ui_bd, i, ui_ready=True)
        else:
            self._update_all_body_periods()

    def _load_initial_settings(self):
        """Charge settings.json du projet au démarrage du dashboard."""
        path = self._active_settings_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._apply_settings(json.load(f))
        except Exception as e:
            if hasattr(self, "console"):
                self.console.append(f"Note: settings.json non chargé ({e})")

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir moteur_astralis.py", "", "Python (*.py)")
        if path:
            self.le_script.setText(path)
            self.le_cwd.setText(os.path.dirname(path))

    def _collect_settings(self) -> dict:
        """Construit le dict settings complet depuis l'UI."""
        bodies = [bw.to_dict() for bw in self._body_widgets]
        from core.body_positions import sync_absolute_positions
        sync_absolute_positions(bodies)
        return {
            "physics": {
                "G":  self.sp_G.value(),
                "UA": self.sp_UA.value(),
            },
            "simulation": {
                "duree_ans":            self.sp_duree.value(),
                "save_every":           self.sp_save.value(),
                "monitor_every":        self.sp_monitor.value(),
                "realtime_every":       self.sp_rt_every.value(),
                "flush_every":          self.sp_flush.value(),
                "web_port":             self.sp_web_port.value(),
                "output_dir":           self.le_output_dir.text().strip() or "outputs",
                "output_name":          self.le_output_name.text().strip(),
                "checkpoint_dir":       self.le_ckpt_dir.text().strip() or "checkpoints",
                "dt_max":               self.sp_dt_max.value(),
                "dt_min":               self.sp_dt_min.value(),
                "softening":            self.sp_soft.value(),
                "dist_seuil_m":         self.sp_seuil.value(),
                "alpha":                self.sp_alpha.value(),
                "enable_pn1":           self.chk_pn1.isChecked(),
                "c_light":              self.sp_c_light.value(),
                "checkpoint_every_ans": self.sp_ckpt.value(),
                "num_threads":          0 if self.chk_threads_auto.isChecked() else self.sp_num_threads.value(),
            },
            "bodies": bodies,
        }

    def _write_settings(self, settings_path: str):
        from config import _normalize_settings
        settings = _normalize_settings(self._collect_settings())
        if settings_path == self._active_settings_path():
            existing = {}
        elif os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        else:
            existing = {}
        if settings_path != self._active_settings_path():
            merged = _deep_merge_settings(existing, settings)
            settings = merged
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

    def _start(self):
        script = self.le_script.text().strip()
        if not _moteur_script_ok(script):
            QMessageBox.warning(self, "Script manquant",
                "Selectionnez moteur_astralis.py avec le bouton Parcourir.")
            return

        cwd = self.le_cwd.text().strip() or os.path.dirname(script)
        settings_path = self._active_settings_path()

        try:
            self._write_settings(settings_path)
            self.console.append(f"Parametres actifs → {settings_path}")
            if cwd != APP_DIR:
                self.console.append(
                    "Note: settings.json est toujours ecrit dans app/ "
                    "(lu par la simulation), pas dans le dossier de travail."
                )
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'écrire settings.json :\n{e}")
            return

        out_preview = _preview_parquet_filename(
            self.le_output_name.text().strip(),
            self.le_output_dir.text().strip() or "outputs",
        )
        self.console.append(f"Fichier Parquet prevu : {out_preview}")

        args = []
        if self.chk_rt.isChecked():
            args.append("--realtime")
        if self.chk_web.isChecked():
            args.append("--web")
            args.extend(["--port", str(int(self.sp_web_port.value()))])
        if self.chk_resume.isChecked():
            args.append("--resume")
        if self.chk_nockpt.isChecked():
            args.append("--no-checkpoint")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        self._progress_file = os.path.join(OUTPUTS_DIR, "sim_progress.json")
        self._progress_timer.start()
        self.lbl_status.setText("Simulation en cours… 0 %")
        self.console.append(f"\n{'-'*55}")
        self.console.append(f"[{datetime.now().strftime('%H:%M:%S')}] Lancement")
        self.console.append(f"{'-'*55}\n")

        python_exe = resolve_python_executable(cwd)
        env_info = describe_python_environment(cwd)
        threads = self._effective_num_threads()
        self.console.append(
            f"Python : {env_info['label']} — {python_exe}"
        )
        self.console.append(
            f"Threads Numba : {threads} ({'auto' if self.chk_threads_auto.isChecked() else 'manuel'}) "
            f"| {env_info['cores']} cœurs détectés"
        )

        self.worker = SimulationWorker(script, args, cwd, python_exe=python_exe)
        self.worker.log_line.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop(self):
        self._progress_timer.stop()
        if self.worker:
            self.worker.stop()

    def _poll_sim_progress(self):
        """Lit sim_progress.json écrit par moteur_astralis.py."""
        if not self._progress_file or not os.path.isfile(self._progress_file):
            return
        try:
            with open(self._progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ratio = float(data.get("ratio", 0))
            pct = int(min(max(ratio * 100.0, 0.0), 100.0))
            self.progress.setValue(pct)
            t_s = float(data.get("t", 0))
            t_tot_s = float(data.get("t_total", 0))
            t_ans = t_s / 86400.0 / 365.25
            t_tot_ans = t_tot_s / 86400.0 / 365.25 if t_tot_s > 0 else 0.0
            frame = int(data.get("frame", 0))
            self.lbl_status.setText(
                f"Simulation… {pct} % — {t_ans:.1f}/{t_tot_ans:.1f} ans — frame {frame:,}"
            )
            if data.get("status") == "done":
                self.progress.setValue(100)
                self._progress_timer.stop()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    def _on_log(self, line: str):
        self.console.append(line)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())

    def _on_finished(self, ok: bool, msg: str):
        self._progress_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if ok:
            self.progress.setValue(100)
        self.progress.setVisible(False)
        status = "Simulation terminee avec succes." if ok else f"Simulation terminee avec erreur. {msg}"
        self.lbl_status.setText(status)
        self.console.append(f"\n{status}\n")

        if ok:
            parquet = load_last_run_parquet(self._project_dir())
            if parquet:
                win = self.window()
                if hasattr(win, "_load_file"):
                    self.console.append(f"Chargement du resultat : {parquet}")
                    win._load_file(parquet)
                    self.lbl_status.setText(
                        f"Analyse chargee — {os.path.basename(parquet)}"
                    )



# ─────────────────────────────────────────────────────────
#  Détecteur d'événements orbitaux
# ─────────────────────────────────────────────────────────
class EventDetector:
    """
    Détecte et catalogue les événements notables dans les données de simulation.
    Tous les temps sont en années simulées.
    Chaque événement est un dict :
      { 'type', 't_ans', 'desc', 'severity' }
      severity : 'info' | 'warning' | 'critical'
    """

    # Couleurs par sévérité pour les marqueurs
    COLORS = {"info": "#4FC3F7", "warning": "#FFD740", "critical": "#EF5350"}

    def __init__(self, data: "SimData"):
        self.data   = data
        self.events: list[dict] = []

    # ── API publique ───────────────────────────────────────
    def detect_all(self,
                   approche_seuil_m: float = None,
                   ejection_e_seuil:  float = 0.95) -> list[dict]:
        """Lance toutes les détections et retourne la liste triée par temps."""
        if approche_seuil_m is None:
            approche_seuil_m = 0.1 * UA
        self.events = []
        self._detect_approches(approche_seuil_m)
        self._detect_minima_distances()
        self._detect_ejections(ejection_e_seuil)
        self._detect_croisements()
        self.events.sort(key=lambda e: e["t_ans"])
        return self.events

    # ── Détections individuelles ───────────────────────────
    def _detect_approches(self, seuil_m: float):
        """Passages sous le seuil de distance pour chaque paire."""
        d = self.data
        t  = d.t_jours / 365.25
        seuil_ua = seuil_m / UA if UA > 0 else seuil_m
        for i in range(d.n_bodies):
            for j in range(i + 1, d.n_bodies):
                dist = d.distance_pair(i, j)           # en UA
                below = dist < seuil_ua
                if not np.any(below):
                    continue
                # Détecter les fronts descendants (entrée sous le seuil)
                transitions = np.where(np.diff(below.astype(int)) == 1)[0]
                ni = d.body_names[i]; nj = d.body_names[j]
                for idx in transitions:
                    d_min_ua = float(dist[idx + 1])
                    d_min_m = d_min_ua * UA
                    self.events.append({
                        "type":     "Approche proche",
                        "t_ans":    float(t[idx + 1]),
                        "desc":     (
                            f"{ni} — {nj} : d = {_fmt_distance_m(d_min_m)}  "
                            f"(seuil {_fmt_distance_m(seuil_m)})"
                        ),
                        "severity": "critical" if d_min_ua < seuil_ua * 0.3 else "warning",
                        "pair":     (i, j),
                    })

    def _detect_minima_distances(self):
        """Distance minimale absolue pour chaque paire (1 événement par paire)."""
        d = self.data
        t = d.t_jours / 365.25
        for i in range(d.n_bodies):
            for j in range(i + 1, d.n_bodies):
                dist = d.distance_pair(i, j)
                idx_min = int(np.argmin(dist))
                ni = d.body_names[i]; nj = d.body_names[j]
                self.events.append({
                    "type":     "Distance minimale",
                    "t_ans":    float(t[idx_min]),
                    "desc":     (
                        f"{ni} — {nj} : dmin = "
                        f"{_fmt_distance_m(float(dist[idx_min]) * UA)}"
                    ),
                    "severity": "info",
                    "pair":     (i, j),
                })

    def _detect_ejections(self, e_seuil: float):
        """Excentricité dépasse e_seuil → risque d'éjection."""
        d   = self.data
        t   = d.t_jours / 365.25
        G   = 6.674e-11
        m0  = 1.989e30   # masse centrale approx
        mu  = G * m0

        for i in range(1, d.n_bodies):
            xi, yi, zi = d.positions(i)     # UA
            x0, y0, z0 = d.positions(0)
            rx = (xi - x0) * UA; ry = (yi - y0) * UA; rz = (zi - z0) * UA
            r  = np.sqrt(rx**2 + ry**2 + rz**2)
            if f"Vx{i}" not in d.df.columns:
                continue
            dvx = d.df[f"Vx{i}"].values - d.df["Vx0"].values
            dvy = d.df[f"Vy{i}"].values - d.df["Vy0"].values
            dvz = (d.df.get(f"Vz{i}", pd.Series(np.zeros(len(t)))).values
                   - d.df.get("Vz0", pd.Series(np.zeros(len(t)))).values)
            v2  = dvx**2 + dvy**2 + dvz**2
            hx  = ry*dvz - rz*dvy
            hy  = rz*dvx - rx*dvz
            hz  = rx*dvy - ry*dvx
            h2  = hx**2 + hy**2 + hz**2
            eps = v2 / 2 - mu / np.maximum(r, 1.0)
            with np.errstate(invalid='ignore'):
                e2 = 1 + 2 * eps * h2 / mu**2
                e  = np.sqrt(np.clip(e2, 0, None))

            # Fronts montants : e passe au-dessus du seuil
            above = e > e_seuil
            transitions = np.where(np.diff(above.astype(int)) == 1)[0]
            name = d.body_names[i]
            for idx in transitions:
                self.events.append({
                    "type":     "Éjection possible",
                    "t_ans":    float(t[idx + 1]),
                    "desc":     f"{name} : e = {float(e[idx+1]):.4f}  (seuil {e_seuil})",
                    "severity": "critical" if float(e[idx+1]) >= 1.0 else "warning",
                    "body":     i,
                })

    def _detect_croisements(self):
        """Le demi-grand axe d'un corps passe au-dessus de celui d'un autre."""
        d   = self.data
        t   = d.t_jours / 365.25
        G   = 6.674e-11
        m0  = 1.989e30
        mu  = G * m0

        # Calculer a(t) pour chaque corps non-central
        a_all = {}
        for i in range(1, d.n_bodies):
            xi, yi, zi = d.positions(i)
            x0, y0, z0 = d.positions(0)
            rx = (xi - x0) * UA; ry = (yi - y0) * UA; rz = (zi - z0) * UA
            r  = np.sqrt(rx**2 + ry**2 + rz**2)
            if f"Vx{i}" not in d.df.columns:
                continue
            dvx = d.df[f"Vx{i}"].values - d.df["Vx0"].values
            dvy = d.df[f"Vy{i}"].values - d.df["Vy0"].values
            v2  = dvx**2 + dvy**2
            eps = v2 / 2 - mu / np.maximum(r, 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.where(eps < 0, -mu / (2 * eps), np.nan) / UA
            a_all[i] = a

        # Chercher les croisements entre paires
        indices = list(a_all.keys())
        for k in range(len(indices)):
            for l in range(k + 1, len(indices)):
                i, j   = indices[k], indices[l]
                ai, aj = a_all[i], a_all[j]
                diff   = ai - aj
                valid  = ~(np.isnan(diff))
                if not np.any(valid):
                    continue
                # Fronts de changement de signe = croisement
                sign   = np.sign(diff[valid])
                crossings = np.where(np.diff(sign) != 0)[0]
                if len(crossings) == 0:
                    continue
                t_valid = t[valid]
                ni = d.body_names[i]; nj = d.body_names[j]
                for idx in crossings[:5]:   # max 5 par paire
                    self.events.append({
                        "type":     "Croisement d'orbites",
                        "t_ans":    float(t_valid[idx + 1]),
                        "desc":     f"{ni} ↔ {nj} : demi-grands axes s'inversent",
                        "severity": "warning",
                        "pair":     (i, j),
                    })


# ─────────────────────────────────────────────────────────
#  Onglet 8 — Diagramme de Phase
# ─────────────────────────────────────────────────────────
class PhaseTab(QWidget):
    """Diagramme de phase : distance au centre vs vitesse scalaire.
    Révèle la nature de l'orbite : cercle = orbite circulaire stable,
    ellipse fermée = orbite elliptique, spirale ouverte = éjection/capture.
    """
    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "Diagramme de phase : distance au barycentre (UA) vs vitesse (km/s). "
            "Une boucle fermée = orbite stable. Une spirale ou courbe ouverte = orbite instable ou éjection."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.cb_corps = QComboBox()
        self.cb_corps.addItem("Tous les corps")
        self.cb_corps.currentIndexChanged.connect(self.refresh)

        self.cb_ref = QComboBox()
        self.cb_ref.addItems(["Barycentre", "Corps 0"])
        self.cb_ref.currentIndexChanged.connect(self.refresh)

        self.cb_color = QComboBox()
        self.cb_color.addItems(["Couleur uniforme", "Couleur par temps"])
        self.cb_color.currentIndexChanged.connect(self.refresh)

        ctrl.addWidget(QLabel("Corps :"))
        ctrl.addWidget(self.cb_corps)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Reference :"))
        ctrl.addWidget(self.cb_ref)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Couleur :"))
        ctrl.addWidget(self.cb_color)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setMaximumHeight(110)
        layout.addWidget(self.table)

    def load_data(self, data: SimData):
        self.data = data
        self.cb_corps.blockSignals(True)
        self.cb_corps.clear()
        self.cb_corps.addItem("Tous les corps")
        for name in data.body_names:
            self.cb_corps.addItem(name)
        self.cb_corps.blockSignals(False)

        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        self.cb_ref.addItem("Barycentre")
        for name in data.body_names:
            self.cb_ref.addItem(name)
        self.cb_ref.blockSignals(False)

        self.refresh()

    def refresh(self):
        if self.data is None:
            return
        d = self.data
        has_vel = "Vx0" in d.df.columns

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)

        corps_sel = self.cb_corps.currentText()
        indices = list(range(d.n_bodies))
        if corps_sel != "Tous les corps":
            idx = self.cb_corps.currentIndex() - 1
            if 0 <= idx < d.n_bodies:
                indices = [idx]

        ref_idx = self.cb_ref.currentIndex() - 1  # -1 = barycentre
        color_by_time = self.cb_color.currentText() == "Couleur par temps"

        n = len(indices)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        axes = [self.figure.add_subplot(nrows, ncols, k + 1) for k in range(n)]
        if n == 1:
            axes = [axes[0]]

        table_data = []
        for k, i in enumerate(indices):
            ax = axes[k]
            apply_dark_axes(ax)
            name = d.body_names[i] if i < len(d.body_names) else f"Corps {i}"
            color = PALETTE[i % len(PALETTE)]

            x, y, z = d.positions(i)
            if ref_idx >= 0 and ref_idx < d.n_bodies:
                xr, yr, zr = d.positions(ref_idx)
                dist = np.sqrt((x - xr)**2 + (y - yr)**2 + (z - zr)**2)
            else:
                # Barycentre
                xs = np.mean([d.positions(j)[0] for j in range(d.n_bodies)], axis=0)
                ys = np.mean([d.positions(j)[1] for j in range(d.n_bodies)], axis=0)
                zs = np.mean([d.positions(j)[2] for j in range(d.n_bodies)], axis=0)
                dist = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)

            if has_vel:
                vx = d.df[f"Vx{i}"].values / 1e3
                vy = d.df[f"Vy{i}"].values / 1e3
                vz = d.df[f"Vz{i}"].values / 1e3 if f"Vz{i}" in d.df.columns else np.zeros(len(vx))
                vel = np.sqrt(vx**2 + vy**2 + vz**2)
            else:
                # Vitesse estimée par différences finies
                dt = np.diff(d.t_jours) * 86400
                dt = np.where(dt > 0, dt, 1)
                dx = np.diff(x) * UA; dy = np.diff(y) * UA; dz = np.diff(z) * UA
                vel_raw = np.sqrt((dx/dt)**2 + (dy/dt)**2 + (dz/dt)**2) / 1e3
                vel = np.concatenate([[vel_raw[0]], vel_raw])

            if color_by_time:
                t_norm = (d.t_jours - d.t_jours[0]) / max(d.t_jours[-1] - d.t_jours[0], 1)
                sc = ax.scatter(dist, vel, c=t_norm, cmap="plasma", s=0.8, alpha=0.6, linewidths=0)
                self.figure.colorbar(sc, ax=ax, label="Temps (normalisé)", pad=0.02)
            else:
                ax.plot(dist, vel, color=color, lw=0.5, alpha=0.7)
                # Point de départ et fin
                ax.scatter([dist[0]],  [vel[0]],  color="#66BB6A", s=30, zorder=5, label="Début")
                ax.scatter([dist[-1]], [vel[-1]], color="#EF5350", s=30, zorder=5, label="Fin")
                ax.legend(fontsize=7, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)

            ax.set_xlabel("Distance (UA)", fontsize=8)
            ax.set_ylabel("Vitesse (km/s)", fontsize=8)
            ax.set_title(name, color=TEXT_COL, fontsize=9, pad=3)

            table_data.append((
                name,
                f"{float(dist.min()):.3f}", f"{float(dist.max()):.3f}",
                f"{float(vel.min()):.2f}",  f"{float(vel.max()):.2f}",
            ))

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()

        self.table.clear()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Corps", "Dist. min (UA)", "Dist. max (UA)", "Vit. min (km/s)", "Vit. max (km/s)"])
        self.table.setRowCount(len(table_data))
        for r, row in enumerate(table_data):
            for c, v in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)


# ─────────────────────────────────────────────────────────
#  Onglet 9 — Carte de Densité
# ─────────────────────────────────────────────────────────
class DensiteTab(QWidget):
    """Carte de chaleur 2D : zones de l'espace les plus fréquentées par les corps.
    Utile pour visualiser les zones de transit, résonances orbitales,
    ou détecter des passages répétés au même endroit.
    """
    def __init__(self):
        super().__init__()
        self.data: SimData = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "Carte de densité spatiale : zones les plus fréquentées par les corps au fil du temps. "
            "Les zones lumineuses indiquent des passages répétés (résonances, orbites stables)."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QHBoxLayout()
        self.cb_plan = QComboBox()
        self.cb_plan.addItems(["Plan XY (écliptique)", "Plan XZ", "Plan YZ"])
        self.cb_plan.currentIndexChanged.connect(self.refresh)

        self.cb_corps = QComboBox()
        self.cb_corps.addItem("Tous les corps")
        self.cb_corps.currentIndexChanged.connect(self.refresh)

        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(["inferno", "plasma", "viridis", "hot", "Blues_r"])
        self.cb_cmap.currentIndexChanged.connect(self.refresh)

        self.spin_bins = QSpinBox()
        self.spin_bins.setRange(50, 500)
        self.spin_bins.setValue(200)
        self.spin_bins.setSuffix(" bins")
        self.spin_bins.editingFinished.connect(self.refresh)

        self.chk_log = QCheckBox("Echelle log")
        self.chk_log.setChecked(True)
        self.chk_log.stateChanged.connect(self.refresh)

        self.chk_orbites = QCheckBox("Superposer orbites")
        self.chk_orbites.setChecked(True)
        self.chk_orbites.stateChanged.connect(self.refresh)

        ctrl.addWidget(QLabel("Plan :"))
        ctrl.addWidget(self.cb_plan)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Corps :"))
        ctrl.addWidget(self.cb_corps)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Palette :"))
        ctrl.addWidget(self.cb_cmap)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.spin_bins)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.chk_log)
        ctrl.addWidget(self.chk_orbites)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.figure = Figure(facecolor=FIG_PANEL)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT_COL};")
        layout.addWidget(nav)
        layout.addWidget(self.canvas)

    def load_data(self, data: SimData):
        self.data = data
        self.cb_corps.blockSignals(True)
        self.cb_corps.clear()
        self.cb_corps.addItem("Tous les corps")
        for name in data.body_names:
            self.cb_corps.addItem(name)
        self.cb_corps.blockSignals(False)
        self.refresh()

    def refresh(self):
        if self.data is None:
            return
        d = self.data

        plan = self.cb_plan.currentText()
        corps_sel = self.cb_corps.currentText()
        cmap = self.cb_cmap.currentText()
        bins = self.spin_bins.value()
        log_scale = self.chk_log.isChecked()
        show_orbites = self.chk_orbites.isChecked()

        indices = list(range(d.n_bodies))
        if corps_sel != "Tous les corps":
            idx = self.cb_corps.currentIndex() - 1
            if 0 <= idx < d.n_bodies:
                indices = [idx]

        # Axes selon le plan
        plan_axes = {
            "Plan XY (écliptique)": ("X", "Y", "x (UA)", "y (UA)"),
            "Plan XZ":              ("X", "Z", "x (UA)", "z (UA)"),
            "Plan YZ":              ("Y", "Z", "y (UA)", "z (UA)"),
        }
        col_a, col_b, lbl_a, lbl_b = plan_axes[plan]

        # Collecter toutes les positions
        all_a, all_b = [], []
        for i in indices:
            try:
                xa = d.df[f"{col_a}{i}"].values.astype(float)
                xb = d.df[f"{col_b}{i}"].values.astype(float)
                all_a.append(xa)
                all_b.append(xb)
            except KeyError:
                pass

        if not all_a:
            return

        arr_a = np.concatenate(all_a)
        arr_b = np.concatenate(all_b)

        self.figure.clear()
        self.figure.patch.set_facecolor(FIG_PANEL)
        ax = self.figure.add_subplot(111)
        apply_dark_axes(ax)

        # Heatmap
        norm = matplotlib.colors.LogNorm() if log_scale else None
        h, xe, ye, im = ax.hist2d(
            arr_a, arr_b,
            bins=bins,
            norm=norm,
            cmap=cmap,
        )
        cb = self.figure.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Passages" + (" (log)" if log_scale else ""), color=TEXT_COL, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT_COL, labelcolor=TEXT_COL)

        # Superposition orbites
        if show_orbites:
            for i in indices:
                try:
                    xa = d.df[f"{col_a}{i}"].values.astype(float)
                    xb = d.df[f"{col_b}{i}"].values.astype(float)
                    color = PALETTE[i % len(PALETTE)]
                    name = d.body_names[i] if i < len(d.body_names) else f"Corps {i}"
                    ax.plot(xa, xb, color=color, lw=0.5, alpha=0.35, label=name)
                    ax.scatter([xa[-1]], [xb[-1]], color=color, s=25, zorder=6)
                except KeyError:
                    pass
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7,
                          facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL)

        title_corps = corps_sel if corps_sel != "Tous les corps" else f"{len(indices)} corps"
        ax.set_title(f"Densite spatiale — {plan} — {title_corps}",
                     color=TEXT_COL, fontsize=10, pad=6)
        ax.set_xlabel(lbl_a, fontsize=9)
        ax.set_ylabel(lbl_b, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

        self.figure.tight_layout(pad=1.2)
        self.canvas.draw()


# ─────────────────────────────────────────────────────────
#  Onglet Événements
# ─────────────────────────────────────────────────────────
class EventTab(QWidget):

    # Signal émis quand l'utilisateur veut naviguer vers un temps
    navigate_to_time = pyqtSignal(float)   # t en années

    def __init__(self):
        super().__init__()
        self.data:     "SimData"      = None
        self.detector: EventDetector  = None
        self.events:   list[dict]     = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Barre de contrôles ────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        lbl_seuil = QLabel("Seuil approche :")
        lbl_seuil.setToolTip("Distance en mètres en-dessous de laquelle une approche est signalée")
        self.sp_seuil = QDoubleSpinBox()
        self.sp_seuil.setRange(1.0, 1e15)
        self.sp_seuil.setDecimals(0)
        self.sp_seuil.setValue(0.1 * UA)
        self.sp_seuil.setSuffix(" m")
        self.sp_seuil.setSingleStep(1e9)
        self.sp_seuil.setToolTip("Distance en mètres en-dessous de laquelle une approche est signalée")

        lbl_e = QLabel("Seuil éjection (e) :")
        lbl_e.setToolTip("Excentricité à partir de laquelle une éjection est signalée (1.0 = parabole)")
        self.sp_e_seuil = QDoubleSpinBox()
        self.sp_e_seuil.setRange(0.5, 1.5)
        self.sp_e_seuil.setDecimals(3)
        self.sp_e_seuil.setValue(0.95)
        self.sp_e_seuil.setSingleStep(0.01)
        self.sp_e_seuil.setToolTip("Excentricité à partir de laquelle une éjection est signalée (1.0 = parabole)")

        self.cb_filtre = QComboBox()
        self.cb_filtre.addItems(["Tous les types",
                                  "Approche proche",
                                  "Distance minimale",
                                  "Éjection possible",
                                  "Croisement d'orbites"])
        self.cb_filtre.currentIndexChanged.connect(self._apply_filter)

        btn_detect = QPushButton("🔍  Détecter les événements")
        btn_detect.setObjectName("primary")
        btn_detect.clicked.connect(self._run_detection)

        btn_csv = QPushButton("📄 CSV")
        btn_csv.setToolTip("Exporter le tableau en CSV")
        btn_csv.clicked.connect(self._export_csv)

        ctrl.addWidget(lbl_seuil);      ctrl.addWidget(self.sp_seuil)
        ctrl.addSpacing(16)
        ctrl.addWidget(lbl_e);          ctrl.addWidget(self.sp_e_seuil)
        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Filtre :")); ctrl.addWidget(self.cb_filtre)
        ctrl.addStretch()
        ctrl.addWidget(btn_csv)
        ctrl.addSpacing(8)
        ctrl.addWidget(btn_detect)
        layout.addLayout(ctrl)

        # ── Compteurs par type ────────────────────────────────────────────────
        self.lbl_counts = QLabel("")
        self.lbl_counts.setObjectName("muted")
        layout.addWidget(self.lbl_counts)

        # ── Tableau des événements ────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["⏱ Temps (ans)", "Type", "Sévérité", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._on_row_double_click)
        layout.addWidget(self.table)

        # ── Graphe de timeline ────────────────────────────────────────────────
        self.figure_tl = Figure(facecolor=FIG_PANEL, figsize=(12, 2.2))
        self.canvas_tl = FigureCanvas(self.figure_tl)
        self.canvas_tl.setMaximumHeight(160)
        self.canvas_tl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.canvas_tl)

        hint = QLabel(
            "Double-cliquer sur un événement pour naviguer vers ce temps dans l'onglet Orbites.")
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

    # ── Logique ───────────────────────────────────────────────────────────────
    def load_data(self, data: "SimData"):
        self.data    = data
        self.events  = []
        self.table.setRowCount(0)
        self.lbl_counts.setText("Cliquez sur 'Détecter les événements' pour lancer l'analyse.")
        self.figure_tl.clear()
        self.canvas_tl.draw()

    def _run_detection(self):
        if self.data is None:
            return
        self.detector = EventDetector(self.data)
        self.events   = self.detector.detect_all(
            approche_seuil_m = self.sp_seuil.value(),
            ejection_e_seuil  = self.sp_e_seuil.value(),
        )
        self._apply_filter()
        self._draw_timeline()
        self._update_counts()

    def _apply_filter(self):
        filtre = self.cb_filtre.currentText()
        shown  = (self.events if filtre == "Tous les types"
                  else [e for e in self.events if e["type"] == filtre])
        self._populate_table(shown)

    def _populate_table(self, events: list[dict]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(events))
        sev_icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
        sev_colors = EventDetector.COLORS
        for r, ev in enumerate(events):
            t_item = QTableWidgetItem(f"{ev['t_ans']:.2f}")
            t_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 0, t_item)
            self.table.setItem(r, 1, QTableWidgetItem(ev["type"]))
            sev = ev["severity"]
            sev_item = QTableWidgetItem(f"{sev_icons.get(sev, '')} {sev}")
            sev_item.setForeground(QColor(sev_colors.get(sev, TEXT_COL)))
            self.table.setItem(r, 2, sev_item)
            self.table.setItem(r, 3, QTableWidgetItem(ev["desc"]))
            # Colorier la ligne selon la sévérité
            row_color = QColor(sev_colors.get(sev, TEXT_COL))
            row_color.setAlpha(20)
            for c in range(4):
                item = self.table.item(r, c)
                if item:
                    item.setBackground(row_color)
                    item.setData(Qt.UserRole, ev["t_ans"])   # stocker le temps pour navigation
        self.table.setSortingEnabled(True)

    def _draw_timeline(self):
        """Dessine une frise chronologique des événements."""
        if not self.events or self.data is None:
            return
        self.figure_tl.clear()
        self.figure_tl.patch.set_facecolor(FIG_PANEL)
        ax = self.figure_tl.add_subplot(111)
        apply_dark_axes(ax)

        duree = self.data.duree_ans()
        ax.set_xlim(0, duree)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Temps (ans)", fontsize=8)
        ax.set_yticks([])
        ax.set_title("Frise chronologique des événements", color=TEXT_COL, fontsize=9, pad=3)

        # Fond par type
        type_y = {
            "Approche proche":      2.0,
            "Distance minimale":    1.4,
            "Éjection possible":    0.8,
            "Croisement d'orbites": 0.2,
        }
        # Labels à gauche
        for label, y in type_y.items():
            ax.text(-duree * 0.01, y, label, color=MUTED, fontsize=7,
                    ha="right", va="center", clip_on=False)

        for ev in self.events:
            y     = type_y.get(ev["type"], 1.0)
            color = EventDetector.COLORS[ev["severity"]]
            ax.scatter([ev["t_ans"]], [y], color=color, s=40, zorder=5,
                       marker="|" if ev["severity"] == "info" else "v",
                       linewidths=1.5)

        ax.set_xlim(-duree * 0.02, duree * 1.02)
        self.figure_tl.tight_layout(pad=0.5)
        self.canvas_tl.draw()

    def _update_counts(self):
        from collections import Counter
        counts = Counter(e["type"] for e in self.events)
        parts  = [f"{v}× {k}" for k, v in counts.items()]
        self.lbl_counts.setText(
            f"{len(self.events)} événements détectés   —   " + "   |   ".join(parts)
        )

    def _on_row_double_click(self, index):
        """Double-clic → émet le signal pour naviguer dans l'onglet Orbites."""
        item = self.table.item(index.row(), 0)
        if item is None:
            return
        t_ans = item.data(Qt.UserRole)
        if t_ans is not None:
            self.navigate_to_time.emit(float(t_ans))

    def _export_csv(self):
        if not self.events:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter CSV", "evenements.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["temps_ans", "type", "severite", "description"])
            for ev in self.events:
                w.writerow([f"{ev['t_ans']:.4f}", ev["type"], ev["severity"], ev["desc"]])



class MainWindow(QMainWindow):

    def __init__(self, initial_file: str = ""):
        super().__init__()
        self.setWindowTitle("Astralis — Simulation Orbitale N-corps")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Barre de chargement ──────────────────────────
        top_bar = QHBoxLayout()

        self.lbl_file = QLabel("Aucun fichier chargé")
        self.lbl_file.setObjectName("muted")
        self.lbl_file.setMaximumWidth(600)

        btn_open = QPushButton("📂  Ouvrir un fichier…")
        btn_open.clicked.connect(self._open_file)

        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("muted")

        top_bar.addWidget(btn_open)
        top_bar.addWidget(self.lbl_file)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_info)
        layout.addLayout(top_bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        layout.addWidget(sep)

        # ── Onglets ──────────────────────────────────────
        self.tabs = QTabWidget()
        self.tab_orbite   = OrbiteTab()
        self.tab_distance = DistanceTab()
        self.tab_energie  = EnergieTab()
        self.tab_sim      = SimulationTab()

        self.tab_vitesse   = VitesseTab()
        self.tab_stabilite = StabiliteTab()
        self.tab_resume    = ResumeTab()
        self.tab_phase     = PhaseTab()
        self.tab_densite   = DensiteTab()

        self.tabs.addTab(self.tab_orbite,   "Orbites")
        self.tabs.addTab(self.tab_distance, "Distances")
        self.tabs.addTab(self.tab_energie,  "Energie")
        self.tabs.addTab(self.tab_vitesse,  "Vitesses")
        self.tabs.addTab(self.tab_stabilite,"Stabilite")
        self.tabs.addTab(self.tab_phase,    "Phase")
        self.tabs.addTab(self.tab_densite,  "Densite")
        self.tabs.addTab(self.tab_resume,   "Resume")
        self.tabs.addTab(self.tab_sim,      "Simulation")

        layout.addWidget(self.tabs)

        # ── Status bar ───────────────────────────────────
        self.statusBar().setStyleSheet(f"background: {PANEL_BG}; color: {MUTED};")
        self.statusBar().showMessage("Astralis · Prêt — Ouvrez un fichier .parquet pour commencer.")

        # Chargement initial
        if initial_file and os.path.exists(initial_file):
            self._load_file(initial_file)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une simulation", _outputs_dir(),
            "Parquet (*.parquet);;Tous (*.*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        self.statusBar().showMessage(f"Chargement : {path}…")
        QApplication.processEvents()
        try:
            data = SimData.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Erreur de chargement", str(e))
            self.statusBar().showMessage("Erreur de chargement.")
            return

        self.lbl_file.setText(os.path.basename(path))
        n_pts = len(data.t_jours)
        duree = data.duree_ans()
        info = (f"{data.n_bodies} corps  •  {n_pts:,} points  •  "
                f"{duree:.1f} ans  •  dérive E = {data.metrics.get('derive_e', float('nan')):.2e} %")
        self.lbl_info.setText(info)

        import traceback as _tb
        _log = os.path.join(OUTPUTS_DIR, "crash.log")
        for _name, _tab in [
            ("OrbiteTab",    self.tab_orbite),
            ("DistanceTab",  self.tab_distance),
            ("EnergieTab",   self.tab_energie),
            ("VitesseTab",   self.tab_vitesse),
            ("StabiliteTab", self.tab_stabilite),
            ("PhaseTab",     self.tab_phase),
            ("DensiteTab",   self.tab_densite),
            ("ResumeTab",    self.tab_resume),
        ]:
            try:
                with open(_log, "a", encoding="utf-8") as _f:
                    _f.write(f"→ {_name}.load_data()\n")
                _tab.load_data(data)
            except Exception as _e:
                _err = _tb.format_exc()
                with open(_log, "a", encoding="utf-8") as _f:
                    _f.write(f"CRASH dans {_name}:\n{_err}\n")
                QMessageBox.critical(self, f"Erreur {_name}", str(_e))
                return

        # Pré-remplir le script dans l'onglet sim si possible
        script_candidate = _default_moteur_script()
        if _moteur_script_ok(script_candidate):
            self.tab_sim.le_script.setText(script_candidate)
            self.tab_sim.le_cwd.setText(APP_DIR)

        self.statusBar().showMessage(f"✅  Chargé : {os.path.basename(path)}  |  {info}")


# ─────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
#  Splash Screen cinématique Astralis  — ~20 s, 7 phases
# ─────────────────────────────────────────────────────────

class AstralisSplash(QWidget):
    """
    Cinématique d'intro Astralis — ~20 secondes, style grand écran.

    7 phases enchaînées (ticks à ~55 fps, intervalle 18 ms) :
      P1  Nuit noire → étoiles qui naissent une à une         (~2.5 s)
      P2  Nébuleuse / halo cosmique qui pulse et s'étend      (~2.5 s)
      P3  Orbites elliptiques qui se tracent lentement        (~3.0 s)
      P4  Logo émerge de l'obscurité (flash de lumière)       (~2.5 s)
      P5  "ASTRALIS" — lettres qui tombent du vide, espacées  (~3.0 s)
      P6  Sous-titre + ligne décorative qui se tire           (~2.5 s)
      P7  Barre de chargement + fade global vers blanc        (~3.5 s)
    Pause finale ~1.5 s puis fermeture.
    Clic pour passer à tout moment.
    """
    finished = pyqtSignal()

    _LOGO_NAMES = ["astralis_logo.png", "astralis_logo.jpg",
                   "logo.png", "logo.jpg", "icon.png"]

    # ── Durées des phases (ticks) ──────────────────────
    # 55 fps → 1 tick ≈ 18 ms
    _DUR = [138, 138, 165, 138, 165, 138, 193]   # durées en ticks
    # cumuls :  138  276  441  579  744  882  1075  ≈ 19.4 s

    def __init__(self, app_icon: "QIcon"):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        from PyQt5.QtWidgets import QDesktopWidget
        import random, math as _math

        screen = QDesktopWidget().screenGeometry()
        W, H = screen.width(), screen.height()
        self.resize(W, H)
        self.move(screen.left(), screen.top())
        self.showFullScreen()

        # ── Catalogue d'étoiles ─────────────────────────
        rng = random.Random(42)
        # (x, y, rayon, alpha_max, vitesse_scintillement, phase_scintill)
        self._stars = [
            (rng.randint(0, W), rng.randint(0, H),
             rng.uniform(0.4, 3.2), rng.uniform(0.25, 1.0),
             rng.uniform(0.015, 0.07), rng.uniform(0, 6.28))
            for _ in range(420)
        ]
        # Étoiles "filantes" générées aléatoirement (x1,y1,x2,y2,life,speed)
        self._meteors = []
        self._next_meteor = rng.randint(60, 140)

        # ── Nébuleuses : nuages colorés fixes ───────────
        self._nebulae = [
            (int(W * 0.18), int(H * 0.22), int(W * 0.38), 30, 205, 225, 255),
            (int(W * 0.72), int(H * 0.15), int(W * 0.32), 25, 180, 200, 240),
            (int(W * 0.50), int(H * 0.78), int(W * 0.42), 20, 190, 210, 245),
        ]

        # ── Points d'orbite pré-calculés ────────────────
        cx, cy = W // 2, H // 2
        a1, b1 = int(W * 0.22), int(H * 0.12)   # orbite externe
        a2, b2 = int(W * 0.14), int(H * 0.07)   # orbite interne
        TILT = -0.35  # radians
        orbit_pts1, orbit_pts2 = [], []
        for i in range(361):
            t = _math.radians(i)
            rx = a1 * _math.cos(t)
            ry = b1 * _math.sin(t)
            orbit_pts1.append((
                cx + rx * _math.cos(TILT) - ry * _math.sin(TILT),
                cy + rx * _math.sin(TILT) + ry * _math.cos(TILT),
            ))
            rx = a2 * _math.cos(t + 1.1)
            ry = b2 * _math.sin(t + 1.1)
            orbit_pts2.append((
                cx + rx * _math.cos(TILT + 0.9) - ry * _math.sin(TILT + 0.9),
                cy + rx * _math.sin(TILT + 0.9) + ry * _math.cos(TILT + 0.9),
            ))
        self._orbit1 = orbit_pts1
        self._orbit2 = orbit_pts2

        # ── Compteur de ticks global ─────────────────────
        self._t = 0          # ticks absolus
        self._done = False
        self._skip = False   # flag "clic pour passer"

        # ── Cumulatif des phases ─────────────────────────
        cum = 0
        self._phase_start = []
        for d in self._DUR:
            self._phase_start.append(cum)
            cum += d
        self._total_ticks = cum

        # ── Chargement logo ─────────────────────────────
        logo_size = int(min(W, H) * 0.28)
        self._logo_pix = None
        for name in self._LOGO_NAMES:
            for search_dir in (_BUNDLE_DIR, APP_DIR):
                candidate = os.path.join(search_dir, name)
                if os.path.isfile(candidate):
                    from PyQt5.QtGui import QPixmap
                    pix = QPixmap(candidate)
                    if not pix.isNull():
                        self._logo_pix = pix.scaled(
                            logo_size, logo_size,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
                    break
            if self._logo_pix:
                break

        self._timer = QTimer(self)
        self._timer.setInterval(18)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Helpers ─────────────────────────────────────────
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Courbe d'accélération douce (cubic)."""
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    @staticmethod
    def _ease_out(t: float) -> float:
        """Décélération pure."""
        t = max(0.0, min(1.0, t))
        return 1.0 - (1.0 - t) ** 3

    def _phase_prog(self, ph: int) -> float:
        """Progression [0..1] de la phase ph (0-indexé)."""
        start = self._phase_start[ph]
        dur   = self._DUR[ph]
        raw   = (self._t - start) / dur
        return max(0.0, min(1.0, raw))

    # ── Tick ────────────────────────────────────────────
    def _tick(self):
        import random, math as _math

        if self._skip:
            # Raccourci rapide : avancer directement à la fin
            self._t = self._total_ticks + 1
        else:
            self._t += 1

        # Gestion des étoiles filantes
        rng = random.Random(self._t * 7 + 13)
        self._next_meteor -= 1
        if self._next_meteor <= 0:
            W, H = self.width(), self.height()
            x1 = rng.randint(0, W)
            y1 = rng.randint(0, int(H * 0.6))
            length = rng.randint(int(W * 0.04), int(W * 0.12))
            angle  = rng.uniform(0.3, 0.7)
            self._meteors.append([x1, y1,
                                   x1 + int(length * _math.cos(angle)),
                                   y1 + int(length * _math.sin(angle)),
                                   0, rng.uniform(0.04, 0.08)])
            self._next_meteor = rng.randint(80, 200)

        self._meteors = [m for m in self._meteors if m[4] < 1.0]
        for m in self._meteors:
            m[4] = min(1.0, m[4] + m[5])

        if self._t >= self._total_ticks and not self._done:
            self._done = True
            QTimer.singleShot(1500, self._close_splash)

        self.update()

    def _close_splash(self):
        self._timer.stop()
        self.close()
        self.finished.emit()

    # ── Rendu ─────────────────────────────────────────────
    def paintEvent(self, _event):   # noqa: C901
        from PyQt5.QtGui import (QPainter, QBrush, QLinearGradient,
                                  QRadialGradient, QColor, QFont,
                                  QPen, QFontMetrics, QPolygonF)
        from PyQt5.QtCore import QRectF, QPointF
        import math as _math

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        W, H = self.width(), self.height()
        cx, cy = W // 2, H // 2
        t = self._t

        # ── Progressions des 7 phases ────────────────────
        p1 = self._ease_out(self._phase_prog(0))   # étoiles
        p2 = self._ease_in_out(self._phase_prog(1))  # nébuleuse
        p3 = self._ease_in_out(self._phase_prog(2))  # orbites
        p4 = self._ease_out(self._phase_prog(3))   # logo
        p5 = self._phase_prog(4)                   # titre (linéaire pour letter-drop)
        p6 = self._ease_in_out(self._phase_prog(5))  # sous-titre
        p7 = self._ease_in_out(self._phase_prog(6))  # barre + fade out

        # ════════════════════════════════════════════════
        # FOND SPATIAL
        # ════════════════════════════════════════════════
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0.0,  QColor(4,   5,  10))
        bg.setColorAt(0.45, QColor(7,  12,  22))
        bg.setColorAt(1.0,  QColor(3,   4,   8))
        p.fillRect(0, 0, W, H, QBrush(bg))

        # ── Vignette ────────────────────────────────────
        vig = QRadialGradient(cx, cy, max(W, H) * 0.75)
        vig.setColorAt(0.0, QColor(0, 0, 0, 0))
        vig.setColorAt(1.0, QColor(0, 0, 0, 175))
        p.setBrush(QBrush(vig))
        p.setPen(Qt.NoPen)
        p.drawRect(0, 0, W, H)

        # ════════════════════════════════════════════════
        # PHASE 1 — ÉTOILES QUI NAISSENT
        # ════════════════════════════════════════════════
        p.setPen(Qt.NoPen)
        n_visible = int(len(self._stars) * p1)
        for i, (sx, sy, sr, sa, spd, sph) in enumerate(self._stars[:n_visible]):
            flicker = 0.55 + 0.45 * _math.sin(t * spd + sph)
            alpha = int(sa * 210 * flicker * min(1.0, (i / max(1, n_visible)) * 3))
            c = int(205 + sa * 50)
            p.setBrush(QBrush(QColor(c, c, min(255, c + 25), alpha)))
            p.drawEllipse(QPointF(sx, sy), sr, sr)

        # ── Étoiles filantes ────────────────────────────
        if p1 > 0.3:
            for mx1, my1, mx2, my2, mlife, _ in self._meteors:
                fade = 1.0 - abs(mlife - 0.4) / 0.6
                alpha = int(fade * 200 * p1)
                if alpha > 5:
                    gpen = QPen(QColor(220, 230, 255, alpha))
                    gpen.setWidthF(1.5)
                    p.setPen(gpen)
                    p.setBrush(Qt.NoBrush)
                    p.drawLine(QPointF(mx1, my1), QPointF(mx2, my2))
            p.setPen(Qt.NoPen)

        # ════════════════════════════════════════════════
        # PHASE 2 — NÉBULEUSE COSMIQUE
        # ════════════════════════════════════════════════
        if p2 > 0:
            pulse = 0.92 + 0.08 * _math.sin(t * 0.022)
            for nx, ny, nr, na, r, g, b in self._nebulae:
                neb = QRadialGradient(nx, ny, int(nr * pulse * p2))
                alpha = int(na * p2 * 0.55)
                neb.setColorAt(0.0, QColor(r, g, b, alpha))
                neb.setColorAt(0.5, QColor(r - 20, g - 15, b, alpha // 3))
                neb.setColorAt(1.0, QColor(r - 30, g - 25, b - 10, 0))
                p.setBrush(QBrush(neb))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(nx, ny), nr * pulse * p2, nr * pulse * p2 * 0.6)

            # Halo central
            halo_r = int(min(W, H) * 0.18 * p2)
            halo_pulse = 0.95 + 0.05 * _math.sin(t * 0.035)
            halo = QRadialGradient(cx, cy, int(halo_r * halo_pulse))
            ha = int(28 * p2)
            halo.setColorAt(0.0, QColor(160, 185, 220, ha))
            halo.setColorAt(0.5, QColor(120, 150, 200, ha // 2))
            halo.setColorAt(1.0, QColor(80, 110, 160, 0))
            p.setBrush(QBrush(halo))
            p.drawEllipse(QPointF(cx, cy), halo_r * halo_pulse, halo_r * halo_pulse)

        # ════════════════════════════════════════════════
        # PHASE 3 — ORBITES QUI SE TRACENT
        # ════════════════════════════════════════════════
        if p3 > 0:
            n1 = int(360 * p3)
            n2 = int(360 * p3)
            orb_alpha1 = int(90 * p3)
            orb_alpha2 = int(65 * p3)

            # Orbite 1
            if n1 > 1:
                for i in range(n1 - 1):
                    seg_prog = i / 360.0
                    fade = max(0, min(1.0, (p3 - seg_prog) * 8))
                    alpha = int(orb_alpha1 * fade)
                    if alpha < 4:
                        continue
                    x0, y0 = self._orbit1[i]
                    x1_, y1_ = self._orbit1[i + 1]
                    pen_o = QPen(QColor(190, 205, 225, alpha))
                    pen_o.setWidthF(1.2)
                    p.setPen(pen_o)
                    p.setBrush(Qt.NoBrush)
                    p.drawLine(QPointF(x0, y0), QPointF(x1_, y1_))

            # Orbite 2
            if n2 > 1:
                for i in range(n2 - 1):
                    seg_prog = i / 360.0
                    fade = max(0, min(1.0, (p3 - seg_prog) * 8))
                    alpha = int(orb_alpha2 * fade)
                    if alpha < 4:
                        continue
                    x0, y0 = self._orbit2[i]
                    x1_, y1_ = self._orbit2[i + 1]
                    pen_o = QPen(QColor(170, 190, 215, alpha))
                    pen_o.setWidthF(0.8)
                    p.setPen(pen_o)
                    p.drawLine(QPointF(x0, y0), QPointF(x1_, y1_))

            # Planètes sur les orbites
            p.setPen(Qt.NoPen)
            if p3 > 0.4:
                planet_alpha = int(min(1.0, (p3 - 0.4) / 0.3) * 200)
                angle1 = t * 0.012
                px1 = cx + int(self._orbit1[0][0] - cx)
                py1 = cy + int(self._orbit1[0][1] - cy)
                idx1 = int((angle1 % (2 * _math.pi)) / (2 * _math.pi) * 360) % 360
                px1, py1 = self._orbit1[idx1]
                grad_p = QRadialGradient(px1, py1, 7)
                grad_p.setColorAt(0.0, QColor(230, 240, 255, planet_alpha))
                grad_p.setColorAt(1.0, QColor(150, 175, 210, 0))
                p.setBrush(QBrush(grad_p))
                p.drawEllipse(QPointF(px1, py1), 6, 6)

                angle2 = t * 0.021 + 2.8
                idx2 = int((angle2 % (2 * _math.pi)) / (2 * _math.pi) * 360) % 360
                px2, py2 = self._orbit2[idx2]
                grad_p2 = QRadialGradient(px2, py2, 5)
                grad_p2.setColorAt(0.0, QColor(210, 225, 245, planet_alpha))
                grad_p2.setColorAt(1.0, QColor(140, 165, 200, 0))
                p.setBrush(QBrush(grad_p2))
                p.drawEllipse(QPointF(px2, py2), 4, 4)

        # ════════════════════════════════════════════════
        # PHASE 4 — LOGO ÉMERGE DE L'OBSCURITÉ
        # ════════════════════════════════════════════════
        logo_size = int(min(W, H) * 0.28)
        ly = cy - int(H * 0.20)

        if p4 > 0:
            # Flash de lumière au moment où le logo apparaît (p4 autour de 0.25)
            flash_t = max(0.0, min(1.0, p4 * 4))            # 0→1 sur 25% de la phase
            flash_decay = max(0.0, 1.0 - (p4 - 0.25) * 4) if p4 > 0.25 else flash_t
            flash_alpha = int(flash_decay * 60)
            if flash_alpha > 2:
                flash_r = int(logo_size * 1.8)
                flash = QRadialGradient(cx, ly + logo_size // 2, flash_r)
                flash.setColorAt(0.0, QColor(220, 230, 245, flash_alpha))
                flash.setColorAt(0.4, QColor(180, 200, 230, flash_alpha // 3))
                flash.setColorAt(1.0, QColor(120, 150, 200, 0))
                p.setBrush(QBrush(flash))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(cx, ly + logo_size // 2),
                              flash_r, flash_r)

            # Halo permanent derrière le logo
            halo_r2 = int(logo_size * 0.72 * p4)
            halo2 = QRadialGradient(cx, ly + logo_size // 2, halo_r2)
            ha2 = int(40 * p4)
            halo2.setColorAt(0.0, QColor(185, 200, 220, ha2))
            halo2.setColorAt(0.5, QColor(145, 165, 195, ha2 // 3))
            halo2.setColorAt(1.0, QColor(100, 125, 165, 0))
            p.setBrush(QBrush(halo2))
            p.drawEllipse(QPointF(cx, ly + logo_size // 2), halo_r2, halo_r2)

            # Logo
            p.setOpacity(p4)
            if self._logo_pix and not self._logo_pix.isNull():
                lw = self._logo_pix.width()
                lh = self._logo_pix.height()
                # Légère translation vers le haut pendant l'émergence
                drift_y = int((1.0 - p4) * 18)
                p.drawPixmap(cx - lw // 2, ly + (logo_size - lh) // 2 + drift_y,
                             self._logo_pix)
            else:
                # Logo de secours
                r_orb = int(logo_size * 0.35)
                from PyQt5.QtGui import QRadialGradient as RG
                grad_fb = RG(cx, ly + logo_size // 2, r_orb * 0.4)
                grad_fb.setColorAt(0.0, QColor("#e0e8f0"))
                grad_fb.setColorAt(1.0, QColor("#8090a8"))
                p.setBrush(QBrush(grad_fb))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(cx, ly + logo_size // 2),
                              r_orb * 0.4, r_orb * 0.4)
                orb_pen = QPen(QColor(200, 212, 228, 200))
                orb_pen.setWidthF(1.8)
                p.setPen(orb_pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(cx, ly + logo_size // 2),
                              r_orb, r_orb * 0.48)
            p.setOpacity(1.0)

        # ════════════════════════════════════════════════
        # PHASE 5 — "ASTRALIS" LETTRE PAR LETTRE
        # ════════════════════════════════════════════════
        TITLE = "ASTRALIS"
        N_LETTERS = len(TITLE)
        title_y = ly + logo_size + int(H * 0.038)

        if p5 > 0:
            font_sz   = max(30, min(78, int(W / 18)))
            font_title = QFont("Segoe UI", font_sz, QFont.Light)
            font_title.setLetterSpacing(QFont.PercentageSpacing, 165)
            fm = QFontMetrics(font_title)
            full_w = fm.horizontalAdvance(TITLE)
            title_h = fm.height()
            p.setFont(font_title)

            # Calculer la position X de chaque lettre
            x_cursor = cx - full_w // 2
            letter_xs = []
            for ch in TITLE:
                letter_xs.append(x_cursor)
                x_cursor += fm.horizontalAdvance(ch)

            for i, (ch, lx) in enumerate(zip(TITLE, letter_xs)):
                letter_prog = max(0.0, min(1.0, (p5 * (N_LETTERS + 1) - i - 0.3)))
                letter_prog = self._ease_out(letter_prog)
                if letter_prog <= 0:
                    continue
                drop_offset = int((1.0 - letter_prog) * 18)
                alpha = int(letter_prog * 255)
                baseline = title_y + fm.ascent() + drop_offset

                glow_a = int(alpha * 0.2)
                p.setPen(QColor(0, 229, 255, glow_a))
                p.drawText(QPointF(lx, baseline + 1), ch)
                p.setPen(QColor(248, 250, 252, alpha))
                p.drawText(QPointF(lx, baseline), ch)

            # Ligne décorative sous le titre
            if p5 > 0.85:
                line_prog = self._ease_out((p5 - 0.85) / 0.15)
                line_half = int(full_w * 0.44 * line_prog)
                line_alpha = int(line_prog * 115)
                if line_half > 2:
                    grad_line = QLinearGradient(cx - line_half, 0, cx + line_half, 0)
                    grad_line.setColorAt(0.0, QColor(175, 192, 215, 0))
                    grad_line.setColorAt(0.5, QColor(175, 192, 215, line_alpha))
                    grad_line.setColorAt(1.0, QColor(175, 192, 215, 0))
                    p.setBrush(QBrush(grad_line))
                    p.setPen(Qt.NoPen)
                    line_y = title_y + title_h + int(H * 0.012)
                    p.drawRect(QRectF(cx - line_half, line_y, line_half * 2, 1.5))

        # ════════════════════════════════════════════════
        # PHASE 6 — SOUS-TITRE
        # ════════════════════════════════════════════════
        sub_y = title_y + (max(30, min(78, int(W / 18))) + 12) + int(H * 0.032)

        if p6 > 0:
            sub_sz = max(11, min(22, int(W / 82)))
            font_sub = QFont("Segoe UI", sub_sz, QFont.Light)
            font_sub.setLetterSpacing(QFont.PercentageSpacing, 130)
            p.setFont(font_sub)
            sub_alpha = int(p6 * 145)
            p.setPen(QColor(105, 122, 148, sub_alpha))
            fm_s = QFontMetrics(font_sub)
            subtitle = "Moteur de simulation orbitale N-corps"
            p.drawText(QRectF(0, sub_y, W, fm_s.height() + 4),
                       Qt.AlignHCenter, subtitle)

        # ════════════════════════════════════════════════
        # PHASE 7 — BARRE DE CHARGEMENT
        # ════════════════════════════════════════════════
        font_sz_ = max(30, min(78, int(W / 18)))
        bar_y = sub_y + int(H * 0.06) + int(max(11, min(22, int(W / 82))) * 1.5)

        if p7 > 0:
            bar_w   = int(W * 0.44)
            bar_h   = max(4, int(H * 0.006))
            bar_x   = cx - bar_w // 2
            fill_w  = int(bar_w * p7)

            # Fond
            p.setBrush(QBrush(QColor(255, 255, 255, 10)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h),
                              bar_h // 2, bar_h // 2)

            # Remplissage avec gradient
            if fill_w > 2:
                grad_bar = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
                grad_bar.setColorAt(0.0, QColor(40,  52,  75, 190))
                grad_bar.setColorAt(0.5, QColor(130, 148, 172, 220))
                grad_bar.setColorAt(1.0, QColor(205, 218, 235, 255))
                p.setBrush(QBrush(grad_bar))
                p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h),
                                  bar_h // 2, bar_h // 2)

                # Lueur à la pointe
                tip    = bar_x + fill_w
                glow_r = bar_h * 4.5
                glow   = QRadialGradient(tip, bar_y + bar_h / 2, glow_r)
                glow.setColorAt(0.0, QColor(215, 230, 252, 160))
                glow.setColorAt(1.0, QColor(215, 230, 252, 0))
                p.setBrush(QBrush(glow))
                p.drawEllipse(QPointF(tip, bar_y + bar_h / 2), glow_r, glow_r)

            # Label "Initialisation..."
            if p7 > 0.08:
                load_sz = max(9, min(15, int(W / 125)))
                font_load = QFont("Segoe UI", load_sz, QFont.Light)
                p.setFont(font_load)
                label_alpha = int(min(1.0, p7 * 5) * 105)
                p.setPen(QColor(95, 112, 138, label_alpha))
                p.drawText(QRectF(0, bar_y + bar_h + int(H * 0.012), W, load_sz * 2),
                           Qt.AlignHCenter, "Initialisation...")

        # ════════════════════════════════════════════════
        # HINT "Appuyer pour passer" — discret, dès phase 2
        # ════════════════════════════════════════════════
        hint_appear = max(0.0, min(1.0, (p2 - 0.5) / 0.5))
        if hint_appear > 0 and not self._done:
            hint_sz = max(8, min(13, int(W / 155)))
            font_hint = QFont("Segoe UI", hint_sz, QFont.Light)
            p.setFont(font_hint)
            hint_alpha = int(hint_appear * 48)
            p.setPen(QColor(125, 140, 162, hint_alpha))
            p.drawText(QRectF(0, H - int(H * 0.048), W, hint_sz * 2 + 4),
                       Qt.AlignHCenter, "Appuyer pour passer")

        # ════════════════════════════════════════════════
        # FADE OUT final (fin de phase 7)
        # ════════════════════════════════════════════════
        if p7 > 0.85:
            fade_alpha = int(((p7 - 0.85) / 0.15) * 255)
            p.fillRect(0, 0, W, H, QColor(5, 7, 14, fade_alpha))

        p.end()

    def mousePressEvent(self, _event):
        """Clic pour passer la cinématique."""
        if not self._done:
            self._skip = True
            self.update()


# ─────────────────────────────────────────────────────────

def _build_app_icon() -> "QIcon":
    """
    Construit l'icône de la fenêtre/barre des tâches.
    Priorité : astralis_logo.png (ou variantes) dans APP_DIR,
    sinon icône géométrique de secours.
    """
    from PyQt5.QtGui import (QPixmap, QPainter, QBrush, QPen,
                              QRadialGradient, QIcon)
    from PyQt5.QtCore import QRectF

    # 1. Essayer de charger l'image fournie par l'utilisateur
    logo_names = ["astralis_logo.png", "astralis_logo.jpg",
                  "logo.png", "logo.jpg", "icon.png", "icon.ico"]
    for name in logo_names:
        candidate = os.path.join(APP_DIR, name)
        if os.path.isfile(candidate):
            pix = QPixmap(candidate)
            if not pix.isNull():
                # Générer plusieurs tailles pour Windows
                icon = QIcon()
                for sz in (16, 32, 48, 64, 128, 256):
                    scaled = pix.scaled(sz, sz,
                                        Qt.KeepAspectRatioByExpanding,
                                        Qt.SmoothTransformation)
                    icon.addPixmap(scaled)
                return icon

    # 2. Icône géométrique de secours (orbite argentée)
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillRect(0, 0, 64, 64, QColor(DARK_BG))
    pen = QPen(QColor(ACCENT), 1.5)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(6, 20, 52, 24))
    grad = QRadialGradient(32, 32, 10)
    grad.setColorAt(0.0, QColor(TEXT_COL))
    grad.setColorAt(1.0, QColor(ACCENT2))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QRectF(22, 22, 20, 20))
    p.setBrush(QBrush(QColor(ACCENT)))
    p.drawEllipse(QRectF(50, 29, 8, 8))
    p.end()
    return QIcon(pix)


def main():
    parser = argparse.ArgumentParser(description="Astralis — Dashboard simulation orbitale")
    parser.add_argument("--file",   default="", help="Fichier .parquet a charger au demarrage")
    parser.add_argument("--splash", action="store_true",
                        help="Lance depuis splash.py : signale quand pret via .ready_flag")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    # ── Icône application ────────────────────────────────
    app_icon = _build_app_icon()
    app.setWindowIcon(app_icon)

    # Palette sombre pour les widgets natifs
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(DARK_BG))
    pal.setColor(QPalette.WindowText,      QColor(TEXT_COL))
    pal.setColor(QPalette.Base,            QColor(CARD_BG))
    pal.setColor(QPalette.AlternateBase,   QColor(PANEL_BG))
    pal.setColor(QPalette.Text,            QColor(TEXT_COL))
    pal.setColor(QPalette.Button,          QColor(CARD_BG))
    pal.setColor(QPalette.ButtonText,      QColor(TEXT_COL))
    pal.setColor(QPalette.Highlight,       QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(DARK_BG))
    pal.setColor(QPalette.Mid,             QColor(BORDER))
    pal.setColor(QPalette.Dark,            QColor(PANEL_BG))
    app.setPalette(pal)

    # Windows : AppUserModelID
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Astralis.SimulationOrbitale.1")
    except Exception:
        pass

    # ── Construction de la fenetre principale ────────────
    win = MainWindow(initial_file=args.file)
    win.setWindowIcon(app_icon)

    if args.splash:
        # Lance depuis splash.py :
        # - fenetre cachee pendant le chargement
        # - on ecrit le flag quand tout est pret
        # - le splash se ferme et on affiche la fenetre
        READY_FLAG = os.path.join(APP_DIR, '.ready_flag')

        def _signal_ready():
            try:
                with open(READY_FLAG, 'w') as f:
                    f.write('ready')
            except Exception:
                pass
            win.showMaximized()
            win.raise_()
            win.activateWindow()

        # Ecrire le flag apres le premier rendu (app entierement chargee)
        QTimer.singleShot(100, _signal_ready)
    else:
        # Lancement direct sans splash : affichage immediat
        win.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err = traceback.format_exc()
        log = os.path.join(OUTPUTS_DIR, "crash.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write(err)
        try:
            from PyQt5.QtWidgets import QMessageBox
            app2 = QApplication.instance()
            if app2 is None:
                app2 = QApplication(sys.argv)
            QMessageBox.critical(None, "Erreur fatale", str(e) + f"\n\nDétails dans : {log}")
        except Exception:
            pass
        raise
