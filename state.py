"""
state.py — Conteneur SoA (Structure of Arrays) JIT-compilé pour l'état global
            du système N-corps.

Toutes les propriétés physiques sont stockées dans des tableaux contigus
(layout SoA) pour maximiser la vectorisation SIMD et l'efficacité du cache L1/L2.

Compatibilité : Numba @njit (nopython mode).
"""

import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

# ── Spécification des champs du jitclass ──────────────────────────────────────
_spec = [
    ('n',        int64),
    ('masses',   float64[:]),
    ('j2r2',     float64[:]),
    ('mdot',     float64[:]),
    ('pos',      float64[:, :]),
    ('vel',      float64[:, :]),
    ('accel',    float64[:, :]),
    ('t',        float64),
    ('t_comp',   float64),
    ('s',        float64),
    ('s_comp',   float64),
    ('U_ref',    float64),
    ('pos_comp', float64[:, :]),
    ('E0',       float64),
    # ── Rotation propre ──────────────────────────────────────────────────────
    # spin[i]  : vecteur moment angulaire propre S_i (kg·m²/s) — norme = I·ω
    #            direction = axe de rotation (normalisée dans forces.py)
    # k2[i]    : nombre de Love de degré 2 (sans dimension, 0 = corps rigide)
    #            mesure la déformabilité du corps sous les marées
    #            k2=0 → pas de couplage spin-orbite
    #            k2~0.3 → planète rocheuse (Terre : 0.30, Mars : 0.17)
    #            k2~0.5 → planète gazeuse (Jupiter : 0.49)
    ('spin',     float64[:, :]),
    ('spin_comp',float64[:, :]),
    ('k2',       float64[:]),
]


@jitclass(_spec)
class SystemState:
    """
    Conteneur SoA haute performance pour un système N-corps.

    • Tableaux continus layout [n, 3] pour positions/vitesses/accélérations.
    • Compensation de Kahan sur le temps et les positions pour annuler
      l'arrondi cumulé sur des millions d'itérations.
    • Évolution dynamique des masses via mdot[i] (kg/s).
    """

    def __init__(self, n):
        self.n        = n
        self.masses   = np.zeros(n,      dtype=np.float64)
        self.j2r2     = np.zeros(n,      dtype=np.float64)
        self.mdot     = np.zeros(n,      dtype=np.float64)
        self.pos      = np.zeros((n, 3), dtype=np.float64)
        self.vel      = np.zeros((n, 3), dtype=np.float64)
        self.accel    = np.zeros((n, 3), dtype=np.float64)
        self.pos_comp = np.zeros((n, 3), dtype=np.float64)
        self.spin      = np.zeros((n, 3), dtype=np.float64)   # axe Z par défaut
        self.spin_comp = np.zeros((n, 3), dtype=np.float64)
        self.k2        = np.zeros(n,      dtype=np.float64)
        self.t        = 0.0
        self.t_comp   = 0.0
        self.s        = 0.0
        self.s_comp   = 0.0
        self.U_ref    = 0.0
        self.E0       = 0.0

    def advance_time(self, dt):
        """Incrémente t (temps physique) avec compensation de Kahan."""
        y           = dt - self.t_comp
        tmp         = self.t + y
        self.t_comp = (tmp - self.t) - y
        self.t      = tmp

    def advance_fictitious_time(self, ds):
        """Incrémente s (temps fictif Sundman) avec compensation de Kahan."""
        y           = ds - self.s_comp
        tmp         = self.s + y
        self.s_comp = (tmp - self.s) - y
        self.s      = tmp

    def kahan_add_pos(self, i, dx, dy, dz):
        """Additionne (dx,dy,dz) à pos[i] avec compensation de Kahan."""
        # axe X
        y               = dx - self.pos_comp[i, 0]
        tmp             = self.pos[i, 0] + y
        self.pos_comp[i, 0] = (tmp - self.pos[i, 0]) - y
        self.pos[i, 0]  = tmp
        # axe Y
        y               = dy - self.pos_comp[i, 1]
        tmp             = self.pos[i, 1] + y
        self.pos_comp[i, 1] = (tmp - self.pos[i, 1]) - y
        self.pos[i, 1]  = tmp
        # axe Z
        y               = dz - self.pos_comp[i, 2]
        tmp             = self.pos[i, 2] + y
        self.pos_comp[i, 2] = (tmp - self.pos[i, 2]) - y
        self.pos[i, 2]  = tmp

    def update_masses(self, dt):
        """Applique dM/dt à chaque corps (perte de masse stellaire, etc.)."""
        for i in range(self.n):
            if self.mdot[i] != 0.0:
                self.masses[i] += self.mdot[i] * dt
                if self.masses[i] < 0.0:
                    self.masses[i] = 0.0

    def kahan_add_spin(self, i, dsx, dsy, dsz):
        """Additionne (dsx,dsy,dsz) à spin[i] avec compensation de Kahan."""
        y                   = dsx - self.spin_comp[i, 0]
        tmp                 = self.spin[i, 0] + y
        self.spin_comp[i, 0] = (tmp - self.spin[i, 0]) - y
        self.spin[i, 0]     = tmp

        y                   = dsy - self.spin_comp[i, 1]
        tmp                 = self.spin[i, 1] + y
        self.spin_comp[i, 1] = (tmp - self.spin[i, 1]) - y
        self.spin[i, 1]     = tmp

        y                   = dsz - self.spin_comp[i, 2]
        tmp                 = self.spin[i, 2] + y
        self.spin_comp[i, 2] = (tmp - self.spin[i, 2]) - y
        self.spin[i, 2]     = tmp
