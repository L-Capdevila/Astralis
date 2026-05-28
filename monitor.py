"""
monitor.py — Diagnostic passif de l'hamiltonien.

Rôle :
──────────────────────────────────────────────────────────────────────────────
• Calcule l'énergie totale (cinétique + potentielle) = hamiltonien H.
• Calcule le moment cinétique total |L|.
• Appelé HORS de la boucle critique (toutes les N frames).

Interprétation de ΔH/H₀ :
  < 1e-9  : excellent
  < 1e-6  : acceptable
  > 1e-4  : réduire dt ou vérifier le code
  > 1e-2  : simulation invalide

Compatibilité : Numba @njit nopython=True.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def compute_hamiltonian(state, G, softening=0.0):
    """
    H = Σ_i ½·M_i·|v_i|²  +  Σ_{i<j} -G·M_i·M_j / sqrt(|r_ij|² + ε²)
    Retourne H en Joules.

    softening : longueur de lissage gravitationnel (même valeur que dans
                forces.py), indispensable pour que H₀ corresponde exactement
                au hamiltonien effectivement intégré et que ΔH/H₀ soit valide.
    """
    n   = state.n
    Ec  = 0.0
    Ep  = 0.0
    eps2 = softening * softening
    for i in range(n):
        vx  = state.vel[i, 0]
        vy  = state.vel[i, 1]
        vz  = state.vel[i, 2]
        Ec += 0.5 * state.masses[i] * (vx*vx + vy*vy + vz*vz)
        for j in range(i + 1, n):
            dx   = state.pos[j, 0] - state.pos[i, 0]
            dy   = state.pos[j, 1] - state.pos[i, 1]
            dz   = state.pos[j, 2] - state.pos[i, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz + eps2)
            Ep  -= G * state.masses[i] * state.masses[j] / dist
    return Ec + Ep


@njit(cache=True)
def compute_angular_momentum_components(state):
    """Retourne (Lx, Ly, Lz) du moment cinétique total."""
    lx = 0.0
    ly = 0.0
    lz = 0.0
    for i in range(state.n):
        m  = state.masses[i]
        x  = state.pos[i, 0]
        y  = state.pos[i, 1]
        z  = state.pos[i, 2]
        vx = state.vel[i, 0]
        vy = state.vel[i, 1]
        vz = state.vel[i, 2]
        lx += m * (y * vz - z * vy)
        ly += m * (z * vx - x * vz)
        lz += m * (x * vy - y * vx)
    return lx, ly, lz


@njit(cache=True)
def compute_angular_momentum(state):
    """Norme 3D du moment cinétique orbital total : |Σ_i M_i (r_i × v_i)|."""
    lx, ly, lz = compute_angular_momentum_components(state)
    return np.sqrt(lx*lx + ly*ly + lz*lz)


@njit(cache=True)
def compute_diagnostics(state, G, softening=0.0):
    """
    Retourne (H, L, dH_rel) en un seul appel.
    dH_rel = (H - H0) / |H0|.  NaN si H0 == 0.
    softening doit être identique à celui utilisé dans forces.py.
    """
    H  = compute_hamiltonian(state, G, softening)
    L  = compute_angular_momentum(state)
    dH = (H - state.E0) / abs(state.E0) if state.E0 != 0.0 else np.nan
    return H, L, dH


def summarize_drift_series(dh_values):
    """Résumé hors boucle critique : max absolu et RMS d'une série de dH."""
    arr = np.asarray(dh_values, dtype=np.float64)
    if arr.size == 0:
        return {"max_abs": np.nan, "rms": np.nan}
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"max_abs": np.nan, "rms": np.nan}
    return {
        "max_abs": float(np.max(np.abs(arr))),
        "rms":     float(np.sqrt(np.mean(arr * arr))),
    }
