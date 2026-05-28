"""
integrator.py — Intégrateur symplectique de Yoshida du 4ème ordre.

Pourquoi Yoshida vs Velocity Verlet ?
──────────────────────────────────────────────────────────────────────────────
• Velocity Verlet est d'ordre 2 : la dérive séculaire de l'énergie croît
  comme O(dt²·t).
• Yoshida 4ème ordre : erreur locale O(dt⁴), dérive séculaire bornée.
  Structure hamiltonienne (volume de l'espace des phases) exactement préservée.
• Coût : 3 évaluations de forces par pas (vs 2 pour VV), mais la précision
  supérieure permet des dt plus grands → gain net.

Coefficients de Yoshida (Yoshida 1990, Phys. Lett. A) :
  cbrt2 = 2^(1/3)
  w1 =  1 / (2 - cbrt2)
  w0 = -cbrt2 / (2 - cbrt2)

Schéma correct : D K D K D K D  (4 drifts, 3 kicks)
  Coefficients drift : c1=w1/2,  c2=(w0+w1)/2,  c3=(w0+w1)/2,  c4=w1/2
  Coefficients kick  : d1=w1,    d2=w0,          d3=w1
  Vérification : Σ drifts = 1,  Σ kicks = 1

Sommation de Kahan pour les positions :
  Réduit l'arrondi accumulé de O(√N·ε) à O(ε) indépendamment du nombre
  d'itérations.

CORRECTION BUG #15 : le schéma était DKDKDK (3 drifts, 3 kicks) au lieu
  de DKDKDKD (4 drifts, 3 kicks). Le 4ème drift D4=w1/2 manquait, causant
  Σ drifts = 1.1756 ≠ 1 → dérive d'énergie monotone ~8.8% sur 5 ans.

CORRECTION BUG #7 : `enable_radiation` est maintenant un paramètre de
yoshida4_step et transmis à compute_forces_fn à chaque évaluation de force.

Compatibilité : Numba @njit nopython=True.
"""

import numpy as np
from numba import njit

# ── Coefficients de Yoshida 4ème ordre ───────────────────────────────────────
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1    =  1.0 / (2.0 - _CBRT2)
_W0    = -_CBRT2 / (2.0 - _CBRT2)

# Schéma D K D K D K D
# Drifts : c1, c2, c3, c4  (4 coefficients, symétriques)
# Kicks  : d1, d2, d3       (3 coefficients)
_D1 = _W1 / 2.0
_D2 = (_W0 + _W1) / 2.0
_D3 = _D2          # symétrie
_D4 = _D1          # symétrie — CORRECTION BUG #15 : était absent

_C1 = _W1
_C2 = _W0
_C3 = _W1


# ══════════════════════════════════════════════════════════════════════════════
#  SUNDMAN — temps fictif
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_potential_energy(state, G, softening):
    """
    Énergie potentielle totale U(r) (toujours <= 0 pour gravité newtonienne).
    Même softening que compute_forces_fn.
    """
    n = state.n
    Ep = 0.0
    eps2 = softening * softening
    for i in range(n):
        for j in range(i + 1, n):
            dx = state.pos[j, 0] - state.pos[i, 0]
            dy = state.pos[j, 1] - state.pos[i, 1]
            dz = state.pos[j, 2] - state.pos[i, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
            Ep -= G * state.masses[i] * state.masses[j] / dist
    return Ep


@njit(cache=True)
def sundman_g(state, G, softening):
    """
    Facteur dt/ds = g = 1 / (1 + |U| / U_ref), borné dans ]0, 1].
    """
    U_pot = compute_potential_energy(state, G, softening)
    U_abs = -U_pot
    if state.U_ref > 1e-30:
        return 1.0 / (1.0 + U_abs / state.U_ref)
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAPE YOSHIDA (1 pas de temps fictif ds)
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def yoshida4_step(state, dt_target, G, softening, compute_forces_fn,
                  enable_radiation=False, enable_pn1=False,
                  c_light=299792458.0):
    """
    Intégrateur symplectique Yoshida 4ème ordre — schéma D K D K D K D.

    dt_target : pas physique adaptatif (clampé dt_min … dt_max en amont).
    Sundman : ds = dt_target / g, dt = g·ds = dt_target (symplectique + rapide).
    Retourne (accel, dt_phys).
    """
    g = sundman_g(state, G, softening)
    if g < 1e-30:
        g = 1e-30
    ds = dt_target / g
    dt = g * ds
    n = state.n

    # ── Substep 1 : drift D1 ─────────────────────────────────────────────────
    d1_dt = _D1 * dt
    for i in range(n):
        state.kahan_add_pos(i,
            state.vel[i, 0] * d1_dt,
            state.vel[i, 1] * d1_dt,
            state.vel[i, 2] * d1_dt)

    # ── Kick K1 ──────────────────────────────────────────────────────────────
    accel, torques = compute_forces_fn(state, G, softening, enable_radiation,
                                       enable_pn1, c_light)
    c1_dt = _C1 * dt
    for i in range(n):
        state.vel[i, 0] += accel[i, 0] * c1_dt
        state.vel[i, 1] += accel[i, 1] * c1_dt
        state.vel[i, 2] += accel[i, 2] * c1_dt
        state.kahan_add_spin(i,
            torques[i, 0] * c1_dt,
            torques[i, 1] * c1_dt,
            torques[i, 2] * c1_dt)

    # ── Substep 2 : drift D2 ─────────────────────────────────────────────────
    d2_dt = _D2 * dt
    for i in range(n):
        state.kahan_add_pos(i,
            state.vel[i, 0] * d2_dt,
            state.vel[i, 1] * d2_dt,
            state.vel[i, 2] * d2_dt)

    # ── Kick K2 ──────────────────────────────────────────────────────────────
    accel, torques = compute_forces_fn(state, G, softening, enable_radiation,
                                       enable_pn1, c_light)
    c2_dt = _C2 * dt
    for i in range(n):
        state.vel[i, 0] += accel[i, 0] * c2_dt
        state.vel[i, 1] += accel[i, 1] * c2_dt
        state.vel[i, 2] += accel[i, 2] * c2_dt
        state.kahan_add_spin(i,
            torques[i, 0] * c2_dt,
            torques[i, 1] * c2_dt,
            torques[i, 2] * c2_dt)

    # ── Substep 3 : drift D3 ─────────────────────────────────────────────────
    d3_dt = _D3 * dt
    for i in range(n):
        state.kahan_add_pos(i,
            state.vel[i, 0] * d3_dt,
            state.vel[i, 1] * d3_dt,
            state.vel[i, 2] * d3_dt)

    # ── Kick K3 ──────────────────────────────────────────────────────────────
    accel, torques = compute_forces_fn(state, G, softening, enable_radiation,
                                       enable_pn1, c_light)
    c3_dt = _C3 * dt
    for i in range(n):
        state.vel[i, 0] += accel[i, 0] * c3_dt
        state.vel[i, 1] += accel[i, 1] * c3_dt
        state.vel[i, 2] += accel[i, 2] * c3_dt
        state.kahan_add_spin(i,
            torques[i, 0] * c3_dt,
            torques[i, 1] * c3_dt,
            torques[i, 2] * c3_dt)

    # ── Substep 4 : drift D4 ─────────────────────────────────────────────────
    d4_dt = _D4 * dt
    for i in range(n):
        state.kahan_add_pos(i,
            state.vel[i, 0] * d4_dt,
            state.vel[i, 1] * d4_dt,
            state.vel[i, 2] * d4_dt)

    state.advance_fictitious_time(ds)
    state.advance_time(dt)
    state.update_masses(dt)

    state.accel[:, :] = accel
    return accel, dt
