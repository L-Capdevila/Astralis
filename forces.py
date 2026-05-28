"""
forces.py — Gestionnaire de forces découplé.

Architecture :
──────────────────────────────────────────────────────────────────────────────
• `compute_forces(state, G, softening, enable_radiation)` est le point
  d'entrée unique appelé par l'intégrateur.

• Chaque contribution physique est une fonction @njit indépendante.

• Pour ajouter une nouvelle force : écrire la fonction @njit et l'appeler
  dans `compute_forces` — c'est tout.

CORRECTION BUG #7 : le paramètre `enable_radiation` était défini dans la
signature de `compute_forces` mais jamais transmis par l'intégrateur car
celui-ci appelait `compute_forces_fn(state, G, softening)` avec 3 args.
Solution : l'intégrateur passe maintenant `enable_radiation` explicitement.

CORRECTION BUG #10 : compute_adaptive_dt recalculait la paire minimale en
double (une fois via compute_min_dist, une fois via une boucle identique).
compute_min_dist retourne maintenant (dist_min, i, j) et compute_adaptive_dt
utilise directement ces indices — une seule boucle au lieu de deux.

Compatibilité : Numba @njit nopython=True.
"""

import numpy as np
from numba import njit, prange


# ══════════════════════════════════════════════════════════════════════════════
#  FORCE 1 : Gravitation newtonienne (avec softening de Plummer)
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, parallel=True)
def compute_gravity_newton(state, G, softening, accel_out):
    """
    Gravité newtonienne O(n²) avec softening de Plummer :
        a_i = Σ_{j≠i} G·M_j·r_ij / (|r_ij|² + ε²)^(3/2)

    Boucle complète (sans symétrie) : chaque thread i écrit exclusivement
    dans accel_out[i], ce qui évite les race conditions en mode parallel.
    """
    n = state.n
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            dx    = state.pos[j, 0] - state.pos[i, 0]
            dy    = state.pos[j, 1] - state.pos[i, 1]
            dz    = state.pos[j, 2] - state.pos[i, 2]
            dist2 = dx*dx + dy*dy + dz*dz + softening*softening
            dist  = np.sqrt(dist2)
            dist3 = dist2 * dist
            fij = G / dist3
            accel_out[i, 0] += fij * state.masses[j] * dx
            accel_out[i, 1] += fij * state.masses[j] * dy
            accel_out[i, 2] += fij * state.masses[j] * dz


# ══════════════════════════════════════════════════════════════════════════════
#  FORCE 2 : Perturbation J2 (aplatissement) — axe de spin arbitraire
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_j2_perturbation(state, G, accel_out):
    """
    Perturbation J2 due à l'aplatissement de chaque corps.
    L'axe de symétrie est aligné sur le vecteur spin[i] (normalisé).
    Si spin[i] est nul, l'axe Z est utilisé par défaut (comportement original).

    Formule générale pour un axe de symétrie ŝ quelconque :
      Φ_J2 = -3/2 · G·M·J2R² / r⁵ · [(r·ŝ)² - r²/3]
    Les accélérations dérivées sont données par le gradient de ce potentiel.
    """
    n = state.n
    for i in range(n):
        if state.j2r2[i] == 0.0:
            continue

        # Axe de spin normalisé pour le corps i
        sx = state.spin[i, 0]
        sy = state.spin[i, 1]
        sz = state.spin[i, 2]
        s_norm = np.sqrt(sx*sx + sy*sy + sz*sz)
        if s_norm > 1e-30:
            sx /= s_norm; sy /= s_norm; sz /= s_norm
        else:
            sx = 0.0; sy = 0.0; sz = 1.0   # axe Z par défaut

        for j in range(n):
            if i == j:
                continue
            dx   = state.pos[j, 0] - state.pos[i, 0]
            dy   = state.pos[j, 1] - state.pos[i, 1]
            dz   = state.pos[j, 2] - state.pos[i, 2]
            r2   = dx*dx + dy*dy + dz*dz
            if r2 <= 1e-30:
                continue
            r    = np.sqrt(r2)
            r5   = r2 * r2 * r

            # Projection du vecteur r sur l'axe de spin : (r · ŝ)
            r_dot_s = dx*sx + dy*sy + dz*sz
            cos2    = (r_dot_s * r_dot_s) / r2   # cos²(θ) où θ = angle r/ŝ

            pref = -1.5 * G * state.masses[i] * state.j2r2[i] / r5

            # Gradient du potentiel J2 (forme tensorielle complète)
            # ∂Φ/∂r_j = pref · [r_j·(1-5·cos²) + 2·(r·ŝ)·s_j]
            accel_out[j, 0] += pref * (dx * (1.0 - 5.0*cos2) + 2.0*r_dot_s*sx)
            accel_out[j, 1] += pref * (dy * (1.0 - 5.0*cos2) + 2.0*r_dot_s*sy)
            accel_out[j, 2] += pref * (dz * (1.0 - 5.0*cos2) + 2.0*r_dot_s*sz)


# ══════════════════════════════════════════════════════════════════════════════
#  FORCE 2b : Couple de marée sur le spin (couplage spin-orbite)
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_tidal_torques(state, G, torque_out):
    """
    Couple de marée exercé par chaque paire de corps sur le spin de chaque corps.

    Principe :
      Le corps j exerce sur le corps i un couple τ = dS_i/dt qui fait
      précescer l'axe de rotation de i. Ce couple est le cross-product entre
      l'axe de spin ŝ_i et l'axe de la force J2 appliquée par j sur i.

    Formule :
      τ_i = -k2_i · G·M_j · (J2R²_i / r_ij⁵) · (r_ij × ŝ_i) × ŝ_i · 3·cos(θ)

    k2 (nombre de Love) module l'amplitude : k2=0 → corps rigide sans précession.

    Résultat stocké dans torque_out[i, 3] (en N·m = kg·m²/s²).
    """
    n = state.n
    for i in range(n):
        if state.j2r2[i] == 0.0 or state.k2[i] == 0.0:
            continue

        sx = state.spin[i, 0]
        sy = state.spin[i, 1]
        sz = state.spin[i, 2]
        s_norm = np.sqrt(sx*sx + sy*sy + sz*sz)
        if s_norm < 1e-30:
            continue
        sx /= s_norm; sy /= s_norm; sz /= s_norm

        for j in range(n):
            if i == j:
                continue
            dx = state.pos[j, 0] - state.pos[i, 0]
            dy = state.pos[j, 1] - state.pos[i, 1]
            dz = state.pos[j, 2] - state.pos[i, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < 1e-30:
                continue
            r  = np.sqrt(r2)
            r5 = r2 * r2 * r

            r_dot_s = dx*sx + dy*sy + dz*sz   # r · ŝ

            # Composante du couple : τ = k2 · pref · (r × ŝ) × ŝ · 3·cos(θ)
            # (r × ŝ) : cross product
            cx = dy*sz - dz*sy
            cy = dz*sx - dx*sz
            cz = dx*sy - dy*sx
            # × ŝ à nouveau
            tx = cy*sz - cz*sy
            ty = cz*sx - cx*sz
            tz = cx*sy - cy*sx

            pref = -3.0 * state.k2[i] * G * state.masses[j] * state.j2r2[i] * r_dot_s / (r5 * r2)

            torque_out[i, 0] += pref * tx
            torque_out[i, 1] += pref * ty
            torque_out[i, 2] += pref * tz


# ══════════════════════════════════════════════════════════════════════════════
#  FORCE 3 : Gabarit — pression de radiation / Yarkovsky / Poynting-Robertson
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_radiation_pressure(state, G, accel_out):
    """
    [GABARIT] Pression de radiation.
    Implémenter la physique ici et activer via enable_radiation=True.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  FORCE 4 : Correction post-newtonienne PN1 (précession des périhélies)
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_pn1_correction(state, G, c_light, accel_out):
    """
    Correction post-newtonienne au premier ordre (PN1) — terme dominant
    relativiste de l'équation EIH (Einstein-Infeld-Hoffmann).

    Ajoute la précession des périhélies (effet Mercure, étoiles à neutrons,
    binaires compactes).

    Forme implémentée : forme simplifiée fournie en spécification produit.
    Pour des études haute précision (ondes gravitationnelles, mouvements
    propres relativistes), il faut passer à la forme EIH complète
    (Will 1993, eq. 4.4.15) qui inclut un terme transverse en
    (v_i - v_j)·(n·(4v_i - 3v_j)) et les corrections d'ordre supérieur.
    """
    n  = state.n
    c2 = c_light * c_light

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = state.pos[j, 0] - state.pos[i, 0]
            dy = state.pos[j, 1] - state.pos[i, 1]
            dz = state.pos[j, 2] - state.pos[i, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < 1e-30:
                continue
            r  = np.sqrt(r2)
            r3 = r2 * r

            vix, viy, viz = state.vel[i, 0], state.vel[i, 1], state.vel[i, 2]
            vjx, vjy, vjz = state.vel[j, 0], state.vel[j, 1], state.vel[j, 2]

            vi2 = vix*vix + viy*viy + viz*viz
            vj2 = vjx*vjx + vjy*vjy + vjz*vjz
            v_dot_n = ((vix - vjx) * dx
                     + (viy - vjy) * dy
                     + (viz - vjz) * dz) / r

            gm_j = G * state.masses[j]
            gm_i = G * state.masses[i]

            factor = gm_j / (r3 * c2)
            pn_scalar = (vi2 + 2.0*vj2
                         - 4.0*(vix*vjx + viy*vjy + viz*vjz)
                         - 1.5*v_dot_n*v_dot_n
                         - 4.0*gm_i/r
                         - 5.0*gm_j/r)

            accel_out[i, 0] += factor * (pn_scalar * dx
                                         + (4.0*vix - 3.0*vjx) * v_dot_n * r)
            accel_out[i, 1] += factor * (pn_scalar * dy
                                         + (4.0*viy - 3.0*vjy) * v_dot_n * r)
            accel_out[i, 2] += factor * (pn_scalar * dz
                                         + (4.0*viz - 3.0*vjz) * v_dot_n * r)


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE UNIQUE
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_forces(state, G, softening=1e6, enable_radiation=False,
                   enable_pn1=False, c_light=299792458.0):
    """
    Orchestre toutes les contributions de forces.
    Retourne (accel [n,3], torques [n,3]).

    torques[i] = couple de marée dS_i/dt sur le spin du corps i.
    Si k2=0 pour tous les corps, torques est un tableau nul (pas de surcoût).
    """
    accel_out  = np.zeros((state.n, 3), dtype=np.float64)
    torque_out = np.zeros((state.n, 3), dtype=np.float64)
    compute_gravity_newton(state, G, softening, accel_out)
    compute_j2_perturbation(state, G, accel_out)
    if enable_radiation:
        compute_radiation_pressure(state, G, accel_out)
    if enable_pn1:
        compute_pn1_correction(state, G, c_light, accel_out)
    compute_tidal_torques(state, G, torque_out)
    return accel_out, torque_out


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES GÉOMÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_min_dist(state):
    """
    Retourne (dist_min, i, j) pour la paire la plus proche.

    CORRECTION BUG #10 : les indices i, j sont maintenant utilisés
    directement dans compute_adaptive_dt, évitant une boucle redondante.
    """
    n        = state.n
    dist_min = 1e30
    pi, pj   = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            dx   = state.pos[j, 0] - state.pos[i, 0]
            dy   = state.pos[j, 1] - state.pos[i, 1]
            dz   = state.pos[j, 2] - state.pos[i, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < dist_min:
                dist_min = dist
                pi, pj   = i, j
    return dist_min, pi, pj


@njit(cache=True)
def compute_adaptive_dt(state, accel, dt_max, dt_min, dist_seuil, alpha=0.1):
    """
    Pas de temps adaptatif combinant deux contraintes :
      1) dynamique  : dt ~ sqrt(r_min / a_max)
      2) cinématique: dt ~ r_min / v_rel_min  (évite de sauter les rencontres)

    CORRECTION BUG #10 : compute_min_dist retourne maintenant (dist_min, i, j).
    On réutilise directement i_best et j_best sans reboucler sur toutes les
    paires — la complexité passe de 2×O(n²) à 1×O(n²) par appel.
    """
    dist_min, i_best, j_best = compute_min_dist(state)
    return compute_adaptive_dt_from_min_dist(
        state, accel, dt_max, dt_min, dist_seuil, dist_min, i_best, j_best, alpha
    )


@njit(cache=True)
def compute_adaptive_dt_from_min_dist(
    state, accel, dt_max, dt_min, dist_seuil, dist_min, i_best, j_best, alpha=0.1
):
    """
    Variante de compute_adaptive_dt qui réutilise une paire déjà calculée.

    Permet au moteur principal d'éviter un second O(n²) quand il a déjà
    (dist_min, i_best, j_best) pour la détection de collision.
    """

    # Vitesse relative de la paire la plus proche (contrainte cinématique)
    v_rel_min = 0.0
    if state.n > 1:
        dvx = state.vel[j_best, 0] - state.vel[i_best, 0]
        dvy = state.vel[j_best, 1] - state.vel[i_best, 1]
        dvz = state.vel[j_best, 2] - state.vel[i_best, 2]
        v_rel_min = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

    accel_max = 0.0
    for i in range(state.n):
        a = np.sqrt(accel[i, 0]**2 + accel[i, 1]**2 + accel[i, 2]**2)
        if a > accel_max:
            accel_max = a

    dt_dyn = alpha * np.sqrt(dist_min / accel_max) if accel_max > 1e-30 else dt_max
    dt_kin = 0.2 * dist_min / v_rel_min            if v_rel_min > 1e-30 else dt_max
    dt     = dt_dyn if dt_dyn < dt_kin else dt_kin

    if dist_min < dist_seuil:
        dt *= dist_min / dist_seuil

    return min(max(dt, dt_min), dt_max)
