"""Convective delivery time of the Fe-metal saturation step (redox.py Step 9).

Schaefer et al. (2024) react the melt to Fe-metal equilibrium at the base of
the magma ocean in every crystallisation step. In a time-resolved model the
melt has to be carried to that cell by convection first. With
config.planet.interior_chem.metal_saturation_relaxation = 'convective_transit'
the metal step applies only the fraction 1 - exp(-dt / tau) of the
equilibrium reaction extent per step, where tau [s] is the time a parcel takes
to travel from the top of the melt column to the reacting cell.

The parcel speed is the mixing-length velocity v = kappa_h / l of the interior
solver. Aragog builds its eddy diffusivity as kappa_h = v * l before the
``eddy_diffusivity_thermal`` scaling and the ``kappah_floor``, so kappa_h / l
is the speed the solver transports heat with, scaling and floor included.
tau = sum of dr / v over the cells crossed is a harmonic sum, dominated by the
slowest layer; a cell that does not convect (kappa_h = 0) makes tau infinite,
and no metal forms until convection reaches the reacting cell again.
"""

from __future__ import annotations

import numpy as np

# Constant mixing length as a fraction of the shell thickness; Aragog's default
# (aragog.mesh.fixed_mesh), which PROTEUS does not override.
CONSTANT_MIXING_LENGTH_FRACTION = 0.25

RELAXATION_MODES = ('none', 'convective_transit')


def mixing_length(radius_b, profile: str = 'nearest') -> np.ndarray:
    """Mixing length [m] at the basic nodes ``radius_b`` [m], built as Aragog
    builds it (``interior_energetics.mixing_length``): the distance to the
    nearer boundary for 'nearest', a quarter of the shell thickness for
    'constant'."""
    r = np.asarray(radius_b, dtype=float)
    r_in, r_out = np.min(r), np.max(r)
    if profile == 'nearest':
        return np.minimum(r_out - r, r - r_in)
    if profile == 'constant':
        return np.full_like(r, CONSTANT_MIXING_LENGTH_FRACTION * (r_out - r_in))
    raise ValueError(
        f"unknown mixing-length profile {profile!r}; expected 'nearest' or 'constant'"
    )


def cell_velocity(radius_b, eddy_diff_b, profile: str = 'nearest') -> np.ndarray:
    """Mixing-length velocity [m s-1] of each staggered cell: kappa_h / l
    [m2 s-1 / m] at the inner basic nodes, averaged onto the cells. The two
    boundary nodes, where l = 0 with 'nearest', take their inner neighbour's
    value. Negative kappa_h (none in Aragog) is treated as no convection."""
    k = np.asarray(eddy_diff_b, dtype=float)
    ell = mixing_length(radius_b, profile)
    if k.shape != ell.shape or k.size < 3:
        raise ValueError(
            f'eddy diffusivity has shape {k.shape}, the basic nodes {ell.shape}; '
            'need the same length and at least 3 nodes'
        )
    v = np.zeros_like(k)
    v[1:-1] = np.maximum(k[1:-1], 0.0) / ell[1:-1]
    v[0], v[-1] = v[1], v[-2]
    return 0.5 * (v[:-1] + v[1:])


def transit_time(radius_b, v_cell, i_react: int, i_top: int) -> float:
    """Time [s] for a parcel to cross the staggered cells from ``i_top`` (top
    of the melt column) to ``i_react`` (the reacting cell), both included, at
    the speeds ``v_cell`` [m s-1]: sum of dr / v. Infinite when one of them
    does not convect."""
    lo, hi = sorted((int(i_react), int(i_top)))
    dr = np.abs(np.diff(np.asarray(radius_b, dtype=float)))[lo : hi + 1]
    v = np.asarray(v_cell, dtype=float)[lo : hi + 1]
    if not np.all(np.isfinite(v) & (v > 0.0)):
        return float('inf')
    return float(np.sum(dr / v))


def relaxed_fraction(dt_s: float, tau_s: float) -> float:
    """Fraction 1 - exp(-dt/tau) of the equilibrium reaction extent applied in
    a step of ``dt_s`` [s] with delivery time ``tau_s`` [s]: 1 for tau = 0
    (instantaneous, Schaefer et al. 2024), 0 for dt <= 0 or tau infinite."""
    if dt_s <= 0.0 or not tau_s < float('inf'):
        return 0.0
    if tau_s <= 0.0:
        return 1.0
    return float(-np.expm1(-dt_s / tau_s))
