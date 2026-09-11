# Evection-resonance dt-cap logic for the planet-satellite orbit model.
# See "Evection resonance" in docs/Explanations/orbit.md.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from proteus.orbit.common import Tides_t

if TYPE_CHECKING:
    from proteus.config import Config


def _evection_rate_cap_yr(tides_o: Tides_t, zone_active: bool, config: Config) -> float:
    """Bound the NEXT macro-step by the secular rate of change of the
    satellite eccentricity, while the system is inside or approaching the
    evection resonance band (see docs/Explanations/orbit.md). Fit over
    ``tides_o.evection_ecc_history``, a rolling window maintained by
    ``satellite.evolve_orbit_satellite``.

    Returns ``np.inf`` when the cap does not apply.
    """
    # Check configured maximum step size for evection resonance
    evection_max = float(config.params.dt.evection_maximum)
    if evection_max <= 0.0 or not zone_active:
        return np.inf

    # Check the history of eccentricity samples
    history = tides_o.evection_ecc_history
    if len(history) < 2:
        # No history to derive a rate from yet, fall back to the ceiling
        return evection_max

    # Determine the window size for the rate calculation, and extract the recent samples
    window = max(2, int(getattr(config.params.dt, 'evection_rate_window', 2)))
    n_use = min(window, len(history))
    recent = history[-n_use:]
    t_window = np.array([t for t, _ in recent], dtype=float)
    e_window = np.array([e for _, e in recent], dtype=float)

    # Compute the secular rate of change of eccentricity over the window
    dt_span = t_window[-1] - t_window[0]
    if dt_span <= 0.0:
        # No time span to derive a rate from, fall back to the ceiling
        return evection_max

    # Compute the secular rate of change of eccentricity over the window
    if n_use == 2:
        # Simple two-point slope
        de_dt = abs(e_window[-1] - e_window[0]) / dt_span
    else:
        # Least-squares secular slope: uses every sample in the window to
        # average out short-period oscillations.
        slope, _ = np.polyfit(t_window, e_window, 1)
        de_dt = abs(float(slope))

    if de_dt <= 0.0:
        # No net secular trend over the window; fall back to the
        # ceiling rather than letting a naive e/de_dt blow up.
        return evection_max

    # Compute the cap on the next step size from the observed |de/dt|
    e_now = float(e_window[-1])
    target_rel_de = float(getattr(config.params.dt, 'evection_target_rel_de', 0.05))
    de_floor = float(getattr(config.params.dt, 'evection_de_floor', 0.02))
    e_ref = max(e_now, de_floor)

    dt_rate_cap = target_rel_de * e_ref / de_dt
    return min(evection_max, dt_rate_cap)


def _estimate_evection_dt_cap_yr(
    tides_o: Tides_t, zone_active: bool, dt_prev_actual_yr: float, config: Config
) -> float:
    """Bound the NEXT macro-step via evection
    Controls ``hf_row['evection_dt_cap_yr']``, folding together the
    secular-rate cap (``_evection_rate_cap_yr``) and a growth limiter that
    bounds dt to ``dt_prev_actual_yr * evection_growth_factor`` while the
    zone is active, or for ``evection_cooldown_iters`` calls after it was
    last active (see docs/Explanations/orbit.md, "Evection resonance").

    Returns ``np.inf`` when neither bound applies.
    """
    dt_cfg = config.params.dt

    # Compute the secular-rate cap on the next step size from the observed |de/dt|
    rate_cap = _evection_rate_cap_yr(tides_o, zone_active, config)

    # Compute the growth limiter
    evection_growth = float(getattr(dt_cfg, 'evection_growth_factor', 0.0))
    cooldown_remaining = int(tides_o.evection_cooldown_remaining)
    growth_cap = np.inf
    # The growth limiter is only active while the zone is active, or for a
    # short cooldown period after it was last active.
    if evection_growth > 0.0 and (zone_active or cooldown_remaining > 0):
        if dt_prev_actual_yr is not None and dt_prev_actual_yr > 0.0:
            growth_cap = dt_prev_actual_yr * evection_growth

    # Refresh/decrement the cooldown counter for the NEXT call, using the
    # zone state observed THIS call.
    if zone_active:
        tides_o.evection_cooldown_remaining = int(getattr(dt_cfg, 'evection_cooldown_iters', 0))
    elif cooldown_remaining > 0:
        tides_o.evection_cooldown_remaining = cooldown_remaining - 1

    return min(rate_cap, growth_cap)
