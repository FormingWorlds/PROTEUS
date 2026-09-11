# Evection-resonance dt-cap logic for the planet-satellite orbit model.
#
# Moved out of proteus.interior_energetics.timestep.next_step (the rate cap)
# and proteus.orbit.satellite (both functions originally lived inline in
# evolve_orbit_satellite/next_step) into their own file so this dt-cap
# machinery -- the secular-rate cap and the evection-scoped growth limiter,
# folded together into the single value evolve_orbit_satellite exports as
# hf_row['evection_dt_cap_yr'] -- reads as one coherent unit rather than
# being split across two modules in two different packages.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from proteus.orbit.common import Tides_t

if TYPE_CHECKING:
    from proteus.config import Config


def _evection_rate_cap_yr(tides_o: Tides_t, zone_active: bool, config: Config) -> float:
    """Bound the NEXT macro-step by the secular rate of change of the
    satellite eccentricity, while the system is inside or approaching the
    evection resonance band.

    This is orbit's own copy of what used to be
    ``proteus.interior_energetics.timestep._estimate_evection_dt_cap``. The
    rate is fit over ``tides_o.evection_ecc_history``, a rolling window
    ``proteus.orbit.satellite.evolve_orbit_satellite`` maintains, rather
    than the main loop's ``hf_all``, so this module does not need a
    dependency on the run's full history DataFrame to compute its own dt
    cap.

    Returns ``np.inf`` when the cap does not apply (disabled, not in/near
    the band, or no rate history yet distinguishable from the ceiling).
    One of the two bounds ``_estimate_evection_dt_cap_yr`` folds together;
    not itself written to ``hf_row``.
    """
    evection_max = float(config.params.dt.evection_maximum)
    if evection_max <= 0.0 or not zone_active:
        return np.inf

    history = tides_o.evection_ecc_history
    if len(history) < 2:
        # No history to derive a rate from yet (e.g. the very first step
        # after the zone flag flips), fall back to the ceiling rather
        # than either ignoring the cap or raising on missing history.
        return evection_max

    window = max(2, int(getattr(config.params.dt, 'evection_rate_window', 2)))
    n_use = min(window, len(history))
    recent = history[-n_use:]
    t_window = np.array([t for t, _ in recent], dtype=float)
    e_window = np.array([e for _, e in recent], dtype=float)

    dt_span = t_window[-1] - t_window[0]
    if dt_span <= 0.0:
        return evection_max

    if n_use == 2:
        de_dt = abs(e_window[-1] - e_window[0]) / dt_span
    else:
        # Least-squares secular slope: uses every sample in the window
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
    """Bound the NEXT macro-step via evection: the single value written
    to ``hf_row['evection_dt_cap_yr']``, folding together the secular-rate
    cap (``_evection_rate_cap_yr``) and the evection-scoped growth
    limiter -- both used to live in
    ``proteus.interior_energetics.timestep.next_step`` (as
    ``_estimate_evection_dt_cap``/the ``in_evection_band``-gated growth-
    limiter block); both are now computed here instead, so ``next_step``
    has no evection-specific logic left at all beyond folding this one
    number into ``min()``, and the helpfile carries only this one column
    rather than the two boolean flags the growth limiter used to read.

    The growth limiter bounds dt to ``dt_prev_actual_yr *
    evection_growth_factor`` while the zone is active OR for
    ``evection_cooldown_iters`` calls after it was last active (so exiting
    the band does not snap dt straight back to whatever the ordinary
    controller wants) -- the counter is refreshed/decremented on
    ``tides_o.evection_cooldown_remaining`` every call, using the zone
    state observed THIS call, regardless of whether either cap actually
    binds this time.

    ``dt_prev_actual_yr`` is the ACTUAL elapsed ``hf_row['Time']`` during
    THIS call (not the requested ``interior_o.dt``, which an incomplete
    substep loop can fall short of) -- see
    ``proteus.orbit.satellite.evolve_orbit_satellite``.

    Returns ``np.inf`` when neither bound applies, so the caller can fold
    it into a ``min()`` unconditionally.
    """
    dt_cfg = config.params.dt
    rate_cap = _evection_rate_cap_yr(tides_o, zone_active, config)

    evection_growth = float(getattr(dt_cfg, 'evection_growth_factor', 0.0))
    cooldown_remaining = int(tides_o.evection_cooldown_remaining)
    growth_cap = np.inf
    if evection_growth > 0.0 and (zone_active or cooldown_remaining > 0):
        if dt_prev_actual_yr is not None and dt_prev_actual_yr > 0.0:
            growth_cap = dt_prev_actual_yr * evection_growth

    # Refresh/decrement the cooldown counter for the NEXT call, using the
    # zone state observed THIS call -- mirrors the old next_step ordering:
    # this update happens regardless of whether either bound above
    # actually constrained anything this time.
    if zone_active:
        tides_o.evection_cooldown_remaining = int(getattr(dt_cfg, 'evection_cooldown_iters', 0))
    elif cooldown_remaining > 0:
        tides_o.evection_cooldown_remaining = cooldown_remaining - 1

    return min(rate_cap, growth_cap)
