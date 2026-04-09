# Contains routines for setting the model timestep
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Constants
LBAVG = 3  # Number of steps to average over
SFINC = 1.6  # Scale factor for step size increase
SFDEC = 0.8  # Scale factor for step size decrease
SMALL = 1e-8  # Small number


def _hf_from_iters(hf_all: pd.DataFrame, i1: int, i2: int):
    # Get helpfile rows for two different iterations

    # i2 must be larger than i1
    if i1 >= i2:
        log.error('Cannot compare helpfile rows (i1=%d  >=  i2=%d)' % (i1, i2))

    # Return HF rows at the requested iterations
    return dict(hf_all.iloc[i1]), dict(hf_all.iloc[i2])


def _estimate_solid(hf_all: pd.DataFrame, i1: int, i2: int) -> float:
    """
    Estimate the time remaining until the planet solidifies.
    """

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

    # Melt fractions
    p1 = h1['Phi_global']
    p2 = h2['Phi_global']

    # Check if planet has already solidified
    if p2 < SMALL:
        dt_solid = np.inf

    else:
        # Change in time and global melt frac
        dt = h2['Time'] - h1['Time']
        dp = p2 - p1

        # Estimate how long Δt until p=0
        if abs(dp / p2) < SMALL:
            dt_solid = np.inf
        else:
            #  dp/dt * Δt + p2 = 0    ->   Δt = -p2/(dp/dt)
            dt_solid = abs(-1.0 * p2 / (dp / dt))

    log.debug('Solidification expected in %.3e yrs' % dt_solid)

    return dt_solid


def _estimate_radeq(hf_all: pd.DataFrame, i1: int, i2: int) -> float:
    """
    Estimate the time remaining until the energy balance is achieved.
    """

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

    # Flux residuals
    f2 = h2['F_atm'] - h2['F_tidal'] - h2['F_radio']
    f1 = h1['F_atm'] - h1['F_tidal'] - h1['F_radio']

    # Check if planet is already at radeq
    if abs(f2) < SMALL:
        dt_radeq = np.inf

    else:
        # Change in time and global melt frac
        dt = h2['Time'] - h1['Time']
        df = f2 - f1

        # Estimate how long until f=0
        if abs(df / f2) < SMALL:
            dt_radeq = np.inf
        else:
            dt_radeq = abs(-1.0 * f2 / (df / dt))

    log.debug('Energy balance expected in %.3e yrs' % dt_radeq)

    return dt_radeq


def _estimate_escape(hf_all: pd.DataFrame, i1: int, i2: int) -> float:
    """
    Estimate the time remaining until the surface pressure is zero.
    """

    # HF at times
    h1, h2 = _hf_from_iters(hf_all, i1, i2)

    # Surface pressures
    p1 = h1['P_surf']
    p2 = h2['P_surf']

    # Change in time and global melt frac
    dt = h2['Time'] - h1['Time']
    dp = p2 - p1

    # Estimate how long Δt until p=0
    if abs(dp / p2) < SMALL:
        # already escaped
        dt_escape = np.inf
    else:
        #  dp/dt * Δt + p2 = 0    ->   Δt = -p2/(dp/dt)
        dt_escape = abs(-1.0 * p2 / (dp / dt))

    log.debug('Escape expected in %.3e yrs' % dt_escape)

    return dt_escape


def next_step(
    config: Config,
    dirs: dict,
    hf_row: dict,
    hf_all: pd.DataFrame,
    step_sf: float,
    interior_o=None,
) -> float:
    """
    Determine the size of the next interior time-step.

    Parameters
    -----------
        config : dict
            Dictionary of configuration options
        dirs : dict
            Dictionary of directories
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        step_sf : float
            Scale factor to apply to step size
        interior_o : Interior_t, optional
            Interior object used to persist stiffness-aware adaptive
            state (hysteresis counter) across calls. When ``None``,
            the hysteresis and stiffness logging features are
            disabled and the controller reduces to its pre-2026-04-09
            behaviour. All call sites should now pass ``interior_o``
            when available.

    Returns
    -----------
        dtswitch : float
            Optimal step size [years].
    """

    # First years, use small step
    if hf_row['Time'] < 2.0:
        dtswitch = 1.0
        log.info('Time-stepping intent: static')

    elif LBAVG + 5 >= len(hf_all['Time']):
        dtswitch = config.params.dt.initial
        log.info('Time-stepping intent: initial')

    else:
        i2 = -1
        i1 = -2
        i0 = i1 - LBAVG

        # Proportional time-step calculation
        if config.params.dt.method == 'proportional':
            log.info('Time-stepping intent: proportional')
            dtswitch = hf_row['Time'] / config.params.dt.propconst

        # Dynamic time-step calculation
        elif config.params.dt.method == 'adaptive':
            # Try to maintain a minimum step size of dt_initial at first
            if hf_row['Time'] > config.params.dt.initial:
                dtprev = float(hf_all.iloc[-1]['Time'] - hf_all.iloc[-2]['Time'])
            else:
                dtprev = config.params.dt.initial
            log.debug('Previous step size: %.2e yr' % dtprev)

            # Change in F_atm
            F_atm_2 = hf_all['F_atm'].iloc[i2]
            F_atm_1 = np.median(hf_all['F_atm'].iloc[i0:i1])
            F_atm_12 = abs(F_atm_2 - F_atm_1)

            # Change in global melt fraction
            phi_2 = hf_all['Phi_global'].iloc[i2]
            phi_1 = np.median(hf_all['Phi_global'].iloc[i0:i1])
            phi_12 = abs(phi_2 - phi_1)

            # Determine new time-step given the tolerances
            dt_rtol = config.params.dt.rtol
            dt_atol = config.params.dt.atol
            speed_up = True
            speed_up = speed_up and (F_atm_12 < dt_rtol * abs(F_atm_2) + dt_atol)
            speed_up = speed_up and (phi_12 < dt_rtol * abs(phi_2) + dt_atol)

            # Hysteresis-aware speed-up factor.
            #
            # After a recent "slow down" decision, suppress the
            # speed-up factor for ``hysteresis_iters`` iterations so
            # the controller cannot ramp straight back into the stiff
            # cliff. Counter lives on interior_o (persists across
            # calls). Disabled when interior_o is None or when
            # config.params.dt.hysteresis_iters is 0.
            hyst_remaining = (
                getattr(interior_o, 'dt_hysteresis_remaining', 0)
                if interior_o is not None
                else 0
            )
            sfinc_effective = SFINC
            if hyst_remaining > 0:
                sfinc_effective = min(
                    float(config.params.dt.hysteresis_sfinc), SFINC
                )
                log.info(
                    'Time-stepping: hysteresis active (%d iters remaining), '
                    'using gentler sfinc=%.2f instead of %.2f',
                    hyst_remaining, sfinc_effective, SFINC,
                )

            if speed_up:
                dtswitch = dtprev * sfinc_effective
                log.info('Time-stepping intent: speed up')
            else:
                dtswitch = dtprev * SFDEC
                log.info('Time-stepping intent: slow down')
                # Arm the hysteresis timer on every slow-down.
                if interior_o is not None:
                    interior_o.dt_hysteresis_remaining = int(
                        config.params.dt.hysteresis_iters
                    )

            # Decrement hysteresis counter (applies to both speed_up
            # and slow_down branches; the arming above would have
            # already reset the counter on a slow-down).
            if (
                interior_o is not None
                and interior_o.dt_hysteresis_remaining > 0
                and speed_up
            ):
                interior_o.dt_hysteresis_remaining -= 1

            # Do not allow step size to exceed predicted point of termination
            if config.params.stop.solid.enabled:
                dtswitch = min(dtswitch, _estimate_solid(hf_all, i1, i2))
            if config.params.stop.radeqm.enabled:
                dtswitch = min(dtswitch, _estimate_radeq(hf_all, i1, i2))
            if config.params.stop.escape.enabled:
                dtswitch = min(dtswitch, _estimate_escape(hf_all, i1, i2) * 1.1)

        # Always use the maximum time-step, which can be adjusted in the cfg file
        elif config.params.dt.method == 'maximum':
            log.info('Time-stepping intent: maximum')
            dtswitch = config.params.dt.maximum

        # Handle all other inputs
        else:
            UpdateStatusfile(dirs, 20)
            raise ValueError(f'Invalid time-stepping method: {config.params.dt.method}')

        # Min step size (adaptive branch only — the static and initial
        # branches set dt from explicit config values and should not be
        # floored to dt.minimum before the retry scaling is applied).
        dtminimum = config.params.dt.minimum  # absolute
        dtminimum += config.params.dt.minimum_rel * hf_row['Time'] * 0.01  # allow small steps
        dtswitch = max(dtswitch, dtminimum)

    # Apply the SPIDER-retry step scale factor uniformly to all branches.
    # This fixes a bug where "static" (Time < 2 yr) and "initial" retries
    # silently ignored step_sf, so each retry tried the same macro step
    # and only tightened tolerances — never actually shrinking dt. See
    # next_step_retry notes in spider.py (max_attempts = 8).
    dtswitch *= step_sf

    # Always enforce the absolute maximum
    dtswitch = min(dtswitch, config.params.dt.maximum)

    # Mushy-regime dt cap (2026-04-09). Independently tightens dt
    # when the interior is actively solidifying, because the
    # phase-boundary Jgrav + rheology contrast creates stiffness
    # cliffs the output-based adaptive controller above cannot
    # detect in advance. Active when:
    #   (1) mushy_maximum > 0 in the config (feature enabled),
    #   (2) Phi_global is inside the mushy band
    #       (stop.solid.phi_crit < Phi_global < mushy_upper).
    # Read from hf_row so it reflects the freshly-updated interior
    # state from the current iteration, not the previous one.
    mushy_max = float(config.params.dt.mushy_maximum)
    if mushy_max > 0.0:
        phi_now = float(hf_row.get('Phi_global', 1.0))
        phi_floor = float(config.params.stop.solid.phi_crit)
        phi_ceiling = float(config.params.dt.mushy_upper)
        if phi_floor < phi_now < phi_ceiling:
            if dtswitch > mushy_max:
                log.info(
                    'Time-stepping: mushy cap active '
                    '(Phi_global=%.4f in (%.3f, %.3f)), '
                    'capping dt at %.2e yr (was %.2e yr)',
                    phi_now, phi_floor, phi_ceiling, mushy_max, dtswitch,
                )
                dtswitch = mushy_max

    # On retries (step_sf < 1) in the static/initial branches we
    # deliberately allow dt to fall below dt.minimum — the whole point of
    # a retry is to shrink the step below what would otherwise be allowed.
    # In the adaptive branch, min has already been enforced before retry
    # scaling, so there is no additional floor to apply here.

    log.info('New time-step target is %.2e years' % dtswitch)
    return dtswitch
