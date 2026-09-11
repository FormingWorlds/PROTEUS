# Common tides model functions
from __future__ import annotations

import logging
from collections import ChainMap
from dataclasses import dataclass, field
from typing import Any, List, Optional

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

from proteus.config import Config
from proteus.interior_energetics.common import Interior_t, get_C_planet
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger('fwl.' + __name__)


@dataclass
class TidalInteraction:
    """Tides interaction between a primary and a perturber.
    This class stores the tidal mode information for a specific interaction
    between a primary and a perturber. It contains the tidal modes (n, m, k),
    the forcing frequencies (sigma), and the complex Love numbers (LNk).
    """

    primary: Any
    perturber: Any

    nmk: Optional[NDArray[np.int_]] = None
    sigma: Optional[NDArray[np.floating]] = None
    LNk: Optional[NDArray[np.complexfloating]] = None


@dataclass
class Tides_t:
    """Registry of tidal interactions (mode, forcing frequency, Love number)
    keyed by (primary, perturber), plus controller-only bookkeeping with no
    physical meaning of its own (adaptive step size, evection-band
    hysteresis state, fine-CSV write cursors). See "Adaptive substep
    controller" and "The three clocks" in docs/Explanations/orbit.md for what
    each of these fields feeds into.
    """

    interactions: List[TidalInteraction] = field(default_factory=list)
    dt_yr: Optional[float] = None
    fine_csv_last_t_yr: Optional[float] = None
    fine_csv_next_target_yr: Optional[float] = None
    resonance_state: dict = field(default_factory=dict)
    evection_ecc_history: List[tuple] = field(default_factory=list)
    evection_zone_active: bool = False
    evection_cooldown_remaining: int = 0

    def add(self, primary, perturber):
        """Add a new tidal interaction between the primary and the perturber.
        If an interaction already exists, it will be returned instead of creating a new one.
        """
        try:
            return self.get(primary, perturber)
        except KeyError:
            interaction = TidalInteraction(primary, perturber)
            self.interactions.append(interaction)
            return interaction

    def get(self, primary, perturber):
        """Get the tidal interaction between the primary and the perturber.
        If no such interaction exists, a KeyError is raised.
        """
        for interaction in self.interactions:
            if interaction.primary == primary and interaction.perturber == perturber:
                return interaction
        raise KeyError(f'No tidal interaction: {primary} <- {perturber}')

    def add_from_file(self, primary, perturber, file_path: str):
        """Add a new tidal interaction between the primary and the perturber,
        loading the tidal mode information from a NetCDF file.
        """
        interaction = self.add(primary, perturber)

        with nc.Dataset(file_path, 'r') as ds:
            n = ds.variables['n'][:]
            m = ds.variables['m'][:]
            k = ds.variables['k'][:]

            interaction.nmk = np.column_stack([n, m, k]).astype(int)
            interaction.sigma = ds.variables['sigma'][:]
            interaction.LNk = ds.variables['LNk_real'][:] + 1j * ds.variables['LNk_imag'][:]

        return interaction


def run_adaptive_orbit_substeps(
    hf_row: dict,
    config: Config,
    dirs: dict,
    tides_o: Tides_t,
    interior_o: Interior_t,
    model: str,
    step_fn,
    state_is_valid_fn,
    rel_change_fn,
    rel_change_limits: dict,
    needs_c_planet: bool,
    on_accept_fn=None,
    log_label: str = 'run_adaptive_orbit_substeps',
):
    """Advance an orbital-evolution model by `interior_o.dt` yr using an
    adaptive accept/reject substep controller, shared by sp1d/ps1d/ps1d_evec
    (ps0d bypasses this and applies its structural update as a single jump).
    Each call site supplies the model-specific ODE step, state-validity
    check, and relative-change metrics as callables; this function owns the
    substep loop, the angular-momentum-conserving `C_int` ramp, and the
    persistence of the step-size state across calls. See "Adaptive substep
    controller" in docs/Explanations/orbit.md for the physical and
    numerical rationale (why C_int is ramped rather than jumped, and why
    attempts are staged in a `ChainMap` overlay rather than mutating
    `hf_row` directly).

    Parameters
    ----------
    hf_row : dict
        Runtime state; mutated only once a substep is accepted.
    config : Config
        Reads `config.orbit.solver` for every tolerance/controller knob.
    dirs : dict
        Directory paths, used only for logging and error messages.
    tides_o : Tides_t
        `tides_o.dt_yr` carries the controller's step-size state across
        calls (one call per PROTEUS main-loop iteration).
    interior_o : Interior_t
        `interior_o.dt` is the total elapsed time to advance by, in years.
    model : str
        Name of the active model, used only for log messages.
    step_fn : callable(attempt, dt_yr, t_elapsed_yr) -> Any
        Advances `attempt` in place by `dt_yr`. May return arbitrary
        "extra" data, forwarded to `on_accept_fn` only if the substep is
        accepted.
    state_is_valid_fn : callable(attempt) -> bool
        Whether the state `step_fn` produced is physical; False (or an
        exception from `step_fn`) rejects and shrinks the substep.
    rel_change_fn : callable(attempt, hf_row) -> dict[str, float]
        Named relative-change metrics; keys must match `rel_change_limits`.
    rel_change_limits : dict[str, float]
        Maximum tolerated value per key; exceeding any rejects the substep.
        Growth only occurs once every value is below 30% of its limit.
    needs_c_planet : bool
        Whether this model reads `hf_row['C_int']`; if True, `C_int` is
        ramped from its call-start value to a freshly computed target
        across the substep loop, with `axial_period` rescaled at each
        accepted substep to conserve `C_int * Omega_p`.
    on_accept_fn : callable(hf_row, extra) -> None, optional
        Called once a substep is confirmed accepted, for side effects that
        must not run on a rejected trial (e.g. ps1d_evec's fine-CSV write).
    log_label : str, optional
        Prefix for log messages, to distinguish the two call sites.
    """
    solver = config.orbit.solver

    # Specify the initial timestep size
    dt_yr = tides_o.dt_yr if tides_o.dt_yr is not None else solver.dt0_yr

    # Setup the solver clock timescales
    t_total_yr = interior_o.dt
    dt_max = solver.dt_max_yr if solver.dt_max_yr is not None else t_total_yr
    dt_yr = min(dt_yr, t_total_yr) if t_total_yr > 0 else dt_yr

    # Accumulators
    t_elapsed = 0.0
    n_steps = 0
    n_rejected = 0

    log.debug(
        '%s: ENTER model=%s Time=%.6e yr t_total_yr=%.6e dt_yr_start=%.3e',
        log_label,
        model,
        float(hf_row['Time']),
        t_total_yr,
        dt_yr,
    )

    # Refresh the planet's moment-of-inertia coefficient from the current
    # interior state. The resulting jump (if any) is smoothed across the
    # substep loop below rather than applied here -- see "Smoothing the
    # structural C_planet update" in this function's own docstring.
    C_p_call_start = None
    C_p_call_target = None
    ramp_c_planet = False

    def _rescale_c_planet_to(row, target_c_p):
        """Move row['C_int'] to target_c_p, rescaling axial_period to
        conserve C_int*Omega_p (a no-op on the rescale if there is no
        valid prior C_int/axial_period to conserve against)."""
        c_p_before = row.get('C_int')
        if (
            c_p_before is not None
            and np.isfinite(c_p_before)
            and c_p_before > 0
            and np.isfinite(target_c_p)
            and target_c_p != 0
        ):
            omega_p_before = 2 * np.pi / float(row['axial_period'])
            row['axial_period'] = 2 * np.pi / (omega_p_before * c_p_before / target_c_p)
        row['C_int'] = target_c_p

    if needs_c_planet:
        C_p_call_start = hf_row.get('C_int')

        try:
            get_C_planet(hf_row, config, interior_o)
            C_p_call_target = hf_row['C_int']

            if not np.isfinite(C_p_call_target) or C_p_call_target == 0:
                log.error(
                    '%s: get_C_planet produced C_p_new=%r (C_p_old=%r) at '
                    'Time=%.6e yr -- structural update will be skipped',
                    log_label,
                    C_p_call_target,
                    C_p_call_start,
                    float(hf_row['Time']),
                )

            ramp_c_planet = (
                C_p_call_start is not None
                and np.isfinite(C_p_call_start)
                and C_p_call_start > 0
                and np.isfinite(C_p_call_target)
                and C_p_call_target != 0
                and t_total_yr > 0
            )
            if not ramp_c_planet:
                # No prior value to ramp from (bootstrap), a degenerate
                # get_C_planet result (logged above), or a zero-length
                # call (no substeps will run to do the ramping): apply
                # the full jump immediately, same as the un-smoothed
                # behaviour in all three cases.
                _rescale_c_planet_to(hf_row, C_p_call_target)
            else:
                # Restore the call-start value so the substep loop below
                # ramps toward the target instead of landing on it in
                # one jump.
                hf_row['C_int'] = C_p_call_start
        except Exception as err:
            log.error(
                '%s: C_planet update RAISED at Time=%.6e yr (C_p_old=%r, model=%s); re-raising',
                log_label,
                float(hf_row['Time']),
                C_p_call_start,
                model,
                exc_info=True,
            )
            UpdateStatusfile(dirs, 26)
            raise RuntimeError(
                f'C_planet update failed for {log_label} at Time={float(hf_row["Time"]):.6e} yr '
                f'(model={model}, C_p_old={C_p_call_start!r})'
            ) from err

    # Loop until the requested total time has been advanced, or until the
    # maximum number of substeps has been reached.
    while t_elapsed < t_total_yr and n_steps < solver.max_substeps:
        dt_yr = min(dt_yr, t_total_yr - t_elapsed)

        # Stage this substep's tentative changes in an overlay rather than
        # mutating hf_row directly, so a rejected attempt costs nothing to
        # discard (docs/Explanations/orbit.md, "Adaptive substep controller").
        attempt = ChainMap({}, hf_row)

        try:
            with np.errstate(all='ignore'):
                # If ramping C_planet, compute the linearly-interpolated target for
                # this substep and rescale the planet's spin to conserve AM.
                if ramp_c_planet:
                    frac = min(1.0, (t_elapsed + dt_yr) / t_total_yr)
                    target_this_substep = (
                        C_p_call_start + (C_p_call_target - C_p_call_start) * frac
                    )
                    _rescale_c_planet_to(attempt, target_this_substep)

                # Run the model's ODE step for this substep, and check whether the
                # resulting state is physically valid. Extra data returned by the step
                # function is only forwarded to on_accept_fn if the substep is accepted.
                extra = step_fn(attempt, dt_yr, t_elapsed)

            # Check whether the resulting state is physically valid. If not, discard
            # this attempt (hf_row was never touched) and shrink the timestep.
            ok = state_is_valid_fn(attempt)
            if not ok:
                bad_fields = {
                    k: v
                    for k, v in attempt.items()
                    if isinstance(v, (int, float)) and not np.isfinite(v)
                }
                log.warning(
                    '%s: state_is_valid_fn rejected the state at '
                    't_elapsed=%.6e/%.6e yr, dt_yr=%.3e; non-finite fields=%r',
                    log_label,
                    t_elapsed,
                    t_total_yr,
                    dt_yr,
                    bad_fields,
                )
        except Exception:
            log.warning(
                '%s: substep raised (model=%s, dt_yr=%.3e, t_elapsed=%.6e/%.6e yr)',
                log_label,
                model,
                dt_yr,
                t_elapsed,
                t_total_yr,
                exc_info=True,
            )
            ok = False
            extra = None

        # Check the tracked quantities' relative change against their limits.
        rel_changes = {}
        if ok:
            rel_changes = rel_change_fn(attempt, hf_row)
            tripped = [
                f'{key}={value:.4g}>{rel_change_limits[key]:.4g}'
                for key, value in rel_changes.items()
                if value > rel_change_limits[key]
            ]
            if tripped:
                ok = False
                log.debug(
                    '%s: reject at t_elapsed=%.6e yr, dt_yr=%.3e -- tripped: %s',
                    log_label,
                    t_elapsed,
                    dt_yr,
                    ', '.join(tripped),
                )

        # If current step gets rejected, discard the attempt and retry with a
        # smaller dt_yr; hf_row was never mutated, so there is nothing to restore.
        if not ok:
            dt_yr *= solver.shrink
            n_rejected += 1

            if dt_yr < 1e-10:
                log.warning(
                    '%s: internal step size collapsed to zero at t=%.3e yr of a '
                    '%.3e yr requested call; stopping early (n_steps=%d, '
                    'n_rejected=%d, last rel_changes=%r)',
                    log_label,
                    t_elapsed,
                    t_total_yr,
                    n_steps,
                    n_rejected,
                    rel_changes,
                )
                break
            continue

        # This substep is now confirmed ACCEPTED. Merge only the keys this
        # attempt actually wrote into the real hf_row, then run any side
        # effect gated on genuine acceptance.
        hf_row.update(attempt.maps[0])

        if on_accept_fn is not None:
            on_accept_fn(hf_row, extra)

        # Update the elapsed time and step count, and log progress every 5000 accepted steps.
        t_elapsed += dt_yr
        n_steps += 1

        if n_steps % 5000 == 0:
            log.debug(
                '%s: progress t_elapsed=%.6e/%.6e yr n_steps=%d n_rejected=%d dt_yr=%.3e',
                log_label,
                t_elapsed,
                t_total_yr,
                n_steps,
                n_rejected,
                dt_yr,
            )

        # Adaptively grow the timestep once every tracked quantity is
        # comfortably inside its limit.
        if all(value < 0.3 * rel_change_limits[key] for key, value in rel_changes.items()):
            dt_yr = min(dt_yr * solver.growth, dt_max)

    # Log a warning if the requested total time was not fully advanced, but do not raise an exception:
    # the caller may have requested a very large time step that cannot be completed in a single call,
    # and the controller is designed to handle that gracefully.
    if t_elapsed < t_total_yr - 1e-9:
        log.warning(
            '%s: only advanced %.3e of the requested %.3e yr (%d accepted / '
            '%d rejected internal steps) before hitting max_substeps=%d',
            log_label,
            t_elapsed,
            t_total_yr,
            n_steps,
            n_rejected,
            solver.max_substeps,
        )

    # Log the exit status of this call, including the final elapsed time, number of steps, and whether
    # the requested total time was fully advanced.
    log.debug(
        '%s: EXIT model=%s t_elapsed=%.6e/%.6e yr n_steps=%d n_rejected=%d '
        'dt_yr_final=%.3e complete=%s',
        log_label,
        model,
        t_elapsed,
        t_total_yr,
        n_steps,
        n_rejected,
        dt_yr,
        t_elapsed >= t_total_yr - 1e-9,
    )

    # Persist the controller's own step-size state for the next call.
    tides_o.dt_yr = dt_yr
