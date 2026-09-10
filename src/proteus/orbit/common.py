# Common tides model functions
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

from proteus.config import Config
from proteus.interior_energetics.common import Interior_t

log = logging.getLogger('fwl.' + __name__)


@dataclass
class TidalInteraction:
    primary: Any
    perturber: Any

    nmk: Optional[NDArray[np.floating]] = None
    sigma: Optional[NDArray[np.floating]] = None
    LNk: Optional[NDArray[np.floating]] = None


@dataclass
class Tides_t:
    interactions: List[TidalInteraction] = field(default_factory=list)

    def add(self, primary, perturber):
        try:
            return self.get(primary, perturber)
        except KeyError:
            interaction = TidalInteraction(primary, perturber)
            self.interactions.append(interaction)
            return interaction

    def get(self, primary, perturber):
        for interaction in self.interactions:
            if interaction.primary == primary and interaction.perturber == perturber:
                return interaction
        raise KeyError(f'No tidal interaction: {primary} <- {perturber}')

    def add_from_file(self, primary, perturber, file_path: str):
        interaction = self.add(primary, perturber)

        with nc.Dataset(file_path, 'r') as ds:
            n = ds.variables['n'][:]
            m = ds.variables['m'][:]
            k = ds.variables['k'][:]

            interaction.nmk = np.column_stack([n, m, k]).astype(int)
            interaction.sigma = ds.variables['sigma'][:]
            interaction.LNk = ds.variables['LNk_real'][:] + 1j * ds.variables['LNk_imag'][:]

        return interaction


def get_C_planet(hf_row: dict, config: Config, interior_o: Interior_t):
    """Compute the planet's principal moment of inertia (C_int) based on the interior structure.

    Note: This function should live in the interior_energetics module, but is currently here for
    convenience. It may be moved in the future. When moving this, ensure that the smoothing of the
    structural C_planet update in run_adaptive_orbit_substeps() is preserved.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        interior_o : Interior_t
            Interior object containing interior arrays
    """
    # Calculate the planet's principal moment of inertia (C_planet)
    # Assuming a spherically symmetric mass distribution, we can use the formula:
    # C = (8/3) * pi * integral_0^R (rho(r) * r^4 dr)
    # where rho(r) is the density profile and R is the radius of the planet.

    # Get the radial grid and density profile from the interior object
    arr_keys = ('density', 'radius')
    lov = {k: np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}
    core_density = hf_row.get('core_density', None)
    if core_density is None:
        core_density = config.interior_struct.core_density
        # Warn user
        log.warning(
            'core_density not found in hf_row; using config value: %.3e kg/m^3',
            core_density,
        )

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == 'spider':
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    # Include the core density as the innermost layer
    r_edges = np.concatenate(([0.0], lov['radius']))
    rho = np.concatenate(([core_density], lov['density']))

    r0 = r_edges[:-1]
    r1 = r_edges[1:]

    integral = np.sum(rho * (r1**5 - r0**5) / 5.0)

    C_planet = (8 * np.pi / 3.0) * integral

    # Store C_planet in the helpfile row for later use
    hf_row['C_int'] = C_planet

    # Check if C_planet is physically reasonable
    C_factor_planet = C_planet / (hf_row['M_int'] * hf_row['R_int'] ** 2)
    log.info(
        f'Computed C_planet: {C_planet:.3e} kg.m^2, C_factor_planet: {C_factor_planet:.3f}'
    )


def run_adaptive_orbit_substeps(
    hf_row: dict,
    config: Config,
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
    adaptive accept/reject substep controller.

    Shared between the star-planet models (sp0d, sp1d) and the
    planet-satellite models (ps0d, ps1d, ps1d_evec): each call site
    supplies the model-specific pieces (the ODE step itself, state
    validity, and which quantities to track for the accept/reject
    gradient check) as callables, and this function owns the substep
    loop, the angular-momentum-conserving C_planet rescale, and the
    persistence of the controller's own step-size state across calls.

    Parameters
    ----------
    hf_row : dict
        Dictionary of current runtime variables. Mutated in place.
    config : Config
        Model configuration; reads `config.orbit.solver` for every
        tolerance and controller knob below.
    interior_o : Interior_t
        Interior object; `interior_o.dt` is the total elapsed time this
        call must advance the system by, in years.
    model : str
        Name of the active model, used only for log messages.
    step_fn : callable(hf_row, dt_yr, t_elapsed_yr) -> Any
        Advances `hf_row` in place by `dt_yr` (already converted inside
        the callee as needed). `t_elapsed_yr` is how much of this call's
        `interior_o.dt` has already been accepted, for callees that need
        an absolute-time label (e.g. ps1d_evec's fine-grained storage).
        May return arbitrary "extra" data; that return value is forwarded
        to `on_accept_fn` only if the substep is subsequently accepted,
        and discarded otherwise.
    state_is_valid_fn : callable(hf_row) -> bool
        Returns whether the state `step_fn` produced is physical. A
        False return (or an exception from `step_fn`) rejects the
        substep: `hf_row` is restored to its pre-substep snapshot and
        `dt_yr` is shrunk before retrying.
    rel_change_fn : callable(hf_row, snapshot) -> dict[str, float]
        Returns named relative-change metrics between the post-substep
        state and the pre-substep snapshot (e.g. `{'da': ..., 'de':
        ...}`). Keys must match `rel_change_limits`.
    rel_change_limits : dict[str, float]
        Maximum tolerated value for each key `rel_change_fn` returns.
        Exceeding any of them rejects the substep, same as a failed
        `state_is_valid_fn` check. The substep size is only grown once
        every value is comfortably inside its limit (below 30% of it).
    needs_c_planet : bool
        Whether this model reads `hf_row['C_int']`. If True, this
        function refreshes it from the live interior state once before
        the substep loop starts, then SMOOTHLY ramps `hf_row['C_int']`
        from its call-start value to that freshly-computed target across
        the substep loop (linear in elapsed time), rescaling the planet's
        spin (`axial_period`) at every accepted substep to conserve
        `C_planet * Omega_p` across that substep's own small slice of the
        move (the "figure skater" effect of a mass-conserving structural
        change) -- see `get_C_planet`'s own docstring for the physical
        justification and its caveat, and "Smoothing the structural
        C_planet update" below for why this is spread out rather than
        applied as one jump.
    on_accept_fn : callable(hf_row, extra) -> None, optional
        Called once a substep is confirmed accepted, with the `extra`
        value `step_fn` returned for that substep. Used for side effects
        that must not run on a subsequently-rejected trial, such as
        ps1d_evec's fine-grained solver-clock CSV storage.
    log_label : str, optional
        Prefix used in log messages, so ENTER/EXIT/reject lines from the
        two call sites (star-planet vs planet-satellite) are
        distinguishable in a shared log stream.

    Smoothing the structural C_planet update
    ------------------------------------------
    A naive implementation would apply the whole C_planet jump (and the
    resulting Omega_p rescale) once, before the substep loop starts. That
    is fine as far as angular-momentum bookkeeping goes, but it means
    every OTHER quantity that depends on Omega_p -- e.g. ps1d_evec's
    planetary-oblateness (J2) term in the apsidal-precession rate, and
    hence the evection-resonance location a_res itself (see
    `satellite.compute_a_res_prime`) -- sees a discontinuous jump at
    t_elapsed=0 of this call, even though the rest of the orbital state
    (a, e, ...) evolves smoothly through the same call. That is a
    numerical artifact of how PROTEUS happens to chunk interior-orbit
    coupling into calls, not a real physical discontinuity.

    Instead, `C_planet` is ramped linearly in elapsed time from its
    call-start value to the freshly-computed target across the accepted
    substeps of this call: at each substep, `hf_row['C_int']` is moved
    to the linearly-interpolated value for `t_elapsed + dt_yr`, and
    `axial_period` is rescaled to conserve `C_planet * Omega_p` across
    just that slice (not the whole jump). Composing many small exact
    rescales this way is itself exactly angular-momentum-conserving
    end to end (the per-substep ratios telescope to the same overall
    ratio a single jump would have applied), so nothing about system AM
    bookkeeping is loosened -- the total structural correction is
    unchanged, only spread out in time so no other quantity sees a step
    function. If a call cannot complete within `solver.max_substeps` /
    the step-size floor, the ramp is simply left partway through
    (`hf_row['C_int']` still short of the target); the next call
    recomputes a fresh target from the (by-then-updated) interior state
    and continues the ramp toward that, so no correction is lost or
    double-applied, only delayed -- the same partial-completion behaviour
    the rest of this controller already has (see the "only advanced ..."
    warning below).

    This structural ramp is entirely orthogonal to whatever a model's own
    ODE (`step_fn`) does to Omega_p/spin angular momentum within the same
    substep -- in particular, the evection-resonance torque inside
    ps1d_evec, which is expected to (and is allowed to) change the
    planet-satellite subsystem's own total angular momentum via a real
    three-body exchange with the star (see ps1d_evec's own docstring/
    tests). The two effects compose without either overriding the other:
    this ramp only ever conserves `C_planet * Omega_p` across the
    non-tidal, non-resonance structural change, exactly as the
    unsmoothed single-jump version did; it does not touch, and does not
    constrain, whatever AM change `step_fn` itself produces afterward.
    """
    solver = config.orbit.solver

    # Specify the initial timestep size
    dt_yr = hf_row.pop('_orbit_dt_yr', solver.dt0_yr)

    # Setup the solver clock timescales
    t_total_yr = interior_o.dt
    dt_max = solver.dt_max_yr if solver.dt_max_yr is not None else t_total_yr
    dt_yr = min(dt_yr, t_total_yr) if t_total_yr > 0 else dt_yr

    # Accumulators
    t_elapsed = 0.0
    n_steps = 0
    n_rejected = 0

    log.info(
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

    def _rescale_c_planet_to(target_c_p):
        """Move hf_row['C_int'] to target_c_p, rescaling axial_period
        to conserve C_int*Omega_p across the move (a no-op on the
        rescale, besides writing target_c_p, if there is no valid prior
        C_int/axial_period to conserve against)."""
        c_p_before = hf_row.get('C_int')
        if (
            c_p_before is not None
            and np.isfinite(c_p_before)
            and c_p_before > 0
            and np.isfinite(target_c_p)
            and target_c_p != 0
        ):
            omega_p_before = 2 * np.pi / float(hf_row['axial_period'])
            hf_row['axial_period'] = 2 * np.pi / (omega_p_before * c_p_before / target_c_p)
        hf_row['C_int'] = target_c_p

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
                _rescale_c_planet_to(C_p_call_target)
            else:
                # Restore the call-start value so the substep loop below
                # ramps toward the target instead of landing on it in
                # one jump.
                hf_row['C_int'] = C_p_call_start
        except Exception:
            log.error(
                '%s: C_planet update RAISED at Time=%.6e yr (C_p_old=%r, model=%s); re-raising',
                log_label,
                float(hf_row['Time']),
                C_p_call_start,
                model,
                exc_info=True,
            )
            raise

    # Loop until the requested total time has been advanced, or until the
    # maximum number of substeps has been reached.
    while t_elapsed < t_total_yr and n_steps < solver.max_substeps:
        dt_yr = min(dt_yr, t_total_yr - t_elapsed)
        snapshot = dict(hf_row)

        try:
            with np.errstate(all='ignore'):
                # If ramping C_planet, compute the linearly-interpolated target for
                # this substep and rescale the planet's spin to conserve AM.
                if ramp_c_planet:
                    frac = min(1.0, (t_elapsed + dt_yr) / t_total_yr)
                    target_this_substep = (
                        C_p_call_start + (C_p_call_target - C_p_call_start) * frac
                    )
                    _rescale_c_planet_to(target_this_substep)

                # Run the model's ODE step for this substep, and check whether the
                # resulting state is physically valid. Extra data returned by the step
                # function is only forwarded to on_accept_fn if the substep is accepted.
                extra = step_fn(hf_row, dt_yr, t_elapsed)

            # Check whether the resulting state is physically valid. If not, restore the
            # snapshot and shrink the timestep for the next attempt.
            ok = state_is_valid_fn(hf_row)
            if not ok:
                bad_fields = {
                    k: v
                    for k, v in hf_row.items()
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
            rel_changes = rel_change_fn(hf_row, snapshot)
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

        # If current step gets rejected, then restore previous step and retry
        if not ok:
            hf_row.clear()
            hf_row.update(snapshot)
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

        # This substep is now confirmed ACCEPTED. Only now is it safe to run
        # any side effect gated on genuine acceptance.
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
    log.info(
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
    hf_row['_orbit_dt_yr'] = dt_yr
