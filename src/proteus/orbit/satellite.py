from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from proteus.interior_energetics.common import Interior_t, get_C_planet
from proteus.orbit.common import Tides_t, run_adaptive_orbit_substeps
from proteus.orbit.hansen import get_all_m_hansen
from proteus.orbit.timestep import _estimate_evection_dt_cap_yr
from proteus.utils.constants import R_earth, const_G, secs_per_year
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def _state_is_valid(hf_row):
    """Reject a substep whose resulting state is unphysical or non-finite.

    Checks the satellite hasn't spiralled below/through the planet's
    surface (a <= 1.05 R_planet), that eccentricity is in the physically
    sane, sub-parabolic range [0, 0.999), and that both spin periods are
    finite. Used by `evolve_orbit_satellite`'s accept/reject controller;
    a False return discards the tentative substep and triggers a smaller
    retry `dt_yr`.
    """
    a = hf_row.get('semimajorax_sat', np.nan)
    e = hf_row.get('eccentricity_sat', 0.0)
    if not np.isfinite(a) or a <= 1.05 * hf_row.get('R_planet', 0.0):
        return False
    if not np.isfinite(e) or e < 0.0 or e >= 0.999:
        return False
    for key in ('axial_period', 'axial_period_sat'):
        val = hf_row.get(key, np.nan)
        if val is not None and not np.isfinite(val):
            return False
    return True


def _in_evection_band(hf_row, resonance_state, margin_enter=0.10, margin_exit=0.35):
    """Debounced, hysteretic near-resonance detector.

    resonance_state is a small dict the CALLER owns and must pass back in
    unchanged on every call, it carries the 2-step history and the
    current on/off state. Mutated in place. As a side effect, this also
    stashes the latest raw (unsmoothed, undebounced) relative distance
    under ``resonance_state['d_a_rel_now']`` used by the caller to
    derive the wider, purely-diagnostic ``near_evection_band`` signal (see
    ``evolve_orbit_satellite``), without recomputing ``compute_a_res_prime``
    a second time.

    Returns True if the satellite is judged to currently be "in" the
    evection resonance band (i.e. the oscillating-filter term should be
    active), False otherwise.
    """
    a_prime_now = hf_row['semimajorax_sat'] / R_earth
    a_res_now = compute_a_res_prime(hf_row)

    # Check resonance history and current state
    hist = resonance_state.setdefault('hist_d_a_rel', [])
    active = resonance_state.get('active', False)

    # Check if resonance is defined
    if not np.isfinite(a_res_now) or a_res_now == 0:
        # Reset the resonance state and history if resonance is not defined
        resonance_state['active'] = False
        resonance_state['d_a_rel_now'] = np.inf
        hist.clear()
        return False

    # Calculate the relative distance
    d_a_rel = (a_prime_now - a_res_now) / a_res_now
    # Update the resonance state and history
    resonance_state['d_a_rel_now'] = d_a_rel
    hist.append(d_a_rel)
    # Keep only the last two entries in the history
    if len(hist) > 2:
        hist.pop(0)

    # Determine the margin based on the current active state
    margin = margin_exit if active else margin_enter
    d_smoothed = float(np.mean(hist))
    smoothed_inside = abs(d_smoothed) <= margin

    # If we have two history points...
    if len(hist) == 2:
        # ...check if both history points are within 1.5 times the margin
        both_in_neighbourhood = all(abs(v) <= margin * 1.5 for v in hist)
    else:
        # ...if we have only one history point, just check that point
        both_in_neighbourhood = smoothed_inside

    # Update the active state based on the smoothed value and the neighbourhood check
    resonance_state['active'] = bool(smoothed_inside and both_in_neighbourhood)
    return resonance_state['active']


def _flush_fine_evection_csv(
    tides_o, data_dir, fine_entry, in_band, storage_target_interval_yr
):
    """Append ONE accepted ps1d_evec macro-step's fine samples to disk, at
    the "storage clock" rate (see "The three clocks" in
    docs/Explanations/orbit.md). This is the only place
    `fine_evection_data.csv` is written, and must only be called AFTER a
    substep has been confirmed accepted, never from inside the solver
    itself, which cannot know if its result will later be rejected.

    Each candidate sample passes two filters: (1) dedup against
    `tides_o.fine_csv_last_t_yr`, the last timestamp actually written,
    dropping any sample that does not advance past it; (2) storage-clock
    density, every deduped sample is kept while `in_band`, otherwise
    only once `t_abs_yr` reaches `tides_o.fine_csv_next_target_yr`, which
    then advances to `t_abs_yr + storage_target_interval_yr`.

    Parameters
    ----------
        tides_o : Tides_t
            Read/written for the two persisted cursors described above.
        data_dir : str
            Directory containing (or to contain) fine_evection_data.csv.
        fine_entry : dict
            One entry from `ps1d_evec`'s `fine_sink` list: equal-length
            1-D arrays keyed by 't_abs_yr', 'omega_p', 'omega_s', 'sma',
            'ecc', 'phi', 'da_planet_tide_cum', 'da_sat_tide_cum',
            'de_planet_tide_cum', 'de_sat_tide_cum', 'filter'.
        in_band : bool
            Whether this accepted substep was inside the evection band;
            selects which storage-clock policy applies.
        storage_target_interval_yr : float
            Target spacing [yr] between stored out-of-band samples for
            the current `evolve_orbit_satellite` call. Unused if in_band.
    """
    t_abs_yr = fine_entry['t_abs_yr']
    if len(t_abs_yr) == 0:
        return

    # Get the last timestamp actually written and the next storage-clock target
    last_t = tides_o.fine_csv_last_t_yr if tides_o.fine_csv_last_t_yr is not None else -np.inf
    next_target = (
        tides_o.fine_csv_next_target_yr
        if tides_o.fine_csv_next_target_yr is not None
        else -np.inf
    )

    # Determine which samples to keep based on the storage-clock
    keep = np.zeros(len(t_abs_yr), dtype=bool)
    for i, t in enumerate(t_abs_yr):
        if t <= last_t:
            continue
        if in_band:
            keep[i] = True
        elif t >= next_target:
            keep[i] = True
            next_target = t + storage_target_interval_yr

    # Persist the storage-clock cursor (even on calls that end up keeping nothing)
    tides_o.fine_csv_next_target_yr = next_target

    # If no samples passed the storage-clock filter, return early
    if not np.any(keep):
        return

    # Collect the filtered samples into a 2D array for CSV writing
    fine_data = np.column_stack(
        (
            t_abs_yr[keep],
            fine_entry['omega_p'][keep],
            fine_entry['omega_s'][keep],
            fine_entry['sma'][keep],
            fine_entry['ecc'][keep],
            fine_entry['phi'][keep],
            fine_entry['da_planet_tide_cum'][keep],
            fine_entry['da_sat_tide_cum'][keep],
            fine_entry['de_planet_tide_cum'][keep],
            fine_entry['de_sat_tide_cum'][keep],
            fine_entry['filter'][keep],
        )
    )

    # Append the filtered samples to the CSV file, creating it with a header if it doesn't exist
    fine_file_path = os.path.join(data_dir, 'fine_evection_data.csv')
    file_exists = os.path.exists(fine_file_path)
    with open(fine_file_path, 'a') as f:
        if not file_exists:
            f.write(
                't_abs_yr,omega_p,omega_s,sma,ecc,phi,'
                'da_planet_tide_cum,da_sat_tide_cum,'
                'de_planet_tide_cum,de_sat_tide_cum,filter\n'
            )
        np.savetxt(f, fine_data, delimiter=',', fmt='%.8e')

    # Advance the cursor to the last timestamp actually written
    tides_o.fine_csv_last_t_yr = float(t_abs_yr[keep][-1])


def evolve_orbit_satellite(
    hf_row: dict,
    config: Config,
    dirs: dict,
    tides_o: Tides_t,
    interior_o: Interior_t,
):
    """Advance the satellite's orbital parameters over `interior_o.dt`
    years. Dispatches to the requested model (`ps0d`, `ps1d`, `ps1d_evec`);
    `ps1d`/`ps1d_evec` use the shared adaptive substep controller (see
    docs/Explanations/orbit.md).

    TODO: Assumes constant planetary mass, if atmospheric mass loss occurs
    between steps, escaping angular momentum must be handled as an explicit
    sink rather than absorbed here.

    Parameters
    ----------
        hf_row : dict
            Current runtime state variables.
        config : Config
            System configuration options.
        dirs : dict
            Output and data directory paths.
        tides_o : Tides_t
            Container for tidal interaction parameters.
        interior_o : Interior_t
            Interior model state; `interior_o.dt` defines total elapsed integration time (years).
    """
    model = config.orbit.planet_satellite_model
    solver = config.orbit.solver

    if model == 'ps0d':
        # Bypasses run_adaptive_orbit_substeps entirely: ps0d has no
        # spin-orbit-tidal stiffness. Preserves the AM-conserving
        # structural rescale, applied as a single unsmoothed jump.
        C_p_old = hf_row.get('C_int')
        get_C_planet(hf_row, config, interior_o)
        C_p_new = hf_row['C_int']
        if (
            C_p_old is not None
            and np.isfinite(C_p_old)
            and C_p_old > 0
            and np.isfinite(C_p_new)
            and C_p_new != 0
        ):
            Omega_p_old = 2 * np.pi / float(hf_row['axial_period'])
            hf_row['axial_period'] = 2 * np.pi / (Omega_p_old * C_p_old / C_p_new)
        ps0d(hf_row, interior_o.dt, config)
        return

    # Get the evection resonance state from tides_o
    resonance_state = tides_o.resonance_state

    # Get the total integration time and the start time of the current window
    t_total_yr = interior_o.dt
    t_window_start_abs_yr = float(hf_row['Time']) - t_total_yr

    # Target spacing [yr] between stored out-of-band fine samples for this call
    storage_target_interval_yr = solver.fine_csv_target_rel_dt * t_total_yr

    last_in_band = [None]  # mutable box: tracks band transitions for logging only

    if model == 'ps1d':
        # Define the integrator: ps1d
        def step_fn(hf_row, dt_yr, t_elapsed_yr):
            ps1d(hf_row, tides_o, dt_yr, config)
            return None

        # No extra action on accept for ps1d
        on_accept_fn = None

    elif model == 'ps1d_evec':
        # Define the integrator: ps1d_evec
        def step_fn(hf_row, dt_yr, t_elapsed_yr):
            # Check if in evection resonance band
            in_band = _in_evection_band(
                hf_row,
                resonance_state,
                margin_enter=solver.resonance_margin_enter,
                margin_exit=solver.resonance_margin_exit,
            )

            # Log band transitions
            if last_in_band[0] is not None and in_band != last_in_band[0]:
                log.debug(
                    'evolve_orbit_satellite: evection-band TRANSITION '
                    '%s -> %s at t_elapsed=%.6e/%.6e yr '
                    '(a=%.6g, e=%.4f, dt_yr=%.3e)',
                    last_in_band[0],
                    in_band,
                    t_elapsed_yr,
                    t_total_yr,
                    hf_row.get('semimajorax_sat', float('nan')),
                    hf_row.get('eccentricity_sat', float('nan')),
                    dt_yr,
                )
            # Update the last_in_band state for the next call
            last_in_band[0] = in_band

            # If in resonance band, then include evection resonance terms
            filter_value = 1.0 if in_band else 0.0

            # Define current time in the solver clock
            substep_start_abs_yr = t_window_start_abs_yr + t_elapsed_yr

            # Determine if the next substep will cross the storage-clock target,
            # and if so, prepare to store fine samples
            next_storage_target_yr = (
                tides_o.fine_csv_next_target_yr
                if tides_o.fine_csv_next_target_yr is not None
                else -np.inf
            )
            might_cross_target = (substep_start_abs_yr + dt_yr) >= next_storage_target_yr
            # Prepare a list to collect fine samples if needed
            fine_sink = [] if (in_band or might_cross_target) else None

            # Run the Planet-Satellite-1D model with evection resonance
            ps1d_evec(
                hf_row,
                tides_o,
                dt_yr,
                config,
                fine_sink,
                20,
                filter_value,
                t_abs_start_yr=substep_start_abs_yr,
            )
            return (fine_sink, in_band)

        # Define the on_accept function to flush fine samples to disk after a substep is accepted
        def on_accept_fn(hf_row, extra):
            # This substep is now confirmed ACCEPTED (state valid, all
            # rel_change_* within tolerance). Only now is it safe to
            # persist its fine-grained samples.
            fine_sink, in_band = extra
            if fine_sink:
                # Flush the fine samples to disk
                _flush_fine_evection_csv(
                    tides_o,
                    dirs['output/data'],
                    fine_sink[0],
                    in_band=bool(in_band),
                    storage_target_interval_yr=storage_target_interval_yr,
                )

    else:
        # Unrecognized model: raise an error and update the status file
        UpdateStatusfile(dirs, 26)
        raise ValueError(f'unrecognised planet_satellite_model: {model!r}')

    # Define the relative-change function for the adaptive substep controller
    def rel_change_fn(attempt, hf_row):
        # This function computes the relative changes in key orbital parameters between
        # the current attempt and the previous state, which are used to determine if the
        # substep is acceptable.

        # A relative-change ratio needs a finite, nonzero prior value.
        with np.errstate(divide='ignore', invalid='ignore'):
            # Compute semimajor-axis gradient
            a_prev = hf_row.get('semimajorax_sat', np.nan)
            da = np.divide(abs(attempt['semimajorax_sat'] - a_prev), a_prev)

            # Compute eccentricity gradient
            e_prev = hf_row.get('eccentricity_sat', 0.0)
            e_new = attempt.get('eccentricity_sat', 0.0)
            de = abs(e_new - e_prev) / max(e_prev, solver.de_floor)

            # Compute planet spin rate gradient
            axp_prev = hf_row.get('axial_period', np.nan)
            axp_new = attempt.get('axial_period', np.nan)
            dOmega_p = np.divide(
                abs(np.divide(1.0, axp_new) - np.divide(1.0, axp_prev)),
                np.divide(1.0, axp_prev),
            )

            # Compute satellite spin rate gradient
            axs_prev = hf_row.get('axial_period_sat', np.nan)
            axs_new = attempt.get('axial_period_sat', np.nan)
            dOmega_s = np.divide(
                abs(np.divide(1.0, axs_new) - np.divide(1.0, axs_prev)),
                np.divide(1.0, axs_prev),
            )

        return {
            key: value
            for key, value in (
                ('da', da),
                ('de', de),
                ('dOmega_p', dOmega_p),
                ('dOmega_s', dOmega_s),
            )
            if np.isfinite(value)
        }

    # Define the relative-change limits for the adaptive substep controller,
    # based on the solver configuration
    rel_change_limits = {
        'da': solver.max_rel_da,
        'de': solver.max_rel_de,
        'dOmega_p': solver.max_rel_dOmega,
        'dOmega_s': solver.max_rel_dOmega,
    }

    # Snapshot Time before the substep controller runs
    t_call_start_yr = float(hf_row['Time'])

    # Run the adaptive substep controller, which will call the appropriate
    # step function (ps1d or ps1d_evec) and manage substep acceptance/rejection
    # based on the relative changes in orbital parameters.
    run_adaptive_orbit_substeps(
        hf_row,
        config,
        dirs,
        tides_o,
        interior_o,
        model,
        step_fn,
        _state_is_valid,
        rel_change_fn,
        rel_change_limits,
        needs_c_planet=True,
        on_accept_fn=on_accept_fn,
        log_label='evolve_orbit_satellite',
    )

    # After the adaptive substep controller has completed, update the evection resonance state
    in_band = bool(resonance_state.get('active', False))
    d_a_rel_now = resonance_state.get('d_a_rel_now', np.inf)
    near_band = abs(d_a_rel_now) <= solver.resonance_margin_approach
    tides_o.evection_zone_active = in_band or near_band

    # Evection dt cap (see docs/Explanations/orbit.md, "Evection resonance").
    ecc_window = max(2, int(getattr(config.params.dt, 'evection_rate_window', 2)))
    # Store the current eccentricity and time in the history
    tides_o.evection_ecc_history.append(
        (float(hf_row['Time']), float(hf_row['eccentricity_sat']))
    )
    # Keep only the last `ecc_window` entries in the history
    del tides_o.evection_ecc_history[:-ecc_window]

    # Compute the time since the start of this call, used for estimating the evection dt cap
    dt_prev_actual_yr = float(hf_row['Time']) - t_call_start_yr
    hf_row['evection_dt_cap_yr'] = _estimate_evection_dt_cap_yr(
        tides_o, tides_o.evection_zone_active, dt_prev_actual_yr, config
    )


def compute_a_res_prime(hf_row):
    """Compute the resonant semi-major axis (a'_res) in units of the planet's radius.
    Following Rufu & Canup (2020), below Eq 11.
    """

    Omega_planet = np.sqrt(const_G * hf_row['M_int'] / hf_row['R_int']**3)
    Omega_sun = 2 * np.pi / secs_per_year
    Lambda = np.sqrt(1.5 * 0.315 * Omega_planet / Omega_sun)

    e = hf_row['eccentricity_sat']
    s_prime = (2 * np.pi / hf_row['axial_period']) / Omega_planet
    with np.errstate(invalid='ignore'):
        return (Lambda * s_prime / (1.0 - e**2)) ** (4.0 / 7.0)


def _solve_e_stationary(a_prime, s_prime, Lambda, Omega_ratio):
    """Solve Rufu & Canup (2020), Eq. 12, for the stable stationary eccentricity e_s.

    Note:
    Rufu & Canup (2020) use the following normalization:
    - a' = a / R_p
    - s' = Omega / Omega_p
    - Lambda = sqrt(1.5 * J_star * Omega_p / Omega_star)
    - Omega_ratio = Omega_star / Omega_p

    where:
    Omega_p = sqrt(const_G * M_p / R_p**3)

    Parameters
    ----------
    a_prime : float
        Scaled semi-major axis.
    s_prime : float
        Scaled spin rate.
    Lambda : float
        Scaled angular momentum.
    Omega_ratio : float
        Ratio of the planet's spin rate to the orbital mean motion.

    Returns
    -------
    e_s : float
        The stable stationary eccentricity, or np.nan if no solution exists.
    """
    if not (np.isfinite(a_prime) and np.isfinite(s_prime)):
        return np.nan

    def f(e):
        return (
            Lambda**2 * s_prime**2 / (a_prime**3.5 * (1.0 - e**2) ** 2)
            - 1.0
            - 3.0 * np.sqrt(1.0 - e**2) * a_prime**1.5 * Omega_ratio
        )

    lo, hi = 1e-8, 1.0 - 1e-8
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan
    return brentq(f, lo, hi)


def ps0d(hf_row, dt, config):
    """Evolve the Satellite's orbital parameters module.

    Updates the semi-major axis and primary spin rate based on angular
    momentum conservation. The model is based on Korenaga (2023).

    DOI: 10.1016/j.icarus.2023.115564

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dt : float
            Time interval over which escape is occuring [yr]
        config : Config
            Configuration options; reads config.orbit.solver
    """

    def Ltot(ω, a, params):
        """Total planet+satellite angular momentum, Korenaga (2023) Eq. 60:
        spin (I*Omega) plus orbital (M_sat * sqrt(G*(M_pl+M_sat)*a)), the
        latter using the M_sat << M_pl limit of the reduced-mass formula
        (~1.2% error for Earth-Moon). Sign convention: positive L is a
        prograde satellite. See docs/Validation/orbit/satellite.md for the
        re-derivation and the discrimination guard against the M_planet
        substitution.
        """
        I, _, G, Mpl, Msa, _ = params
        return I * ω + Msa * (G * (Mpl + Msa) * a) ** 0.5

    def dω_dt(a, ω, params):
        """Planet-rotation ODE, Korenaga (2023) Eq. 58: spin slows as
        tidal energy `dE_tidal` [W] is dissipated, transferring angular
        momentum to the satellite's orbit. See
        docs/Validation/orbit/satellite.md.
        """
        I, L, G, Mpl, Msa, dE_tidal = params
        return -dE_tidal / (I * ω + (G * Mpl * Msa * I) / (a * (L - I * ω)))

    def da_dt(a, ω, params):
        """Satellite semi-major-axis ODE, Korenaga (2023) Eq. 59: follows
        from differentiating the Ltot closure (Eq. 60) at constant L. The
        orbit expands (da/dt > 0) as spin slows (dOmega/dt < 0), provided
        L > I*Omega (the prograde regime PROTEUS targets).
        """
        I, L, *_ = params
        return -2 * I * a / (L - I * ω) * dω_dt(a, ω, params)

    def orbitals(t, z, params):
        """
        Helper function for solving coupled ODEs.
        """
        a, ω = z
        return [da_dt(a, ω, params), dω_dt(a, ω, params)]

    # Set parameters from helpfile
    Rpl = hf_row['R_int']
    Mpl = hf_row['M_int']
    Msa = hf_row['M_sat']

    sma = float(hf_row['semimajorax_sat'])
    omega = 2 * np.pi / float(hf_row['axial_period'])

    L = hf_row['plan_sat_am']

    # Calculate bulk tidal power
    dE_tidal = hf_row['F_tidal'] * 4 * np.pi * Rpl**2  # Js-1

    # Planet's moment-of-inertia coefficient, from the live interior
    # state (evolve_orbit_satellite's get_C_planet + spin-rescale
    # block keeps this angular-momentum-consistent across structural
    # changes, see that function's docstring), not a fixed
    # uniform-sphere approximation unlike in Korenaga (2023).
    I = hf_row['C_int']  # kg m^2

    # Convert time to seconds
    dt = float(dt) * secs_per_year

    # Time step
    current_time = float(hf_row['Time'])

    # On the first run of this orbital module, instantiate the system angular-momentum
    if current_time <= 10 and L == 0:
        # Calculate the system angular-momentum integration constant
        # via the dedicated ``Ltot`` helper above, which implements
        # Korenaga (2023) Eq. 60 with the satellite-mass prefactor in
        # the orbital sqrt. Using the helper avoids duplicating the
        # formula and keeps any future revision in one place. Uses the
        # same interior-derived I as the ODE itself just above.
        L = Ltot(omega, sma, (I, 0, const_G, Mpl, Msa, 0))
        hf_row['plan_sat_am'] = L

    # Collect system parameters at previous_time
    params = (I, L, const_G, Mpl, Msa, dE_tidal)

    # Find new satellite semimajor axis and axial frequency using RK5(4) integration method
    solver = config.orbit.solver
    log.debug('Integrating the ps0d orbital model with solve_ivp')
    sol = solve_ivp(
        orbitals,
        [0, dt],
        [sma, omega],
        args=(params,),
        method=solver.method,
        rtol=solver.rtol,
        atol=solver.atol,
    )

    # Update semimajor axis and axial period
    hf_row['semimajorax_sat'] = sol.y[0][-1]
    hf_row['axial_period'] = 2 * np.pi / sol.y[1][-1]


def ps1d(hf_row, tides_o, dt, config):
    """Evolve planet-satellite orbit based on Correia & Valente (2022)

    Evolve both primary and perturber spin rates, semi-major axis, and eccentricity using the
    secular tidal model of Correia & Valente (2022). It assumes a vectorial approach expressed
    on Hansen coefficients.

    DOI: 10.1007/s10569-022-10079-3

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        tides_o : Tides_t
            Tides object containing tidal interactions
        dt : float
            Time interval over which escape is occuring [yr]
        config : Config
            Configuration options; reads config.orbit.solver
    """

    # Convert time to seconds
    dt = float(dt) * secs_per_year

    # Orbital parameters from helpfile
    axial_p = 2 * np.pi / float(hf_row['axial_period'])
    axial_s = 2 * np.pi / float(hf_row['axial_period_sat'])
    sma = float(hf_row['semimajorax_sat'])
    ecc = float(hf_row['eccentricity_sat'])

    # Setup Initial State and Parameters
    y0 = [
        axial_p,
        axial_s,
        sma,
        ecc,
        0.0,  # cumulative delta-a from planet-raised tide
        0.0,  # cumulative delta-a from satellite-raised tide
        0.0,  # cumulative delta-e from planet-raised tide
        0.0,  # cumulative delta-e from satellite-raised tide
    ]

    params = {
        'M_p': hf_row['M_int'],
        'M_s': hf_row['M_sat'],
        'R_p': hf_row['R_int'],
        'R_s': hf_row['R_sat'],
        'C_p': hf_row['C_int'],
        'C_s': hf_row['C_sat'],
    }

    # Retrieve tidal mode information from tides_o object
    nmk_p = np.asarray(tides_o.get(primary='planet', perturber='satellite').nmk)
    LNk_p = np.asarray(tides_o.get(primary='planet', perturber='satellite').LNk)

    nmk_s = np.asarray(tides_o.get(primary='satellite', perturber='planet').nmk)
    LNk_s = np.asarray(tides_o.get(primary='satellite', perturber='planet').LNk)

    kmin, kmax = int(np.min(nmk_p[:, 2])), int(np.max(nmk_p[:, 2]))
    n_k = kmax - kmin + 1

    def _dense_love(nmk, LNk, m_target):
        # Sparse-mode-safe (real tidal data need not have a row for every
        # integer s in [kmin, kmax]) AND folds in the m=0/s<0 modes that
        # Obliqua's own emission only supplies for s>=0.
        mask = (nmk[:, 1] == m_target) & (nmk[:, 2] >= kmin) & (nmk[:, 2] <= kmax)
        dense = np.zeros(n_k, dtype=complex)
        dense[(nmk[mask, 2] - kmin).astype(int)] = LNk[mask]
        if m_target == 0:
            pos_mask = mask & (nmk[:, 2] > 0)
            s_pos = nmk[pos_mask, 2].astype(int)
            neg_idx = -s_pos - kmin
            valid = (neg_idx >= 0) & (neg_idx < n_k)
            dense[neg_idx[valid]] = np.conj(LNk[pos_mask][valid])
        return dense

    LNk_p_m0 = _dense_love(nmk_p, LNk_p, 0)
    LNk_p_m2 = _dense_love(nmk_p, LNk_p, 2)
    LNk_s_m0 = _dense_love(nmk_s, LNk_s, 0)
    LNk_s_m2 = _dense_love(nmk_s, LNk_s, 2)

    def domega_dt(I_j, C_j, sum_dOmega):
        """Planar secular tidal spin"""
        # Eq 132 from Correia & Valente (2022)
        return -(3.0 * I_j / (2.0 * C_j)) * sum_dOmega

    def smooth_sign(sigma, scale=1e-12):
        """Smooth approximation to sign(sigma) using tanh to avoid solver kinks."""
        return np.tanh(sigma / scale)

    def dE_dt(z, p):
        """Tidal energy dissipation rate"""
        Omega_p, Omega_s, a, e, *_ = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6
        I_s = (const_G * p['M_p'] ** 2 * p['R_s'] ** 5) / a**6

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)
        s_arr = k.astype(float)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X0_sq = X_0**2
        X2_sq = X_2**2

        K_p0 = -LNk_p_m0.imag
        K_p2 = -LNk_p_m2.imag
        K_s0 = -LNk_s_m0.imag
        K_s2 = -LNk_s_m2.imag

        # Eqs 133, 134, 135 from Correia & Valente (2022)
        dE_orb_p = I_p * n_mm * np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)) / 4
        dE_orb_s = I_s * n_mm * np.sum(s_arr * (K_s0 * X0_sq + 3.0 * K_s2 * X2_sq)) / 4

        dE_rot_p = -I_p * 3 * Omega_p * np.sum(K_p2 * X2_sq) / 2
        dE_rot_s = -I_s * 3 * Omega_s * np.sum(K_s2 * X2_sq) / 2

        return -(dE_orb_p + dE_rot_p), -(dE_orb_s + dE_rot_s)

    def orbitals(t, z, p):
        Omega_p, Omega_s, a, e, *_ = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        # Eqs 91 and 84 from Correia & Valente (2022)
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)
        E_p = n_mm * (p['M_s'] / p['M_p']) * (p['R_p'] / a) ** 5
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6
        E_s = n_mm * (p['M_p'] / p['M_s']) * (p['R_s'] / a) ** 5
        I_s = (const_G * p['M_p'] ** 2 * p['R_s'] ** 5) / a**6

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)
        s_arr = k.astype(float)
        sig_scale = max(1e-12, 1e-4 * n_mm)
        sigma_0 = -s_arr * n_mm
        sigma_p2 = 2 * Omega_p - s_arr * n_mm
        sigma_s2 = 2 * Omega_s - s_arr * n_mm

        K_p0 = np.abs(LNk_p_m0.imag) * smooth_sign(sigma_0, sig_scale)
        K_p2 = np.abs(LNk_p_m2.imag) * smooth_sign(sigma_p2, sig_scale)
        K_s0 = np.abs(LNk_s_m0.imag) * smooth_sign(sigma_0, sig_scale)
        K_s2 = np.abs(LNk_s_m2.imag) * smooth_sign(sigma_s2, sig_scale)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X0_sq = X_0**2
        X2_sq = X_2**2
        sqrt_e = np.sqrt(1.0 - e_safe**2)

        # Eqs 129, 131, 132 from Correia & Valente (2022)
        dOmega_p = np.sum(K_p2 * X2_sq)
        dOmega_s = np.sum(K_s2 * X2_sq)
        da_p = np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq))
        da_s = np.sum(s_arr * (K_s0 * X0_sq + 3.0 * K_s2 * X2_sq))
        de_p = np.sum(
            K_p0 * X0_sq * s_arr * sqrt_e - 3.0 * K_p2 * X2_sq * (2.0 - s_arr * sqrt_e)
        )
        de_s = np.sum(
            K_s0 * X0_sq * s_arr * sqrt_e - 3.0 * K_s2 * X2_sq * (2.0 - s_arr * sqrt_e)
        )

        sqrt_term = np.sqrt(1.0 - e_safe**2)
        da_dt_p = a * (E_p / 2.0) * da_p
        da_dt_s = a * (E_s / 2.0) * da_s
        de_dt_p = (E_p * sqrt_term / (4.0 * e_safe)) * de_p
        de_dt_s = (E_s * sqrt_term / (4.0 * e_safe)) * de_s

        return [
            domega_dt(I_p, p['C_p'], dOmega_p),
            domega_dt(I_s, p['C_s'], dOmega_s),
            da_dt_p + da_dt_s,
            de_dt_p + de_dt_s,
            da_dt_p,
            da_dt_s,
            de_dt_p,
            de_dt_s,
        ]

    # Integration
    solver = config.orbit.solver
    log.debug('Integrating the ps1d orbital model with solve_ivp')
    sol = solve_ivp(
        fun=lambda t, y: orbitals(t, y, params),
        t_span=(0, dt),
        y0=y0,
        method=solver.method,
        rtol=solver.rtol,
        atol=solver.atol,
    )

    # Compute total angular momentum at the end of the integration
    L_final = (
        params['C_p'] * sol.y[0][-1]
        + params['C_s'] * sol.y[1][-1]
        + (params['M_p'] * params['M_s'])
        / (params['M_p'] + params['M_s'])
        * np.sqrt(
            const_G * (params['M_p'] + params['M_s']) * sol.y[2][-1] * (1 - sol.y[3][-1] ** 2)
        )
    )

    # Compute total energy dissipated by tides over the time step
    dE_tide_p, _ = dE_dt(y0, params)

    # log energy per surface area for debugging
    energy_per_area = dE_tide_p / (4 * np.pi * params['R_p'] ** 2)
    log.debug(
        f'Total tidal power: {dE_tide_p:.3e} W, Energy per unit area: {energy_per_area:.3e} W/m^2'
    )

    # Exact, solver-consistent split of the changes accumulated over this step
    da_planet_tide = sol.y[4][-1] - sol.y[4][0]
    da_sat_tide = sol.y[5][-1] - sol.y[5][0]
    de_planet_tide = sol.y[6][-1] - sol.y[6][0]
    de_sat_tide = sol.y[7][-1] - sol.y[7][0]

    # Self-consistency check: the two contributions must sum to the total change
    da_total_check = da_planet_tide + da_sat_tide - (sol.y[2][-1] - sol.y[2][0])
    de_total_check = de_planet_tide + de_sat_tide - (sol.y[3][-1] - sol.y[3][0])
    log.debug(f'tidal split residuals: da={da_total_check:.3e} m, de={de_total_check:.3e}')

    # Update semimajor axis and axial period
    hf_row['sma_dot_planet'] = da_planet_tide / dt  # m/s, planet-raised tide
    hf_row['sma_dot_sat'] = da_sat_tide / dt  # m/s, satellite-raised tide
    hf_row['ecc_dot_planet'] = de_planet_tide / dt  # 1/s
    hf_row['ecc_dot_sat'] = de_sat_tide / dt  # 1/s

    hf_row['axial_period'] = 2 * np.pi / sol.y[0][-1]
    hf_row['axial_period_sat'] = 2 * np.pi / sol.y[1][-1]
    hf_row['semimajorax_sat'] = sol.y[2][-1]
    # Circularization (e -> 0) is a valid terminal state, but the ODE can
    # cross exactly zero and land on a floating-point-noise-scale negative
    # value; left unclamped, _state_is_valid's e < 0.0 check rejects that
    # step forever and the step-size controller collapses trying to
    # satisfy an unsatisfiable condition. Clamp rather than loosen the
    # validity check.
    hf_row['eccentricity_sat'] = max(sol.y[3][-1], 0.0)
    hf_row['plan_sat_am'] = L_final


def ps1d_evec(
    hf_row,
    tides_o,
    dt,
    config,
    fine_sink=None,
    fine_stride=1,
    filter_value=None,
    t_abs_start_yr=None,
):
    """Evolve planet-satellite orbit (spins, semi-major axis, eccentricity, evection_angle).

    Combines the Correia & Valente (2022) secular model with Rufu & Canup (2020)
    evection-resonance terms. See "Evection resonance" in docs/Explanations/orbit.md.

    Parameters
    ----------
    hf_row : dict
        Current runtime variables
    tides_o : Tides_t
        Tidal interactions container
    dt : float
        Integration time step [yr]
    config : Config
        Configuration options (reads `config.orbit.solver` for `solve_ivp` parameters)
    fine_sink : list, optional
        List to append solver accepted-step samples for high-resolution inspection
    fine_stride : int, default=1
        Downsampling stride for internal solver steps saved to `fine_sink`
    filter_value : float, optional
        Resonance activation weight (0.0 to 1.0) gating only `dw_evection_oscillating`.
        Secular apsidal precession remains active when 0.0. Default is 1.0
    t_abs_start_yr : float, optional
        Absolute start time [yr] for `fine_sink` timestamps; defaults to `hf_row['Time']`
    """

    # Convert time to seconds
    dt = float(dt) * secs_per_year

    # Orbital parameters from helpfile
    axial_p = 2 * np.pi / float(hf_row['axial_period'])
    axial_s = 2 * np.pi / float(hf_row['axial_period_sat'])
    sma = float(hf_row['semimajorax_sat'])
    ecc = float(hf_row['eccentricity_sat'])
    evection_angle = float(hf_row['evection_angle'])

    # Setup Initial State and Parameters
    y0 = [
        axial_p,
        axial_s,
        sma,
        ecc,
        evection_angle,
        0.0,  # cumulative delta-a from planet-raised tide
        0.0,  # cumulative delta-a from satellite-raised tide
        0.0,  # cumulative delta-e from planet-raised tide
        0.0,  # cumulative delta-e from satellite-raised tide
    ]

    # Mean motion of star-planet system
    n_star = np.sqrt(
        const_G * (hf_row['M_star'] + hf_row['M_int']) / hf_row['semimajorax'] ** 3
    )

    params = {
        'M_p': hf_row['M_int'],
        'M_s': hf_row['M_sat'],
        'R_p': hf_row['R_int'],
        'R_s': hf_row['R_sat'],
        'C_p': hf_row['C_int'],
        'C_s': hf_row['C_sat'],
        'n_star': n_star,
        'J_struc': 0.315,
    }

    # Retrieve tidal mode information from tides_o object
    nmk_p = np.asarray(tides_o.get(primary='planet', perturber='satellite').nmk)
    LNk_p = np.asarray(tides_o.get(primary='planet', perturber='satellite').LNk)

    nmk_s = np.asarray(tides_o.get(primary='satellite', perturber='planet').nmk)
    LNk_s = np.asarray(tides_o.get(primary='satellite', perturber='planet').LNk)

    kmin, kmax = int(np.min(nmk_p[:, 2])), int(np.max(nmk_p[:, 2]))
    n_k = kmax - kmin + 1

    def _dense_love(nmk, LNk, m_target):
        # Sparse-mode-safe scatter (real tidal-mode data doesn't have a row
        # for every integer s in [kmin, kmax]) PLUS an m=0/s<0 mirror fix.
        mask = (nmk[:, 1] == m_target) & (nmk[:, 2] >= kmin) & (nmk[:, 2] <= kmax)
        dense = np.zeros(n_k, dtype=complex)
        dense[(nmk[mask, 2] - kmin).astype(int)] = LNk[mask]
        if m_target == 0:
            pos_mask = mask & (nmk[:, 2] > 0)
            s_pos = nmk[pos_mask, 2].astype(int)
            neg_idx = -s_pos - kmin
            valid = (neg_idx >= 0) & (neg_idx < n_k)
            dense[neg_idx[valid]] = np.conj(LNk[pos_mask][valid])
        return dense

    LNk_p_m0 = _dense_love(nmk_p, LNk_p, 0)
    LNk_p_m2 = _dense_love(nmk_p, LNk_p, 2)
    LNk_s_m0 = _dense_love(nmk_s, LNk_s, 0)
    LNk_s_m2 = _dense_love(nmk_s, LNk_s, 2)

    def domega_dt(I_j, C_j, sum_dOmega):
        """Planar secular tidal spin"""
        # Eq 132 from Correia & Valente (2022)
        return -(3.0 * I_j / (2.0 * C_j)) * sum_dOmega

    def dw_dt(e, e_safe, n_mm, n_star, phi, dw_J2, E_p, E_s, sum_dw_p, sum_dw_s, scale_width):
        """Apsidal precession / Evection Angle"""
        # Eq 11 from Rufu & Canup (2020)
        prefactor = 1.0 / (e_safe**2 * np.sqrt(1.0 - e_safe**2))

        dw_tide_p = E_p * prefactor * sum_dw_p
        dw_tide_s = E_s * prefactor * sum_dw_s

        dw_secular_star = (3.0 / 4.0) * (n_star**2 / n_mm) * np.sqrt(1.0 - e_safe**2)
        dw_secular_total = dw_J2 + dw_tide_p + dw_tide_s + dw_secular_star

        if filter_value is not None:
            local_filter = filter_value
        else:
            local_filter = 1.0

        dw_evection_oscillating = (
            (15.0 / 4.0) * np.sqrt(1.0 - e_safe**2) * (n_star**2 / n_mm) * np.cos(2.0 * phi)
        )
        dphi = dw_secular_total + (local_filter * dw_evection_oscillating) - n_star
        return dphi, local_filter

    def smooth_sign(sigma, scale=1e-12):
        """Smooth approximation to sign(sigma) using tanh to avoid solver kinks."""
        return np.tanh(sigma / scale)

    def smooth_amplitude_near_zero(val_real, sigma, scale):
        """Smoothly blend the amplitude of a real value towards a target (1.5) near zero forcing frequency."""
        zero_weight = np.exp(-((sigma / scale) ** 2))
        return zero_weight * 1.5 + (1.0 - zero_weight) * val_real

    def dE_dt(z, p):
        """Tidal energy dissipation rate"""
        Omega_p, Omega_s, a, e, *_ = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6
        I_s = (const_G * p['M_p'] ** 2 * p['R_s'] ** 5) / a**6

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)
        s_arr = k.astype(float)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X0_sq = X_0**2
        X2_sq = X_2**2

        K_p0 = -LNk_p_m0.imag
        K_p2 = -LNk_p_m2.imag
        K_s0 = -LNk_s_m0.imag
        K_s2 = -LNk_s_m2.imag

        # Eqs 133, 134, 135 from Correia & Valente (2022)
        dE_orb_p = I_p * n_mm * np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)) / 4
        dE_orb_s = I_s * n_mm * np.sum(s_arr * (K_s0 * X0_sq + 3.0 * K_s2 * X2_sq)) / 4

        dE_rot_p = -I_p * 3 * Omega_p * np.sum(K_p2 * X2_sq) / 2
        dE_rot_s = -I_s * 3 * Omega_s * np.sum(K_s2 * X2_sq) / 2

        return -(dE_orb_p + dE_rot_p), -(dE_orb_s + dE_rot_s)

    def orbitals(t, z, p):
        Omega_p, Omega_s, a, e, phi, *_ = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        # Eqs 91 and 84 from Correia & Valente (2022)
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)
        E_p = n_mm * (p['M_s'] / p['M_p']) * (p['R_p'] / a) ** 5
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6
        E_s = n_mm * (p['M_p'] / p['M_s']) * (p['R_s'] / a) ** 5
        I_s = (const_G * p['M_p'] ** 2 * p['R_s'] ** 5) / a**6

        # Below Eq 11 from Rufu & Canup (2020)
        Omega_b = np.sqrt(const_G * p['M_p'] / p['R_p'] ** 3)
        J2 = p['J_struc'] * (Omega_p / Omega_b) ** 2
        dw_J2 = 1.5 * J2 * n_mm * (p['R_p'] / a) ** 2 / (1.0 - e_safe**2) ** 2

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)

        s_arr = k.astype(float)
        sig_scale = max(1e-12, 1e-4 * n_mm)
        sigma_0 = -s_arr * n_mm
        sigma_p2 = 2 * Omega_p - s_arr * n_mm
        sigma_s2 = 2 * Omega_s - s_arr * n_mm

        A_p0 = smooth_amplitude_near_zero(LNk_p_m0.real, sigma_0, sig_scale)
        A_p2 = smooth_amplitude_near_zero(LNk_p_m2.real, sigma_p2, sig_scale)
        A_s0 = smooth_amplitude_near_zero(LNk_s_m0.real, sigma_0, sig_scale)
        A_s2 = smooth_amplitude_near_zero(LNk_s_m2.real, sigma_s2, sig_scale)
        K_p0 = np.abs(LNk_p_m0.imag) * smooth_sign(sigma_0, sig_scale)
        K_p2 = np.abs(LNk_p_m2.imag) * smooth_sign(sigma_p2, sig_scale)
        K_s0 = np.abs(LNk_s_m0.imag) * smooth_sign(sigma_0, sig_scale)
        K_s2 = np.abs(LNk_s_m2.imag) * smooth_sign(sigma_s2, sig_scale)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X_m1 = X_all[-1]
        X_1 = X_all[1]
        X_m2 = X_all[-2]
        X0_sq = X_0**2
        X2_sq = X_2**2
        sqrt_e = np.sqrt(1.0 - e_safe**2)

        # Eqs 129, 131, 132 from Correia & Valente (2022)
        # Eqs 10, 11 from Rufu & Canup (2020)
        dOmega_p = np.sum(K_p2 * X2_sq)
        dOmega_s = np.sum(K_s2 * X2_sq)
        da_p = np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq))
        da_s = np.sum(s_arr * (K_s0 * X0_sq + 3.0 * K_s2 * X2_sq))
        de_p = np.sum(
            K_p0 * X0_sq * s_arr * sqrt_e - 3.0 * K_p2 * X2_sq * (2.0 - s_arr * sqrt_e)
        )
        de_s = np.sum(
            K_s0 * X0_sq * s_arr * sqrt_e - 3.0 * K_s2 * X2_sq * (2.0 - s_arr * sqrt_e)
        )
        term0 = (
            2.0 * e_safe**2 * X0_sq
            + e_safe**2 * X_0 * (X_m2 + X_2)
            + 2.0 * e_safe * X_0 * (X_m1 + X_1)
        )
        term2 = (
            (12.0 * (2.0 - s_arr * sqrt_e**3) - 9.0 * e_safe**2) * X2_sq
            + 3.0 * e_safe**2 * X_2 * X_m2
            + (4.0 * s_arr * sqrt_e**3 - 6.0 * e_safe**2) * X_0 * X_2
            + 6.0 * e_safe * X_2 * (X_m1 + X_1)
        )
        dw_p = np.sum((3.0 / 16.0) * A_p0 * term0 - (1.0 / 16.0) * A_p2 * term2)
        dw_s = np.sum((3.0 / 16.0) * A_s0 * term0 - (1.0 / 16.0) * A_s2 * term2)

        sums = {'dw_p': dw_p, 'dw_s': dw_s}

        dphi_dt, r_filter = dw_dt(
            e,
            e_safe,
            n_mm,
            p['n_star'],
            phi,
            dw_J2,
            E_p,
            E_s,
            sums['dw_p'],
            sums['dw_s'],
            scale_width=1e-8,
        )

        sqrt_term = np.sqrt(1.0 - e_safe**2)
        da_dt_p = a * (E_p / 2.0) * da_p
        da_dt_s = a * (E_s / 2.0) * da_s
        de_dt_p = (E_p * sqrt_term / (4.0 * e_safe)) * de_p
        de_dt_s = (E_s * sqrt_term / (4.0 * e_safe)) * de_s
        de_res = (
            (15.0 / 4.0) * e_safe * sqrt_term * (p['n_star'] ** 2 / n_mm) * np.sin(2.0 * phi)
        )

        return [
            domega_dt(I_p, p['C_p'], dOmega_p),
            domega_dt(I_s, p['C_s'], dOmega_s),
            da_dt_p + da_dt_s,
            de_dt_p + de_dt_s + (r_filter * de_res),
            dphi_dt,
            da_dt_p,
            da_dt_s,
            de_dt_p,
            de_dt_s,
        ]

    # Integration
    solver = config.orbit.solver
    log.debug('Integrating the ps1d_evec orbital model with solve_ivp')
    sol = solve_ivp(
        fun=lambda t, y: orbitals(t, y, params),
        t_span=(0, dt),
        y0=y0,
        method=solver.method,
        rtol=solver.rtol,
        atol=solver.atol,
    )

    y_end = sol.y[:, -1]

    # Compute total angular momentum at the end of the integration
    L_final = (
        params['C_p'] * y_end[0]
        + params['C_s'] * y_end[1]
        + (params['M_p'] * params['M_s'])
        / (params['M_p'] + params['M_s'])
        * np.sqrt(const_G * (params['M_p'] + params['M_s']) * y_end[2] * (1 - y_end[3] ** 2))
    )

    dE_tide_p, _ = dE_dt(y0, params)
    energy_per_area = dE_tide_p / (4 * np.pi * params['R_p'] ** 2)
    log.debug(
        f'Total tidal power: {dE_tide_p:.3e} W, Energy per unit area: {energy_per_area:.3e} W/m^2'
    )

    da_planet_tide = sol.y[5][-1] - sol.y[5][0]
    da_sat_tide = sol.y[6][-1] - sol.y[6][0]
    de_planet_tide = sol.y[7][-1] - sol.y[7][0]
    de_sat_tide = sol.y[8][-1] - sol.y[8][0]

    if fine_sink is not None:
        # Populates fine_sink with the solver-clock samples (band-
        # independent); does not touch disk or apply the storage-clock
        # throttle, which is the caller's job.
        t_start = t_abs_start_yr if t_abs_start_yr is not None else hf_row['Time']
        t_abs_yr = t_start + sol.t / secs_per_year
        omega_p_f = sol.y[0]
        omega_s_f = sol.y[1]
        sma_f = sol.y[2]
        ecc_f = sol.y[3]
        phi_f = sol.y[4]
        cum_da_p_f = sol.y[5]
        cum_da_s_f = sol.y[6]
        cum_de_p_f = sol.y[7]
        cum_de_s_f = sol.y[8]

        # Band-independent solver-clock decimation (storage/plotting
        # thinning only, not a physics decision.
        if fine_stride is not None and fine_stride > 1:
            keep = np.zeros(len(t_abs_yr), dtype=bool)
            keep[::fine_stride] = True
            keep[-1] = True
        else:
            keep = np.ones(len(t_abs_yr), dtype=bool)

        t_abs_yr_k = t_abs_yr[keep]
        omega_p_k = omega_p_f[keep]
        omega_s_k = omega_s_f[keep]
        sma_k = sma_f[keep]
        ecc_k = ecc_f[keep]
        phi_k = phi_f[keep]
        cum_da_p_k = cum_da_p_f[keep]
        cum_da_s_k = cum_da_s_f[keep]
        cum_de_p_k = cum_de_p_f[keep]
        cum_de_s_k = cum_de_s_f[keep]
        filter_k = np.full(len(t_abs_yr_k), filter_value)

        fine_sink.append(
            {
                't_abs_yr': t_abs_yr_k,
                'omega_p': omega_p_k,
                'omega_s': omega_s_k,
                'sma': sma_k,
                'ecc': ecc_k,
                'phi': phi_k,
                'da_planet_tide_cum': cum_da_p_k,
                'da_sat_tide_cum': cum_da_s_k,
                'de_planet_tide_cum': cum_de_p_k,
                'de_sat_tide_cum': cum_de_s_k,
                'filter': filter_k,
            }
        )

    hf_row['sma_dot_planet'] = da_planet_tide / dt
    hf_row['sma_dot_sat'] = da_sat_tide / dt
    hf_row['ecc_dot_planet'] = de_planet_tide / dt
    hf_row['ecc_dot_sat'] = de_sat_tide / dt

    hf_row['axial_period'] = 2 * np.pi / y_end[0]
    hf_row['axial_period_sat'] = 2 * np.pi / y_end[1]
    hf_row['semimajorax_sat'] = y_end[2]
    # Circularization (e -> 0) is a valid terminal state, but the ODE can
    # cross exactly zero and land on a floating-point-noise-scale negative
    # value; left unclamped, _state_is_valid's e < 0.0 check rejects that
    # step forever and evolve_orbit_satellite's step-size controller
    # collapses trying to satisfy an unsatisfiable condition. Clamp rather
    # than loosen the validity check.
    hf_row['eccentricity_sat'] = max(y_end[3], 0.0)
    hf_row['evection_angle'] = y_end[4]
    hf_row['plan_sat_am'] = L_final
