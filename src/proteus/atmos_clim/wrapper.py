# Generic atmosphere wrapper
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import pi, unique

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.atmos_clim.common import (
    Atmos_t,
    LevelsSource,
    clip_radius_to_hill,
    get_spfile_path,
)
from proteus.utils.constants import const_R, gas_list
from proteus.utils.helper import UpdateStatusfile, safe_rm

log = logging.getLogger('fwl.' + __name__)

# Scalar properties of the photospheric and XUV levels, all read off the
# atmospheric structure the solver returns. They are carried together so that
# each level keeps a radius, pressure, temperature and gravity that describe the
# same structure. Fluxes and surface state are deliberately absent: the coupling
# and the deadlock detector both need to see those move when a solve fails.
LEVEL_KEYS = ('R_obs', 'p_obs', 'T_obs', 'g_obs', 'R_xuv', 'p_xuv', 'T_xuv', 'g_xuv')

# Phrase used for each level source in the substitution warning.
_SOURCE_PHRASE = {
    LevelsSource.CONVERGED_SOLVE: 'last converged solve',
    LevelsSource.COMMITTED_ROW: 'last committed row',
}

# Composition at the XUV level, read off the same structure and used by BOREAS
# to set the mean molecular weight of the outflow. Carried with the levels, so
# a held radius is never combined with the composition of a rejected structure.
XUV_VMR_KEYS = tuple(f'{gas}_vmr_xuv' for gas in gas_list)

# Number of consecutive iterations on carried levels after which the run says so
# at error level. Long enough that a solver which stumbles for an iteration or
# two passes unremarked, short enough to appear while a run is still worth
# steering.
CARRIED_LEVELS_ALERT = 10


def _finite_levels(source: dict, keys) -> dict:
    """Read the level properties out of `source`, skipping the unusable ones.

    A key is taken only when it is present and its value is finite. A solver
    that returns NaN for a level has not measured it, so storing that value
    would replace the fallback with something that cannot be fallen back on.
    """
    out = {}
    for key in keys:
        if key not in source:
            continue
        val = float(source[key])
        if np.isfinite(val):
            out[key] = val
    return out


def _alert_on_long_streak(atmos_o: Atmos_t):
    """Report a long run of iterations whose levels the run did not resolve.

    A streak this long is no longer a solver that stumbled once. The interior
    keeps evolving while the levels stand still, so escape is computed on a
    planet that has moved on, and the deadlock detector cannot see it: that
    detector fires only when the interior has stopped moving as well.
    """
    if atmos_o.levels_stale_iters >= CARRIED_LEVELS_ALERT:
        log.error(
            'Atmosphere levels have been unresolved for %d consecutive iterations; '
            'escape is running on a structure this run has not converged',
            atmos_o.levels_stale_iters,
        )


def carry_converged_levels(atmos_o: Atmos_t, hf_row: dict, previous_row: dict | None = None):
    """Carry the photospheric and XUV levels across a solve that was rejected.

    A solver that rejects its solution still returns an atmospheric structure,
    and the levels read off that structure can sit far outside the planet. The
    energy-limited escape rate goes as the cube of the XUV radius, so a rejected
    structure otherwise becomes a large mass-loss rate that looks like a
    physical result. Each converged solve records its levels; a rejected one has
    its levels replaced by that record, and the substitution is logged.

    Modules without a nonlinear solve (JANUS, dummy, and AGNI's transparent and
    prescribed-temperature branches) always report convergence, so this records
    their levels and never substitutes.

    The record of converged levels lives only in memory, so a resumed run starts
    without one. If the first solve after the resume is rejected, the fallback is
    taken from the last helpfile row instead: that row was written before this
    run began, and it is the state escape used on the resumed iteration anyway.
    The warning states which of the two sources was used, because a helpfile row
    is not guaranteed to come from a converged solve.

    The helpfile fallback applies to the first solve only. If every solve since
    then was rejected, the record is empty, but each helpfile row written since
    the resume contains rejected levels too, so there is nothing better to read
    back and the rejected levels are kept.

    Parameters
    ----------
        atmos_o : Atmos_t
            Atmosphere struct. Carries the convergence flag of the solve that
            has just run, and the record of converged levels, which this
            function updates.
        hf_row : dict
            Simulation variables for this iteration, modified in place. Holds
            the levels the atmosphere module has just written.
        previous_row : dict, optional
            The last committed row, used only when this run's first solve is
            rejected.
    """

    keys = LEVEL_KEYS + XUV_VMR_KEYS
    atmos_o.solves_seen += 1

    # Solve was accepted: its levels become the ones to fall back on. Merged
    # rather than replaced, so a module that reports one level intermittently
    # does not empty the record of that level.
    if atmos_o.converged:
        atmos_o.levels_converged.update(_finite_levels(hf_row, keys))
        atmos_o.levels_source = LevelsSource.CONVERGED_SOLVE
        atmos_o.levels_stale_iters = 0
        return

    # First solve of this run, and it failed. The last committed row predates
    # the run, which is the situation a resumed run is in, so fall back on that.
    if atmos_o.solves_seen == 1 and previous_row:
        atmos_o.levels_converged.update(_finite_levels(previous_row, keys))
        atmos_o.levels_source = LevelsSource.COMMITTED_ROW

    # Counts every iteration whose levels this run did not resolve itself, so a
    # run with nothing to fall back on is counted too. That case is the worse
    # of the two, since escape then runs on the rejected structure directly.
    atmos_o.levels_stale_iters += 1

    if not atmos_o.levels_converged:
        log.warning(
            'Atmosphere solve did not converge and no earlier levels are '
            'available; using the levels of the rejected structure '
            '(%d consecutive iterations)',
            atmos_o.levels_stale_iters,
        )
        _alert_on_long_streak(atmos_o)
        return

    log.warning(
        'Atmosphere solve did not converge; holding levels from the %s '
        '(%d consecutive iterations)',
        _SOURCE_PHRASE[atmos_o.levels_source],
        atmos_o.levels_stale_iters,
    )
    _alert_on_long_streak(atmos_o)

    for key in LEVEL_KEYS:
        if key in atmos_o.levels_converged and key in hf_row:
            log.debug(
                '    %-5s %.5e  ->  %.5e'
                % (key, float(hf_row[key]), atmos_o.levels_converged[key])
            )
    for key in keys:
        if key in atmos_o.levels_converged and key in hf_row:
            hf_row[key] = atmos_o.levels_converged[key]


def realign_xuv_gravity(hf_row: dict, radius: float):
    """Place the XUV gravity at a radius that has been moved on its own.

    The level properties describe one height and are carried as a group, so
    moving the radius by itself leaves a pressure, temperature and gravity
    belonging to a different one. Gravity is the only one recoverable from the
    radius alone, by the inverse-square form the dummy and JANUS modules use to
    place it, so it is the only one brought along. AGNI reads gravity off the
    profile it solved, which is not held once the solve is behind us, so the
    radius-only form is the closest available estimate on every module.

    Pressure, temperature and the composition in ``XUV_VMR_KEYS`` keep the
    values the last converged solve read, at the height it read them from. A
    reading from a nearby height is worth more than none: ``p_xuv`` is what
    AGNI and JANUS derive ``R_xuv`` from when ``xuv_defined_by_radius`` is off,
    BOREAS builds the outflow mean molecular weight from the composition, and
    every helpfile column is required to stay finite.

    The gravity is left alone rather than invalidated where the planet is
    unknown, for the same reason: a stale gravity is readable and a missing one
    is not.

    Parameters
    ----------
        hf_row : dict
            Simulation variables for this iteration, modified in place.
        radius : float
            The radius the level now sits at [m].
    """
    if 'g_xuv' not in hf_row:
        return

    gravity = float(hf_row.get('gravity', np.nan))
    r_int = float(hf_row.get('R_int', np.nan))
    placeable = (
        np.isfinite(radius)
        and radius > 0.0
        and np.isfinite(gravity)
        and np.isfinite(r_int)
        and r_int > 0.0
    )
    if placeable:
        hf_row['g_xuv'] = gravity * (r_int / radius) ** 2


def run_atmosphere(
    atmos_o: Atmos_t,
    config: Config,
    dirs: dict,
    loop_counter: dict,
    wl: list,
    fl: list,
    update_stellar_spectrum: bool,
    hf_all: pd.DataFrame,
    hf_row: dict,
    write_data: bool = True,
):
    """Run Atmosphere submodule.

    Generic function to run an atmospheric simulation with either JANUS, AGNI or dummy.
    Writes into the hf_row generic variable passed as an arguement.

    Parameters
    ----------
        atmos_o: Atmos_t
            Atmosphere struct
        config : Config
            Configuration options and other variables
        dirs : dict
            Dictionary containing paths to directories
        loop_counter : dict
            Dictionary containing iteration information
        wl : list
            Wavelength [nm]
        fl : list
            Flux at 1 AU [erg s-1 cm-2 nm-1]
        update_stellar_spectrum : bool
            Spectral file path
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        write_data : bool, optional
            Whether to write data to files, by default True

    """

    log.info('Solving atmosphere...')

    # Update bond albedo
    if config.atmos_clim.albedo_pl > 1.0e-9:
        # Warn if invalid
        if config.atmos_clim.rayleigh:
            log.warning(
                'Physically inconsistent options selected: '
                '`albedo_pl > 0` and `rayleigh = True`'
            )
        if config.atmos_clim.cloud_enabled:
            log.warning(
                'Physically inconsistent options selected: '
                '`albedo_pl > 0` and `cloud_enabled = True`'
            )

        # Update value of input albedo
        hf_row['albedo_pl'] = float(config.atmos_clim.albedo_pl)
        log.debug(f'Set albedo by config: {hf_row["albedo_pl"] * 100:.3f}%')

    else:
        # Held at zero
        hf_row['albedo_pl'] = 0.0

    # Handle new surface temperature
    if config.interior_energetics.module == 'boundary':
        # T_surf is already advanced by the Boundary backend's ODE, so the
        # atmosphere wrapper must not overwrite it.
        pass

    elif config.atmos_clim.surf_state == 'mixed_layer':
        # Argument order: (current hf_row, previous committed row). The
        # function integrates forward from hf_pre['Time'] to hf_cur['Time']
        # starting at hf_pre['T_surf'].
        hf_row['T_surf'] = ShallowMixedOceanLayer(hf_row, hf_all.iloc[-1].to_dict())

    elif config.atmos_clim.surf_state == 'fixed':
        hf_row['T_surf'] = hf_row['T_magma']

    # elif surf_state=='skin':
    #    Don't do anything here, because this will be handled by the atmosphere model.

    if config.atmos_clim.module == 'janus':
        # Import
        from proteus.atmos_clim.janus import InitAtm, InitStellarSpectrum, RunJANUS

        # Run JANUS
        no_atm = bool(atmos_o._atm is None)
        if no_atm and not config.params.resume:
            hf_row['T_surf'] = hf_row['T_magma']

        # Init atm object if first iteration or change in stellar spectrum
        if no_atm or update_stellar_spectrum:
            spectral_file_nostar = get_spfile_path(dirs['fwl'], config)
            if not os.path.exists(spectral_file_nostar):
                UpdateStatusfile(dirs, 20)
                raise FileNotFoundError(
                    "Spectral file does not exist at '%s'" % spectral_file_nostar
                )

            wl = np.array(wl)
            fl = np.array(fl)
            idx = unique(wl, return_index=True)[1]
            wl_un = wl[idx]
            fl_un = fl[idx]

            InitStellarSpectrum(dirs, wl_un, fl_un, spectral_file_nostar)
            atmos_o._atm = InitAtm(dirs, config)

        atmos_o._atm_janus_last, atm_output = RunJANUS(
            atmos_o._atm, dirs, config, hf_row, hf_all, write_data=write_data
        )

    elif config.atmos_clim.module == 'agni':
        # Import
        from proteus.atmos_clim.agni import (
            activate_julia,
            deallocate_atmos,
            init_agni_atmos,
            run_agni,
            update_agni_atmos,
        )

        # Run AGNI
        # Initialise atmosphere struct
        spfile_path = os.path.join(dirs['output'], 'runtime.sf')
        no_atm = bool(atmos_o._atm is None)
        if no_atm or update_stellar_spectrum:
            log.debug('Initialise new atmosphere struct')

            # first run?
            if no_atm:
                activate_julia(dirs, config.atmos_clim.agni.verbosity)
                if not config.params.resume:
                    hf_row['T_surf'] = hf_row['T_magma']
            else:
                # Remove old spectral file if it exists
                safe_rm(spfile_path)
                safe_rm(spfile_path + '_k')

                # deallocate old atmosphere
                deallocate_atmos(atmos_o._atm)

            # allocate new
            atmos_o._atm = init_agni_atmos(dirs, config, hf_row)

            # Check allocation was ok
            if not bool(atmos_o._atm.is_alloc):
                UpdateStatusfile(dirs, 22)
                raise RuntimeError('Atmosphere struct not allocated')

        # Update profile
        atmos_o._atm = update_agni_atmos(atmos_o._atm, hf_row, dirs, config)

        # Run solver
        atmos_o._atm, atm_output = run_agni(
            atmos_o._atm,
            loop_counter['total'],
            dirs,
            config,
            hf_row,
            write_data=write_data,
        )

    elif config.atmos_clim.module == 'dummy':
        # Import
        from proteus.atmos_clim.dummy import RunDummyAtm

        # Run dummy atmosphere model
        atm_output = RunDummyAtm(dirs, config, hf_row)

    # Capture the atmosphere convergence flag onto the transient struct
    # (not persisted to helpfile). AGNI sets this from its Newton solver;
    # JANUS / dummy / transparent always succeed and default to True.
    atmos_o.converged = bool(atm_output.pop('agni_converged', True))

    # Store variables common to `hf_row` and `atm_output`
    for key in atm_output.keys():
        if key in hf_row.keys():
            hf_row[key] = atm_output[key]

    # Keep escape and the observables off a structure the solver rejected. This
    # runs on the merged row, after the module output and the module's direct
    # writes to `hf_row` are both in place, and before the quantities derived
    # from the levels below.
    previous_row = None
    if hf_all is not None and len(hf_all) > 0:
        previous_row = hf_all.iloc[-1].to_dict()
    carry_converged_levels(atmos_o, hf_row, previous_row=previous_row)

    # A carried radius can come from a row written before the clip existed, or
    # under a different Hill radius, so bound it here as well. The rate escape
    # computes goes as the cube of this radius, so the bound wins; the gravity
    # is then placed at the radius the bound left behind. The finiteness test is
    # on the RESULT rather than on the carried value, because a radius of NaN
    # compares unequal to everything: testing it there skips the placement in
    # the one case where the bound does supply a real radius.
    if hasattr(config, 'escape') and 'R_xuv' in hf_row:
        carried = float(hf_row['R_xuv'])
        clipped = clip_radius_to_hill(config, hf_row, carried)
        hf_row['R_xuv'] = clipped
        if np.isfinite(clipped) and clipped != carried:
            realign_xuv_gravity(hf_row, clipped)

    # Persist the solve outcome, so a row whose levels were carried can be
    # identified from the output alone rather than from the log.
    hf_row['atm_converged'] = 1.0 if atmos_o.converged else -1.0
    hf_row['atm_levels_stale'] = float(atmos_o.levels_stale_iters)

    # Copy special cases
    hf_row['rho_obs'] = 3 * hf_row['M_planet'] / (4 * pi * hf_row['R_obs'] ** 3)
    hf_row['F_net'] = hf_row['F_int'] - hf_row['F_atm']
    hf_row['bond_albedo'] = atm_output['albedo']

    # Calculate bolometric observables (measured at infinite distance)
    update_bolometry(hf_row)

    # Estimate WTG parameter
    update_wtg_surf(hf_row)


def write_atmosphere_snapshot(atmos_o: Atmos_t, config: Config, dirs: dict, hf_row: dict):
    """Write the current atmosphere state to a NetCDF snapshot.

    Uses the corresponding module's writer, as appropriate.

    Arguments
    ---------
        atmos_o : Atmos_t
            Atmosphere struct
        config : Config
            Configuration options for PROTEUS
        dirs : dict
            Dictionary containing paths to directories
        hf_row : dict
            Dictionary containing simulation variables for current iteration
    """

    # Get time from this iteration
    time = float(hf_row['Time'])

    # Use AGNI writer
    if config.atmos_clim.module == 'agni':
        from proteus.atmos_clim.agni import write_atmos_ncdf

        if atmos_o._atm is None:
            log.warning('Cannot write atmosphere; AGNI struct unallocated')
            return

        write_atmos_ncdf(atmos_o._atm, dirs, time)

    # Use JANUS writer
    elif config.atmos_clim.module == 'janus':
        from proteus.atmos_clim.janus import write_atmos_ncdf

        # The solved column, not `_atm`: JANUS copies `_atm` before it
        # integrates and resamples the copy, so `_atm` carries no profile of
        # its own, only the surface boundary condition and the fluxes written
        # back onto it. Its arrays are sized for the integration grid while
        # those fluxes are on the radiative one, so it cannot be written as a
        # single consistent snapshot.
        if atmos_o._atm_janus_last is None:
            log.warning('Cannot write atmosphere; JANUS has not solved a column yet')
            return

        write_atmos_ncdf(atmos_o._atm_janus_last, dirs, time)

    # Otherwise, write no atmosphere NetCDF


def update_wtg_surf(hf_row: dict):
    """
    Update WTG parameter.

    https://royalsocietypublishing.org/doi/full/10.1098/rspa.2016.0107
    """

    omega = 2 * pi / hf_row['axial_period']  # Angular rotation rate
    R_mix = const_R / hf_row['atm_kg_per_mol']  # Specific gas constant
    hf_row['wtg_surf'] = (R_mix * hf_row['T_surf']) ** 0.5 / (omega * hf_row['R_int'])


def update_bolometry(hf_row: dict):
    """
    Update bolometric observables (transit depth, contrast ratio.)

    https://link.springer.com/content/pdf/10.1007/978-3-319-30648-3_40-1.pdf
    """

    # Transit depth
    hf_row['transit_depth'] = (hf_row['R_obs'] / hf_row['R_star']) ** 2.0

    # Eclipse depth
    #    Accounting for fact that F_ins is scaled to TOA, not to stellar surface.
    #    Also, F_ins can be zero when bol_scale is being appled.
    if hf_row['F_ins'] == 0.0:
        hf_row['eclipse_depth'] = 0.0
    else:
        hf_row['eclipse_depth'] = ((hf_row['F_olr'] + hf_row['F_sct']) / hf_row['F_ins']) * (
            hf_row['R_obs'] / hf_row['separation']
        ) ** 2.0


def ShallowMixedOceanLayer(hf_cur: dict, hf_pre: dict):
    # This scheme is not typically used, but it maintained here from legacy code
    from scipy.integrate import solve_ivp

    log.info('>>>>>>>>>> Flux convergence scheme <<<<<<<<<<<')

    # For SI conversion
    yr = 3.154e7  # s

    # Last T_surf and time from atmosphere, K
    t_cur = hf_cur['Time'] * yr
    t_pre = hf_pre['Time'] * yr
    Ts_pre = hf_pre['T_surf']

    # Properties of the shallow mixed ocean layer
    c_p_layer = 1000  # J kg-1 K-1
    rho_layer = 3000  # kg m-3
    depth_layer = 1000  # m

    def ocean_evolution(t, y):
        # Specific heat of mixed ocean layer
        mu = c_p_layer * rho_layer * depth_layer  # J K-1 m-2
        # RHS of ODE
        RHS = -hf_cur['F_net'] / mu
        return RHS

    # Solve ODE
    sol_curr = solve_ivp(ocean_evolution, [t_pre, t_cur], [Ts_pre])

    # New current surface temperature from shallow mixed layer
    Ts_cur = sol_curr.y[0][-1]  # K

    return Ts_cur
