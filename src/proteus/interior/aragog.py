# Aragog interior module
from __future__ import annotations  # noqa: I001

import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4 as nc
import numpy as np
import pandas as pd
import platformdirs

from aragog import Output, Solver, aragog_file_logger
from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _Radionuclide,
    _ScalingsParameters,
    _SolverParameters,
)
from proteus.interior.common import Interior_t
from proteus.interior.timestep import next_step
from proteus.utils.constants import radnuc_data

logger = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus.config import Config


FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))


class AragogRunner:
    def __init__(
        self,
        config: Config,
        dirs: dict,
        hf_row: dict,
        hf_all: pd.DataFrame,
        interior_o: Interior_t,
    ):
        AragogRunner.setup_logger(config, dirs)
        dt = AragogRunner.compute_time_step(config, dirs, hf_row, hf_all, interior_o)
        self.setup_or_update_solver(config, hf_row, interior_o, dt, dirs)
        self.aragog_solver = interior_o.aragog_solver

    @staticmethod
    def setup_logger(config: Config, dirs: dict):
        file_level = logging.getLevelName(config.interior.aragog.logging)
        aragog_file_logger(
            console_level=logging.WARNING, file_level=file_level, log_dir=dirs['output']
        )

    @staticmethod
    def compute_time_step(
        config: Config, dirs: dict, hf_row: dict, hf_all: pd.DataFrame, interior_o: Interior_t
    ) -> float:
        if interior_o.ic == 1:
            return 0.0
        else:
            step_sf = 1.0  # dt scale factor
            return next_step(config, dirs, hf_row, hf_all, step_sf)

    @staticmethod
    def setup_or_update_solver(
        config: Config, hf_row: dict, interior_o: Interior_t, dt: float, dirs: dict
    ):
        if interior_o.aragog_solver is None:
            AragogRunner.setup_solver(config, hf_row, interior_o, dirs['output'])
            if config.params.resume:
                AragogRunner.update_solver(dt, hf_row, interior_o, output_dir=dirs['output'])
            interior_o.aragog_solver.initialize()
            AragogRunner._verify_entropy_ic(config, interior_o, dirs['output'])
        else:
            if interior_o.ic == 1:
                AragogRunner.update_structure(config, hf_row, interior_o)
                # Preserve the evolved T field across equilibration resets.
                # Without this, reset() re-runs get_adiabat() from the config
                # IC, losing the entropy-table correction from iteration 0.
                Tfield = interior_o.aragog_solver.temperature_staggered[:, -1]
                interior_o.aragog_solver.parameters.initial_condition.initial_condition = 2
                interior_o.aragog_solver.parameters.initial_condition.init_temperature = Tfield
            else:
                # Refresh mesh with current planet radius and gravity,
                # matching SPIDER which re-reads R_int and gravity at each call.
                AragogRunner.update_structure(config, hf_row, interior_o)
                AragogRunner.update_solver(dt, hf_row, interior_o)
            interior_o.aragog_solver.reset()

    @staticmethod
    def setup_solver(config: Config, hf_row: dict, interior_o: Interior_t, outdir: str):
        # Non-dimensionalization removed: all quantities in SI, time in years.
        # ScalingsParameters is vestigial (all scales = 1.0).
        scalings = _ScalingsParameters()

        solver = _SolverParameters(
            start_time=0,
            end_time=0,
            # After the non-dim removal, T is in K (O(1000-5000)).
            # atol must be in K: 1.0 K is ~0.03% relative at 3000 K.
            # rtol from config (typically 1e-6 to 1e-8) is fine as-is.
            atol=max(config.interior.aragog.tolerance, 1.0),
            rtol=config.interior.aragog.tolerance,
            tsurf_poststep_change=config.interior.aragog.tsurf_poststep_change,
            event_triggering=config.interior.aragog.event_triggering,
        )

        boundary_conditions = _BoundaryConditionsParameters(
            # 4 = prescribed heat flux
            outer_boundary_condition=4,
            # first guess surface heat flux [W/m2]
            outer_boundary_value=hf_row['F_atm'],
            # 1 = core cooling model
            # 2 = prescribed heat flux
            # 3 = prescribed temperature
            inner_boundary_condition=(config.interior.aragog.inner_boundary_condition),
            # core temperature [K], if inner_boundary_condition = 3
            inner_boundary_value=(config.interior.aragog.inner_boundary_value),
            # only used in gray body BC, outer_boundary_condition = 1
            emissivity=1,
            # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature=hf_row['T_eqm'],
            # used if inner_boundary_condition = 1
            core_heat_capacity=config.struct.core_heatcap,
            # core T_avg/T_cmb ratio from adiabatic gradient (Bower+2018 Table 2)
            tfac_core_avg=1.147,
            # ultra-thin boundary layer parameterization (Bower et al. 2018, Eq. 18)
            param_utbl=config.interior.aragog.param_utbl,
            param_utbl_const=config.interior.aragog.param_utbl_const,
        )

        # Define the inner_radius for the mesh
        if config.struct.module == 'self':
            inner_radius = config.struct.corefrac * hf_row['R_int']  # core radius [m]
        elif config.struct.module == 'zalmoxis':
            # Read core radius from hf_row (already computed by Zalmoxis in
            # determine_interior_radius_with_zalmoxis, no need to re-run solver)
            inner_radius = hf_row.get('R_core', config.struct.corefrac * hf_row['R_int'])
        else:
            raise ValueError("Invalid module configuration. Expected 'self' or 'zalmoxis'.")

        mesh = _MeshParameters(
            # planet radius [m]
            outer_radius=hf_row['R_int'],
            # core radius [m]
            inner_radius=inner_radius,
            # basic nodes
            number_of_nodes=config.interior.aragog.num_levels,
            mixing_length_profile='constant',
            core_density=config.struct.core_density,
            eos_method=1,  # 1: Adams-Williamson / 2: User defined
            surface_density=4090,  # AdamsWilliamsonEOS parameter [kg/m3]
            gravitational_acceleration=hf_row['gravity'],  # [m/s-2]
            adiabatic_bulk_modulus=config.interior.aragog.bulk_modulus,  # AW-EOS parameter [Pa]
            mass_coordinates=config.interior.aragog.mass_coordinates,
            surface_pressure=0.0,  # TODO: wire to atmospheric overburden when available
        )

        # Update the mesh if the module is 'zalmoxis'
        if config.struct.module == 'zalmoxis':
            mesh.eos_method = 2  # User-defined EOS based on Zalmoxis
            mesh.eos_file = os.path.join(
                outdir, 'data', 'zalmoxis_output.dat'
            )  # Zalmoxis output file with mantle parameters

        energy = _EnergyParameters(
            conduction=config.interior.aragog.conduction,
            convection=config.interior.aragog.convection,
            gravitational_separation=(config.interior.aragog.gravitational_separation),
            mixing=config.interior.aragog.mixing,
            dilatation=config.interior.aragog.dilatation,
            radionuclides=config.interior.radiogenic_heat,
            tidal=config.interior.tidal_heat,
            tidal_array=interior_o.tides,
        )

        # Define initial conditions for prescribing temperature profile
        if config.struct.module == 'self':
            initial_condition_temperature_profile = config.interior.aragog.initial_condition
            init_file_temperature_profile = os.path.join(
                FWL_DATA_DIR, f'interior_lookup_tables/{config.interior.aragog.init_file}'
            )
        elif config.struct.module == 'zalmoxis':
            _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
            if config.struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES):
                # When using Zalmoxis with temperature-dependent silicate EOS, set initial condition to user-defined temperature field (from file) in Aragog
                initial_condition_temperature_profile = 2
                init_file_temperature_profile = os.path.join(
                    outdir, 'data', 'zalmoxis_output_temp.txt'
                )
            elif (
                config.struct.zalmoxis.mantle_eos.startswith('PALEOS:')
                and config.interior.aragog.initial_condition == 3
            ):
                # For PALEOS EOS with adiabatic IC: Aragog uses IC=3 with
                # entropy tables for its entropy-conserving adiabat. After
                # initialization, _verify_entropy_ic compares against an
                # independent PALEOS entropy inversion and corrects the IC
                # if the discrepancy exceeds 1% (table resolution effect).
                initial_condition_temperature_profile = 3
                init_file_temperature_profile = ''
            else:
                # Otherwise, use the initial condition from aragog config
                initial_condition_temperature_profile = config.interior.aragog.initial_condition
                init_file_temperature_profile = os.path.join(
                    FWL_DATA_DIR, f'interior_lookup_tables/{config.interior.aragog.init_file}'
                )
        else:
            raise ValueError("Invalid module configuration. Expected 'self' or 'zalmoxis'.")

        # When initial_thermal_state = 'self_consistent', Zalmoxis computes
        # T_surface from accretion + differentiation energy (White+Li 2025)
        # and passes it via hf_row. This overrides the config ini_tmagma.
        ini_tmagma = config.interior.aragog.ini_tmagma
        T_surface_computed = hf_row.get('T_surface_initial', 0)
        if T_surface_computed and T_surface_computed > 0:
            logger.info(
                'Overriding ini_tmagma with self-consistent thermal state: '
                '%.0f K -> %.0f K', ini_tmagma, T_surface_computed,
            )
            ini_tmagma = T_surface_computed

        initial_condition = _InitialConditionParameters(
            # 1 = linear profile
            # 2 = user-defined profile
            # 3 = adiabatic profile
            initial_condition=initial_condition_temperature_profile,
            # initial top temperature (K)
            surface_temperature=ini_tmagma,
            basal_temperature=config.interior.aragog.basal_temperature,
            init_file=init_file_temperature_profile,
        )

        # EOS lookup directory for phase properties (Cp, alpha, density, entropy).
        # For PALEOS EOS: generate P-T tables from PALEOS data.
        # Prefer PALEOS-2phase (separate solid/liquid) over unified table:
        # 2-phase tables give clean phase-specific entropy values at
        # solidus/liquidus, enabling correct Delta_S for mixing flux and IC.
        # The unified table has interpolation artifacts across the melting
        # curve discontinuity.
        if (
            config.struct.module == 'zalmoxis'
            and config.struct.zalmoxis.mantle_eos.startswith('PALEOS:')
        ):
            from proteus.interior.zalmoxis import load_zalmoxis_material_dictionaries

            mat_dicts = load_zalmoxis_material_dictionaries()

            # Get unified table path (needed for melting curves and fallback)
            eos_entry = mat_dicts.get(config.struct.zalmoxis.mantle_eos, {})
            paleos_eos_file = eos_entry.get('eos_file', '')

            mass_tot = config.struct.mass_tot or 1.0
            P_max = min(200e9, 50e9 * mass_tot + 100e9)
            LOOK_UP_DIR = Path(outdir) / 'data' / 'aragog_pt'

            # Try PALEOS-2phase first (separate solid/liquid tables)
            twophase_entry = mat_dicts.get('PALEOS-2phase:MgSiO3', {})
            solid_eos = twophase_entry.get('solid_mantle', {}).get('eos_file', '')
            liquid_eos = twophase_entry.get('melted_mantle', {}).get('eos_file', '')
            has_2phase = (
                solid_eos and os.path.isfile(solid_eos)
                and liquid_eos and os.path.isfile(liquid_eos)
            )

            if has_2phase:
                if not (LOOK_UP_DIR / 'density_melt.dat').is_file():
                    from zalmoxis.eos_export import generate_aragog_pt_tables_2phase

                    logger.info(
                        'Generating phase-specific Aragog P-T tables from PALEOS-2phase'
                    )
                    generate_aragog_pt_tables_2phase(
                        solid_eos_file=solid_eos,
                        liquid_eos_file=liquid_eos,
                        P_range=(1e5, P_max),
                        n_P=200,
                        n_T=200,
                        output_dir=LOOK_UP_DIR,
                    )
                else:
                    logger.info('PALEOS-2phase tables already exist, skipping generation')
            elif not has_2phase:
                # Fall back to unified table (identical solid/melt files)
                from zalmoxis.eos_export import generate_aragog_pt_tables

                if paleos_eos_file and os.path.isfile(paleos_eos_file):
                    from proteus.interior.zalmoxis import (
                        load_zalmoxis_solidus_liquidus_functions,
                    )

                    melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                        config.struct.zalmoxis.mantle_eos, config
                    )
                    if melt_funcs is not None:
                        sol_func, liq_func = melt_funcs
                    else:
                        from zalmoxis.melting_curves import (
                            get_solidus_liquidus_functions,
                        )

                        sol_func, liq_func = get_solidus_liquidus_functions(
                            'Stixrude14-solidus', 'PALEOS-liquidus'
                        )

                    if not (LOOK_UP_DIR / 'density_melt.dat').is_file():
                        logger.warning(
                            'PALEOS-2phase tables not found, falling back to '
                            'unified table (entropy near melting curve may be '
                            'unreliable)'
                        )
                        generate_aragog_pt_tables(
                            eos_file=paleos_eos_file,
                            solidus_func=sol_func,
                            liquidus_func=liq_func,
                            P_range=(1e5, P_max),
                            n_P=200,
                            n_T=200,
                            output_dir=LOOK_UP_DIR,
                        )
            else:
                logger.warning(
                    'PALEOS EOS file not found (%s), falling back to legacy tables',
                    paleos_eos_file,
                )
                LOOK_UP_DIR = (
                    FWL_DATA_DIR
                    / 'interior_lookup_tables'
                    / '1TPa-dK09-elec-free'
                    / 'MgSiO3_Wolf_Bower_2018_1TPa'
                )
        else:
            LOOK_UP_DIR = (
                FWL_DATA_DIR
                / 'interior_lookup_tables'
                / 'EOS'
                / 'dynamic'
                / config.interior.eos_dir
                / 'P-T'
            )
            if not (LOOK_UP_DIR / 'heat_capacity_melt.dat').is_file():
                LOOK_UP_DIR = (
                    FWL_DATA_DIR
                    / 'interior_lookup_tables'
                    / '1TPa-dK09-elec-free'
                    / 'MgSiO3_Wolf_Bower_2018_1TPa'
                )
        # Determine melting curves. When using PALEOS EOS via Zalmoxis,
        # generate PALEOS-derived melting curves so Aragog uses the SAME
        # solidus/liquidus as SPIDER (PALEOS-liquidus * mushy_zone_factor).
        # Without this, Aragog uses Monteux-600 which differs by ~600 K,
        # making melt fractions incomparable.
        if (
            config.struct.module == 'zalmoxis'
            and config.struct.zalmoxis.mantle_eos.startswith('PALEOS:')
        ):
            paleos_melt_dir = Path(outdir) / 'data' / 'paleos_melting'
            paleos_melt_dir.mkdir(parents=True, exist_ok=True)
            sol_file = paleos_melt_dir / 'solidus_P-T.dat'
            liq_file = paleos_melt_dir / 'liquidus_P-T.dat'
            if not sol_file.is_file():
                from proteus.interior.zalmoxis import (
                    _make_derived_solidus,
                    load_zalmoxis_solidus_liquidus_functions,
                )

                melt_fns = load_zalmoxis_solidus_liquidus_functions(
                    config.struct.zalmoxis.mantle_eos, config
                )
                if melt_fns is not None:
                    s_fn, l_fn = melt_fns
                else:
                    from zalmoxis.melting_curves import (
                        get_solidus_liquidus_functions as _gslf,
                    )

                    _, l_fn = _gslf('Stixrude14-solidus', 'PALEOS-liquidus')
                    s_fn = _make_derived_solidus(
                        l_fn, config.struct.zalmoxis.mushy_zone_factor
                    )

                P_arr = np.logspace(8, 12, 500)
                sol_data = np.column_stack([P_arr, [s_fn(P) for P in P_arr]])
                liq_data = np.column_stack([P_arr, [l_fn(P) for P in P_arr]])
                np.savetxt(str(sol_file), sol_data, header='pressure temperature', comments='#')
                np.savetxt(str(liq_file), liq_data, header='pressure temperature', comments='#')
                logger.info('Generated PALEOS melting curves for Aragog: %s', paleos_melt_dir)

            solidus_path = sol_file
            liquidus_path = liq_file
        else:
            MELTING_DIR = FWL_DATA_DIR / 'interior_lookup_tables/Melting_curves/'
            solidus_path = MELTING_DIR / config.interior.melting_dir / 'solidus_P-T.dat'
            liquidus_path = MELTING_DIR / config.interior.melting_dir / 'liquidus_P-T.dat'

        # check data exist
        if not (LOOK_UP_DIR / 'heat_capacity_melt.dat').is_file():
            raise FileNotFoundError(f'Aragog lookup data not found at {LOOK_UP_DIR}')

        # Entropy tables (optional): enable entropy-conserving adiabatic IC.
        # Generated by Zalmoxis from the same PALEOS data as Cp/alpha/rho.
        entropy_melt = LOOK_UP_DIR / 'entropy_melt.dat'
        entropy_solid = LOOK_UP_DIR / 'entropy_solid.dat'
        entropy_melt_arg = str(entropy_melt) if entropy_melt.is_file() else ""
        entropy_solid_arg = str(entropy_solid) if entropy_solid.is_file() else ""
        if entropy_melt_arg:
            logger.info('Entropy tables found: enabling entropy-conserving adiabat IC')
        else:
            logger.info('No entropy tables: Aragog will use single-phase dTdPs adiabat')

        phase_liquid = _PhaseParameters(
            density=LOOK_UP_DIR / 'density_melt.dat',
            viscosity=1e2,
            heat_capacity=LOOK_UP_DIR / 'heat_capacity_melt.dat',
            melt_fraction=1,
            thermal_conductivity=4,
            thermal_expansivity=LOOK_UP_DIR / 'thermal_exp_melt.dat',
            entropy=entropy_melt_arg,
        )

        phase_solid = _PhaseParameters(
            density=LOOK_UP_DIR / 'density_solid.dat',
            viscosity=1e21,
            heat_capacity=LOOK_UP_DIR / 'heat_capacity_solid.dat',
            melt_fraction=0,
            thermal_conductivity=4,
            thermal_expansivity=LOOK_UP_DIR / 'thermal_exp_solid.dat',
            entropy=entropy_solid_arg,
        )

        phase_mixed = _PhaseMixedParameters(
            latent_heat_of_fusion=4e6,
            rheological_transition_melt_fraction=config.interior.rheo_phi_loc,
            rheological_transition_width=config.interior.rheo_phi_wid,
            solidus=solidus_path,
            liquidus=liquidus_path,
            phase='mixed',
            phase_transition_width=0.1,
            grain_size=config.interior.grain_size,
        )

        radionuclides = []
        if config.interior.radiogenic_heat:
            # offset by age_ini, which converts model simulation time to the
            # actual age
            radio_t0 = config.delivery.radio_tref - config.star.age_ini
            radio_t0 *= 1e9  # Convert Gyr to yr

            def _append_radnuc(_iso, _cnc):
                radionuclides.append(
                    _Radionuclide(
                        name=_iso,
                        t0_years=radio_t0,
                        abundance=radnuc_data[_iso]['abundance'],
                        concentration=_cnc,
                        heat_production=radnuc_data[_iso]['heatprod'],
                        half_life_years=radnuc_data[_iso]['halflife'],
                    )
                )

            if config.delivery.radio_K > 0.0:
                _append_radnuc('k40', config.delivery.radio_K)

            if config.delivery.radio_Th > 0.0:
                _append_radnuc('th232', config.delivery.radio_Th)

            if config.delivery.radio_U > 0.0:
                _append_radnuc('u235', config.delivery.radio_U)
                _append_radnuc('u238', config.delivery.radio_U)

        param = Parameters(
            boundary_conditions=boundary_conditions,
            energy=energy,
            initial_condition=initial_condition,
            mesh=mesh,
            phase_solid=phase_solid,
            phase_liquid=phase_liquid,
            phase_mixed=phase_mixed,
            radionuclides=radionuclides,
            scalings=scalings,
            solver=solver,
        )

        interior_o.aragog_solver = Solver(param)

    @staticmethod
    def _verify_entropy_ic(config: Config, interior_o: Interior_t, outdir: str):
        """Verify Aragog's IC against an independent PALEOS entropy inversion.

        When using PALEOS EOS with adiabatic IC, compute T(P) from PALEOS
        entropy inversion and compare against Aragog's initialized T profile.
        Logs a warning if the two disagree by more than 1%.

        Parameters
        ----------
        config : Config
            PROTEUS configuration.
        interior_o : Interior_t
            Interior object with initialized Aragog solver.
        outdir : str
            Output directory for diagnostic files.
        """
        if not (
            config.struct.module == 'zalmoxis'
            and config.struct.zalmoxis.mantle_eos.startswith('PALEOS:')
            and config.interior.aragog.initial_condition == 3
        ):
            return

        try:
            from zalmoxis.eos_export import compute_entropy_adiabat

            from proteus.interior.zalmoxis import (
                load_zalmoxis_material_dictionaries,
                load_zalmoxis_solidus_liquidus_functions,
            )

            mat_dicts = load_zalmoxis_material_dictionaries()
            eos_entry = mat_dicts.get(config.struct.zalmoxis.mantle_eos, {})
            paleos_eos_file = eos_entry.get('eos_file', '')
            if not paleos_eos_file or not os.path.isfile(paleos_eos_file):
                return

            melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                config.struct.zalmoxis.mantle_eos, config
            )
            sol_func = liq_func = None
            if melt_funcs is not None:
                sol_func, liq_func = melt_funcs

            # Check for 2-phase tables
            twophase = mat_dicts.get('PALEOS-2phase:MgSiO3', {})
            solid_eos = twophase.get('solid_mantle', {}).get('eos_file', '')
            liquid_eos = twophase.get('melted_mantle', {}).get('eos_file', '')
            solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
            liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None

            # Compute PROTEUS-side entropy-inverted profile
            T_surf = config.interior.aragog.ini_tmagma
            solver = interior_o.aragog_solver
            # Get Aragog's basic node pressures and temperatures
            P_basic = solver.evaluator.mesh.basic_pressure[:, -1]
            T_stag_aragog = solver.evaluator.initial_condition.temperature

            # Use 1 bar for surface pressure (Aragog mesh surface_pressure=0
            # causes log10(0)=NaN in entropy inversion)
            P_surf = max(float(P_basic[-1]), 1e5)
            P_cmb = float(P_basic[0])

            result = compute_entropy_adiabat(
                eos_file=paleos_eos_file,
                T_surface=T_surf,
                P_surface=P_surf,
                P_cmb=P_cmb,
                n_points=500,
                solidus_func=sol_func,
                liquidus_func=liq_func,
                solid_eos_file=solid_eos,
                liquid_eos_file=liquid_eos,
            )

            # Interpolate PROTEUS profile onto Aragog's staggered mesh
            from scipy.interpolate import interp1d
            P_stag = solver.evaluator.mesh.staggered_pressure[:, -1]

            T_proteus_interp = interp1d(
                result['P'], result['T'], bounds_error=False, fill_value='extrapolate'
            )(P_stag)

            # Compare (with length check)
            if len(T_stag_aragog) != len(T_proteus_interp):
                logger.warning(
                    'Entropy IC verification skipped: IC temperature array length (%d) '
                    'does not match staggered pressure array length (%d)',
                    len(T_stag_aragog), len(T_proteus_interp),
                )
                return
            T_diff = np.abs(T_stag_aragog - T_proteus_interp)
            T_rel_diff = T_diff / T_stag_aragog * 100
            max_diff = np.max(T_diff)
            max_rel = np.max(T_rel_diff)
            mean_rel = np.mean(T_rel_diff)

            logger.info(
                'Entropy IC verification: Aragog vs PROTEUS T(P) '
                'max_diff=%.1f K (%.2f%%), mean_rel=%.4f%%',
                max_diff, max_rel, mean_rel,
            )

            if max_rel > 1.0:
                logger.warning(
                    'Entropy IC mismatch > 1%%: Aragog entropy tables too '
                    'coarse. max_diff=%.1f K (%.2f%%). Overriding Aragog IC '
                    'with high-resolution PALEOS entropy inversion.',
                    max_diff, max_rel,
                )
                # Override Aragog's IC with the PROTEUS-computed profile.
                solver.evaluator.initial_condition._temperature = T_proteus_interp

            # Save diagnostic data
            diag_dir = Path(outdir) / 'data' / 'entropy_ic_verification'
            diag_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                diag_dir / 'entropy_ic_comparison.npz',
                P_staggered=P_stag,
                T_aragog=T_stag_aragog,
                T_proteus=T_proteus_interp,
                T_diff=T_diff,
                S_target=result['S_target'],
            )

        except Exception as e:
            logger.warning('Entropy IC verification failed: %s', e)

    @staticmethod
    def update_solver(dt: float, hf_row: dict, interior_o: Interior_t, output_dir: str = None):
        # Set solver time
        # hf_row["Time"] is in years, matching Aragog's time coordinate
        interior_o.aragog_solver.parameters.solver.start_time = hf_row['Time']
        interior_o.aragog_solver.parameters.solver.end_time = hf_row['Time'] + dt

        # Get temperature field from previous run
        if output_dir is not None:  # read it from output directory
            Tfield = read_last_Tfield(output_dir, hf_row['Time'])
        else:  # get it from solver
            Tfield = interior_o.aragog_solver.temperature_staggered[:, -1]

        # Update initial condition: switch to user-defined init
        (interior_o.aragog_solver.parameters.initial_condition.initial_condition) = 2
        (interior_o.aragog_solver.parameters.initial_condition.init_temperature) = Tfield

        # Update boundary conditions
        (interior_o.aragog_solver.parameters.boundary_conditions.outer_boundary_value) = hf_row[
            'F_atm'
        ]

        # Update tidal heating within the mantle
        interior_o.aragog_solver.parameters.energy.tidal_array = interior_o.tides

    @staticmethod
    def update_structure(config: Config, hf_row: dict, interior_o: Interior_t):
        """Refresh the Aragog mesh with current planet structure.

        Called at every coupling timestep to keep the mesh consistent
        with the evolving planet radius and gravity, matching SPIDER's
        behavior of re-reading R_int and gravity at each call.

        For 'self' mode: updates outer/inner radius and gravity directly.
        For 'zalmoxis' mode: outer radius and gravity are updated from
        hf_row; inner radius and EOS profile come from Zalmoxis output
        which is refreshed by the structure solver before the interior step.
        """
        solver = interior_o.aragog_solver
        solver.parameters.mesh.outer_radius = hf_row['R_int']
        solver.parameters.mesh.gravitational_acceleration = hf_row['gravity']

        if config.struct.module == 'self':
            solver.parameters.mesh.inner_radius = (
                config.struct.corefrac * hf_row['R_int']
            )
        # For Zalmoxis: inner_radius is set at setup_solver from Zalmoxis output.
        # The EOS file (zalmoxis_output.dat) is refreshed by the Zalmoxis solver
        # before the interior step, so Aragog.reset() will re-read it.

    def run_solver(self, hf_row, interior_o, dirs):
        # Run Aragog solver
        self.aragog_solver.solve()
        # Get Aragog output
        output = self.get_output(hf_row, interior_o)
        sim_time = self.aragog_solver.parameters.solver.end_time

        # Write output to a file
        self.write_output(dirs['output'], sim_time)

        return sim_time, output

    def write_output(self, output_dir: str, time: float):
        aragog_output: Output = Output(self.aragog_solver)

        fpath = os.path.join(output_dir, 'data', '%d_int.nc' % time)
        aragog_output.write_at_time(fpath, -1)

    def get_output(self, hf_row: dict, interior_o: Interior_t):
        aragog_output: Output = Output(self.aragog_solver)
        aragog_output.state.update(aragog_output.solution.y, aragog_output.solution.t)

        # Debug: check what solve_ivp actually returned
        sol = aragog_output.solution
        T_stag_last = sol.y[:, -1]
        T_stag_first = sol.y[:, 0]
        logger.info(
            'Aragog solve_ivp: t=[%.2e, %.2e], n_t=%d, '
            'T_stag_surface: first=%.2f last=%.2f, '
            'T_basic_surface=%.2f, status=%d',
            sol.t[0], sol.t[-1], len(sol.t),
            T_stag_last[-1], T_stag_first[-1],
            aragog_output.temperature_K_basic[-1, -1],
            sol.status,
        )

        # F_int = F_atm by energy conservation: the prescribed surface BC
        # determines the actual energy loss rate. The interior transport flux
        # at near-surface nodes can differ from F_atm (thermal boundary layer),
        # but the energy leaving the planet is F_atm.
        output = {
            'M_mantle': aragog_output.mantle_mass,
            'T_magma': aragog_output.solution_top_temperature,
            'Phi_global': aragog_output.melt_fraction_global,
            'RF_depth': aragog_output.rheological_front,
            'F_int': hf_row['F_atm'],
        }

        # Per-layer melt fraction and mass for proper liquid/solid mass (fixes #273)
        phi_s = np.array(aragog_output.melt_fraction_staggered[:, -1])  # per layer
        mass_s_arr = aragog_output.mass_staggered[:, -1]  # [kg]

        output['M_mantle_liquid'] = float(np.sum(phi_s * mass_s_arr))
        output['M_mantle_solid'] = float(output['M_mantle'] - output['M_mantle_liquid'])

        # Volumetric melt fraction: computed per-layer from densities.
        # porosity = (rho_s - rho_bulk) / (rho_s - rho_l), volume-weighted.
        P_stag = aragog_output.evaluator.mesh.staggered_pressure[:, -1]
        mixed_eval = aragog_output.evaluator.phases.mixed
        mixed_eval.set_pressure(P_stag)
        # Liquid and solid densities at the liquidus/solidus [kg/m3]
        rho_liq = np.array(mixed_eval._liquid.density()).flatten()
        rho_sol = np.array(mixed_eval._solid.density()).flatten()
        # Bulk density from volume additivity: 1/rho = phi/rho_l + (1-phi)/rho_s
        rho_bulk = 1.0 / (
            phi_s.flatten() / np.where(rho_liq > 0, rho_liq, 1.0)
            + (1 - phi_s.flatten()) / np.where(rho_sol > 0, rho_sol, 1.0)
        )
        # Volume fraction of melt per node: porosity = (rho_s - rho_bulk) / (rho_s - rho_l)
        drho = rho_sol - rho_liq
        safe_drho = np.where(np.abs(drho) > 1e-10, drho, 1.0)
        porosity = np.clip((rho_sol - rho_bulk) / safe_drho, 0, 1)
        # Volume-weighted average porosity (= true melt volume fraction)
        vol = aragog_output.evaluator.mesh.basic.volume.flatten()
        output['Phi_global_vol'] = float(
            np.sum(porosity * vol) / np.sum(vol)
        )
        output['T_pot'] = float(output['T_magma'])

        # Core (CMB) temperature: bottom-most basic node
        output['T_core'] = float(aragog_output.temperature_K_basic[0, -1])

        # Calculate surface area
        radii = aragog_output.radii_km_basic * 1e3  # [m]
        area = 4 * np.pi * float(radii[-1].item()) ** 2  # [m^2]

        # Mass at each mesh layer
        mass_s = aragog_output.mass_staggered[:, -1]  # [kg]

        # Radiogenic heating
        Hradio_s = aragog_output.heating_radio[:, -1]  # [W kg-1]
        output['F_radio'] = float(np.dot(Hradio_s, mass_s)) / area  # [W m-2]

        # Tidal heating flux
        Htidal_s = aragog_output.heating_tidal[:, -1]  # [W kg-1]
        output['F_tidal'] = float(np.dot(Htidal_s, mass_s)) / area

        # Store arrays
        # FIX ME - Should extract values from staggered nodes rather than cropping
        # basic nodes.
        interior_o.phi = np.array(aragog_output.melt_fraction_staggered[:, -1])
        interior_o.visc = np.power(10.0, aragog_output.log10_viscosity_staggered[:, -1])
        interior_o.density = np.array(aragog_output.density_basic[:, -1])[1:]
        interior_o.radius = radii[:]  # length N+1
        interior_o.mass = mass_s[:]
        interior_o.temp = np.array(aragog_output.temperature_K_staggered[:, -1])
        interior_o.pres = np.array(aragog_output.pressure_GPa_staggered[:, -1] * 1e9)

        return output


def read_last_Tfield(output_dir: str, time: float):
    # Read Aragog output at last run
    fpath = os.path.join(output_dir, 'data', '%d_int.nc' % time)
    out = read_ncdf(fpath)

    # Get temperature field at basic nodes and interpolate to staggered nodes
    T_basic = out['temp_b']
    N_basic = len(T_basic)
    T_staggered = (T_basic[0 : N_basic - 1] + T_basic[1:N_basic]) / 2.0

    return T_staggered


def get_all_output_times(output_dir: str):
    files = glob.glob(output_dir + '/data/*_int.nc')
    years = [int(f.split('/')[-1].split('_int')[0]) for f in files]
    mask = np.argsort(years)

    return [years[i] for i in mask]


def read_ncdf(fpath: str):
    out = {}
    ds = nc.Dataset(fpath)

    for key in ds.variables.keys():
        out[key] = ds.variables[key][:]

    ds.close()
    return out


def read_ncdfs(output_dir: str, times: list):
    return [read_ncdf(os.path.join(output_dir, 'data', '%d_int.nc' % t)) for t in times]
