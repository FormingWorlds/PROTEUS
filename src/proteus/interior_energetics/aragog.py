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

from aragog import aragog_file_logger
from aragog.eos.entropy import EntropyEOS
from aragog.solver import EntropySolver, SolverOutput
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
from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.timestep import next_step
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
        # Store P-S EOS directory path for entropy solver initialization
        interior_o._spider_eos_dir = dirs.get('spider_eos_dir', '')
        dt = AragogRunner.compute_time_step(config, dirs, hf_row, hf_all, interior_o)
        self.setup_or_update_solver(config, hf_row, interior_o, dt, dirs)
        self.aragog_solver = interior_o.aragog_solver
        self._config = config
        self._use_jax = config.interior_energetics.aragog.jax

        # Build JAX components if needed (cached on interior_o across steps)
        if self._use_jax:
            from proteus.interior_energetics.aragog_jax import AragogJAXRunner

            self._jax_runner = AragogJAXRunner(config, dirs, hf_row, hf_all, interior_o)

    @staticmethod
    def setup_logger(config: Config, dirs: dict):
        file_level = logging.getLevelName(config.params.out.logging)
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
            # Set entropy IC from Zalmoxis T(r) profile via PALEOS inversion
            AragogRunner._set_entropy_ic(config, interior_o, dirs['output'])
        else:
            if interior_o.ic == 1:
                AragogRunner.update_structure(config, hf_row, interior_o)
                # Preserve the evolved S field across equilibration resets.
                sol = interior_o.aragog_solver.solution
                if sol is not None and sol.y.size > 0:
                    S_last = sol.y[:, -1]
                    interior_o._last_entropy = S_last
            else:
                AragogRunner.update_structure(config, hf_row, interior_o)
                AragogRunner.update_solver(dt, hf_row, interior_o)
            interior_o.aragog_solver.reset()
            # Restore entropy IC from previous solve
            if hasattr(interior_o, '_last_entropy') and interior_o._last_entropy is not None:
                interior_o.aragog_solver.set_initial_entropy(interior_o._last_entropy)

    @staticmethod
    def setup_solver(config: Config, hf_row: dict, interior_o: Interior_t, outdir: str):
        # Non-dimensionalization removed: all quantities in SI, time in years.
        # ScalingsParameters is vestigial (all scales = 1.0).
        scalings = _ScalingsParameters()

        solver = _SolverParameters(
            start_time=0,
            end_time=0,
            # T is in K (O(1000-5000)). For a convecting magma ocean near
            # adiabatic equilibrium, dT/dt ~ 0.1-1 K/yr. atol must be small
            # enough that the BDF solver resolves this: atol = 0.01 K ensures
            # ~0.01 K precision, well below the ~0.3 K/yr cooling rate.
            # Too large (e.g. 1.0 K) makes the solver skip the evolution entirely.
            atol=max(config.interior_energetics.num_tolerance, 0.01),
            rtol=config.interior_energetics.num_tolerance,
            tsurf_poststep_change=config.interior_energetics.aragog.tsurf_poststep_change,
            event_triggering=config.interior_energetics.aragog.event_triggering,
        )

        boundary_conditions = _BoundaryConditionsParameters(
            # 4 = prescribed heat flux (PROTEUS coupling mode)
            # 1 = native grey-body (standalone mode, sigma*T^4)
            outer_boundary_condition=config.interior_energetics.aragog.outer_boundary_condition,
            # first guess surface heat flux [W/m2] (only used if outer_bc=4)
            outer_boundary_value=hf_row['F_atm'],
            # 1 = core cooling model
            # 2 = prescribed heat flux
            # 3 = prescribed temperature
            inner_boundary_condition=(config.interior_energetics.aragog.inner_boundary_condition),
            # core temperature [K], if inner_boundary_condition = 3
            inner_boundary_value=(config.interior_energetics.aragog.inner_boundary_value),
            # only used in gray body BC, outer_boundary_condition = 1
            emissivity=1,
            # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature=hf_row['T_eqm'],
            # used if inner_boundary_condition = 1
            core_heat_capacity=config.interior_struct.core_heatcap,
            # core T_avg/T_cmb ratio from adiabatic gradient (Bower+2018 Table 2)
            tfac_core_avg=1.147,
            # ultra-thin boundary layer parameterization (Bower et al. 2018, Eq. 18)
            param_utbl=config.interior_energetics.aragog.param_utbl,
            param_utbl_const=config.interior_energetics.aragog.param_utbl_const,
        )

        # Define the inner_radius for the mesh
        if config.interior_struct.module == 'self':
            inner_radius = config.interior_struct.corefrac * hf_row['R_int']  # core radius [m]
        elif config.interior_struct.module == 'zalmoxis':
            # Read core radius from hf_row (already computed by Zalmoxis in
            # determine_interior_radius_with_zalmoxis, no need to re-run solver)
            inner_radius = hf_row.get('R_core', config.interior_struct.corefrac * hf_row['R_int'])
        else:
            raise ValueError("Invalid module configuration. Expected 'self' or 'zalmoxis'.")

        mesh = _MeshParameters(
            # planet radius [m]
            outer_radius=hf_row['R_int'],
            # core radius [m]
            inner_radius=inner_radius,
            # basic nodes
            number_of_nodes=config.interior_energetics.num_levels,
            mixing_length_profile='constant',
            core_density=config.interior_struct.core_density,
            eos_method=1,  # 1: Adams-Williamson / 2: User defined
            surface_density=4090,  # AdamsWilliamsonEOS parameter [kg/m3]
            gravitational_acceleration=hf_row['gravity'],  # [m/s-2]
            adiabatic_bulk_modulus=config.interior_energetics.aragog.bulk_modulus,  # AW-EOS parameter [Pa]
            mass_coordinates=config.interior_energetics.aragog.mass_coordinates,
            surface_pressure=0.0,  # TODO: wire to atmospheric overburden when available
        )

        # Update the mesh if the module is 'zalmoxis'
        if config.interior_struct.module == 'zalmoxis':
            mesh.eos_method = 2  # User-defined EOS based on Zalmoxis
            mesh.eos_file = os.path.join(
                outdir, 'data', 'zalmoxis_output.dat'
            )  # Zalmoxis output file with mantle parameters

        energy = _EnergyParameters(
            conduction=config.interior_energetics.trans_conduction,
            convection=config.interior_energetics.trans_convection,
            gravitational_separation=(config.interior_energetics.trans_grav_sep),
            mixing=config.interior_energetics.trans_mixing,
            dilatation=config.interior_energetics.aragog.dilatation,
            radionuclides=config.interior_energetics.heat_radiogenic,
            tidal=config.interior_energetics.heat_tidal,
            tidal_array=interior_o.tides,
            kappah_floor=config.interior_energetics.kappah_floor,
        )

        # Define initial conditions for prescribing temperature profile
        if config.interior_struct.module == 'self':
            initial_condition_temperature_profile = config.interior_energetics.aragog.initial_condition
            init_file_temperature_profile = os.path.join(
                FWL_DATA_DIR, f'interior_lookup_tables/{config.interior_energetics.aragog.init_file}'
            )
        elif config.interior_struct.module == 'zalmoxis':
            _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
            if config.interior_struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES):
                # When using Zalmoxis with temperature-dependent silicate EOS, set initial condition to user-defined temperature field (from file) in Aragog
                initial_condition_temperature_profile = 2
                init_file_temperature_profile = os.path.join(
                    outdir, 'data', 'zalmoxis_output_temp.txt'
                )
            elif (
                config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:')
                and config.interior_energetics.aragog.initial_condition == 3
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
                initial_condition_temperature_profile = config.interior_energetics.aragog.initial_condition
                init_file_temperature_profile = os.path.join(
                    FWL_DATA_DIR, f'interior_lookup_tables/{config.interior_energetics.aragog.init_file}'
                )
        else:
            raise ValueError("Invalid module configuration. Expected 'self' or 'zalmoxis'.")

        # When initial_thermal_state = 'self_consistent', Zalmoxis computes
        # T_surface from accretion + differentiation energy (White+Li 2025)
        # and passes it via hf_row. This overrides the config tsurf_init.
        tsurf_init = config.interior_energetics.tsurf_init
        T_surface_computed = hf_row.get('T_surface_initial', 0)
        if T_surface_computed and T_surface_computed > 0:
            logger.info(
                'Overriding tsurf_init with self-consistent thermal state: '
                '%.0f K -> %.0f K', tsurf_init, T_surface_computed,
            )
            tsurf_init = T_surface_computed

        initial_condition = _InitialConditionParameters(
            # 1 = linear profile
            # 2 = user-defined profile
            # 3 = adiabatic profile
            initial_condition=initial_condition_temperature_profile,
            # initial top temperature (K)
            surface_temperature=tsurf_init,
            basal_temperature=config.interior_energetics.aragog.basal_temperature,
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
            config.interior_struct.module == 'zalmoxis'
            and config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:')
        ):
            from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries

            mat_dicts = load_zalmoxis_material_dictionaries()

            # Get unified table path (needed for melting curves and fallback)
            eos_entry = mat_dicts.get(config.interior_struct.zalmoxis.mantle_eos, {})
            paleos_eos_file = eos_entry.get('eos_file', '')

            mass_tot = config.planet.planet_mass_tot or 1.0
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
                    from proteus.interior_struct.zalmoxis import (
                        load_zalmoxis_solidus_liquidus_functions,
                    )

                    melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                        config.interior_struct.zalmoxis.mantle_eos, config
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
                / config.interior_struct.eos_dir
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
            config.interior_struct.module == 'zalmoxis'
            and config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:')
        ):
            paleos_melt_dir = Path(outdir) / 'data' / 'paleos_melting'
            paleos_melt_dir.mkdir(parents=True, exist_ok=True)
            sol_file = paleos_melt_dir / 'solidus_P-T.dat'
            liq_file = paleos_melt_dir / 'liquidus_P-T.dat'
            if not sol_file.is_file():
                from proteus.interior_struct.zalmoxis import (
                    _make_derived_solidus,
                    load_zalmoxis_solidus_liquidus_functions,
                )

                melt_fns = load_zalmoxis_solidus_liquidus_functions(
                    config.interior_struct.zalmoxis.mantle_eos, config
                )
                if melt_fns is not None:
                    s_fn, l_fn = melt_fns
                else:
                    from zalmoxis.melting_curves import (
                        get_solidus_liquidus_functions as _gslf,
                    )

                    _, l_fn = _gslf('Stixrude14-solidus', 'PALEOS-liquidus')
                    s_fn = _make_derived_solidus(
                        l_fn, config.interior_struct.zalmoxis.mushy_zone_factor
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
            solidus_path = MELTING_DIR / config.interior_struct.melting_dir / 'solidus_P-T.dat'
            liquidus_path = MELTING_DIR / config.interior_struct.melting_dir / 'liquidus_P-T.dat'

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
            rheological_transition_melt_fraction=config.interior_energetics.rfront_loc,
            rheological_transition_width=config.interior_energetics.rfront_wid,
            solidus=solidus_path,
            liquidus=liquidus_path,
            phase='mixed',
            phase_transition_width=0.1,
            grain_size=config.interior_energetics.grain_size,
        )

        radionuclides = []
        if config.interior_energetics.heat_radiogenic:
            # offset by age_ini, which converts model simulation time to the
            # actual age
            radio_t0 = config.interior_energetics.radio_tref - config.star.age_ini
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

            if config.interior_energetics.radio_K > 0.0:
                _append_radnuc('k40', config.interior_energetics.radio_K)

            if config.interior_energetics.radio_Th > 0.0:
                _append_radnuc('th232', config.interior_energetics.radio_Th)

            if config.interior_energetics.radio_U > 0.0:
                _append_radnuc('u235', config.interior_energetics.radio_U)
                _append_radnuc('u238', config.interior_energetics.radio_U)

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

        # Load PALEOS P-S EOS tables for entropy solver.
        # The P-S tables are generated by Zalmoxis (generate_spider_tables)
        # and stored at dirs['spider_eos_dir'].
        spider_eos_dir = interior_o._spider_eos_dir
        if spider_eos_dir and os.path.isdir(spider_eos_dir):
            entropy_eos = EntropyEOS(Path(spider_eos_dir))
        else:
            # Fallback: look for P-S tables in the output directory
            fallback_dir = Path(outdir) / 'data' / 'spider_eos'
            if fallback_dir.is_dir():
                entropy_eos = EntropyEOS(fallback_dir)
            else:
                raise FileNotFoundError(
                    f'PALEOS P-S tables not found. Aragog entropy solver '
                    f'requires P-S tables. Checked: {spider_eos_dir}, {fallback_dir}'
                )
        interior_o.aragog_solver = EntropySolver(param, entropy_eos)

    @staticmethod
    def _set_entropy_ic(config: Config, interior_o: Interior_t, outdir: str):
        """Set the entropy IC from the Zalmoxis T(r) profile via PALEOS.

        Converts the temperature initial condition to entropy using the
        PALEOS P-S EOS tables. The temperature profile comes from either:
        - Zalmoxis adiabatic profile (IC=3)
        - Zalmoxis output file (IC=2, T-dependent EOS)
        - Config tsurf_init with linear/adiabatic profile

        Parameters
        ----------
        config : Config
            PROTEUS configuration.
        interior_o : Interior_t
            Interior object with initialized EntropySolver.
        outdir : str
            Output directory.
        """
        solver = interior_o.aragog_solver
        P_stag = solver._P_stag_flat

        # Compute entropy from T(P) via PALEOS lookup.
        # For a Zalmoxis-derived adiabatic IC: T decreases outward,
        # and S should be roughly constant (isentropic).
        try:
            from zalmoxis.eos_export import compute_entropy_adiabat

            from proteus.interior_struct.zalmoxis import (
                load_zalmoxis_material_dictionaries,
                load_zalmoxis_solidus_liquidus_functions,
            )

            mat_dicts = load_zalmoxis_material_dictionaries()
            eos_entry = mat_dicts.get(config.interior_struct.zalmoxis.mantle_eos, {})
            paleos_eos_file = eos_entry.get('eos_file', '')

            melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                config.interior_struct.zalmoxis.mantle_eos, config
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

            tsurf_init = config.interior_energetics.tsurf_init
            T_surface_computed = interior_o.__dict__.get('T_surface_initial', 0)
            if T_surface_computed and T_surface_computed > 0:
                tsurf_init = T_surface_computed

            # Use 1 bar as surface pressure for the entropy lookup.
            # The mesh surface P can be negative or zero; the adiabat
            # computation needs a physically meaningful surface P.
            P_surf = 1e5  # 1 bar
            P_cmb = max(float(P_stag[0]), 1e9)

            # Compute target entropy: S_target = S(P_surf, tsurf_init)
            # Then set this S uniformly across the mantle (isentrope).
            # This ensures the surface T will match tsurf_init when
            # the EOS is evaluated at (P_surf, S_target).
            result = compute_entropy_adiabat(
                eos_file=paleos_eos_file,
                T_surface=tsurf_init,
                P_surface=P_surf,
                P_cmb=P_cmb,
                n_points=500,
                solidus_func=sol_func,
                liquidus_func=liq_func,
                solid_eos_file=solid_eos,
                liquid_eos_file=liquid_eos,
            )

            # The isentropic profile: S_target is constant by construction.
            # Use the target entropy (not the profile, which may have
            # numerical noise from the root-finding).
            S_target = result['S_target']
            N = len(P_stag)
            S_init = np.full(N, S_target)

            # Verify: T at surface should match tsurf_init
            T_check = solver.entropy_eos.temperature(
                np.array([P_surf]), np.array([S_target])
            )
            logger.info(
                'Entropy IC: tsurf_init=%.0f K -> S_target=%.1f J/kg/K '
                '(verification: T(P_surf, S_target)=%.0f K)',
                tsurf_init, S_target, float(T_check),
            )

            solver.set_initial_entropy(S_init)

        except Exception as e:
            # Fallback: uniform entropy from SPIDER default
            logger.warning(
                'Could not compute entropy IC from PALEOS (%s). '
                'Using uniform S=3200 J/kg/K.', e,
            )
            N = len(P_stag)
            solver.set_initial_entropy(np.full(N, 3200.0))

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
            config.interior_struct.module == 'zalmoxis'
            and config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:')
            and config.interior_energetics.aragog.initial_condition == 3
        ):
            return

        try:
            from zalmoxis.eos_export import compute_entropy_adiabat

            from proteus.interior_struct.zalmoxis import (
                load_zalmoxis_material_dictionaries,
                load_zalmoxis_solidus_liquidus_functions,
            )

            mat_dicts = load_zalmoxis_material_dictionaries()
            eos_entry = mat_dicts.get(config.interior_struct.zalmoxis.mantle_eos, {})
            paleos_eos_file = eos_entry.get('eos_file', '')
            if not paleos_eos_file or not os.path.isfile(paleos_eos_file):
                return

            melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                config.interior_struct.zalmoxis.mantle_eos, config
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
            T_surf = config.interior_energetics.tsurf_init
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
        """Update solver for next coupling step (entropy formulation)."""
        solver = interior_o.aragog_solver
        solver.parameters.solver.start_time = hf_row['Time']
        solver.parameters.solver.end_time = hf_row['Time'] + dt

        # Get entropy field from previous run
        if output_dir is not None:
            S_field = read_last_Sfield(output_dir, hf_row['Time'])
        else:
            sol = solver.solution
            S_field = sol.y[:, -1] if sol is not None else None

        if S_field is not None:
            interior_o._last_entropy = S_field

        # Update boundary conditions (F_atm from atmosphere module)
        solver.parameters.boundary_conditions.outer_boundary_value = hf_row['F_atm']

        # Update tidal heating
        solver.parameters.energy.tidal_array = interior_o.tides

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

        if config.interior_struct.module == 'self':
            solver.parameters.mesh.inner_radius = (
                config.interior_struct.corefrac * hf_row['R_int']
            )
        # For Zalmoxis: inner_radius is set at setup_solver from Zalmoxis output.
        # The EOS file (zalmoxis_output.dat) is refreshed by the Zalmoxis solver
        # before the interior step, so Aragog.reset() will re-read it.

    def run_solver(self, hf_row, interior_o, dirs):
        # Dispatch to JAX solver if configured
        if self._use_jax:
            return self._jax_runner.run_solver(hf_row, interior_o, dirs)

        # Run scipy BDF Aragog solver
        self.aragog_solver.solve()

        # Get clean output via the public API
        out = self.aragog_solver.get_state()

        # Build PROTEUS helpfile output from SolverOutput
        output = self._build_helpfile_output(out, hf_row)

        # Store arrays on interior object for inter-module access
        interior_o.phi = out.phi_stag
        interior_o.visc = out.visc_stag
        interior_o.density = out.rho_stag
        interior_o.radius = out.r_basic / 1e3  # km
        interior_o.mass = out.mass_stag
        interior_o.temp = out.T_stag
        interior_o.pres = out.P_stag

        sim_time = self.aragog_solver.parameters.solver.end_time

        # Write output to a file
        self._write_output_ncdf(dirs['output'], sim_time, out)

        return sim_time, output

    @staticmethod
    def _build_helpfile_output(out: SolverOutput, hf_row: dict) -> dict:
        """Build the PROTEUS helpfile dict from SolverOutput."""
        logger.info(
            'Aragog entropy: T_surf=%.0f K, T_cmb=%.0f K, '
            'Phi=%.3f, status=%d',
            out.T_magma, out.T_core, out.Phi_global, out.status,
        )

        return {
            'M_mantle': out.M_mantle,
            'T_magma': out.T_magma,
            'Phi_global': out.Phi_global,
            'RF_depth': out.RF_depth,
            'F_int': hf_row['F_atm'],
            'M_mantle_liquid': out.M_mantle_liquid,
            'M_mantle_solid': out.M_mantle_solid,
            'Phi_global_vol': out.Phi_global_vol,
            'T_pot': out.T_magma,
            'T_core': out.T_core,
            'E_th_mantle': out.E_th,
            'Cp_eff': out.Cp_eff,
            'F_radio': out.F_heat_total,
            'F_tidal': 0.0,
        }

    @staticmethod
    def _write_output_ncdf(output_dir: str, time: float, out: SolverOutput):
        """Write entropy solver output to NetCDF using SolverOutput."""
        fpath = os.path.join(output_dir, 'data', '%d_int.nc' % time)
        ds = nc.Dataset(fpath, mode='w')
        ds.description = 'Aragog entropy solver output'

        n_stag = len(out.S_final)
        n_basic = len(out.r_basic)
        ds.createDimension('staggered', n_stag)
        ds.createDimension('basic', n_basic)

        def _add(name, data, dim, units=''):
            v = ds.createVariable(name, np.float64, (dim,))
            v[:] = data
            v.units = units

        _add('entropy_s', out.S_final, 'staggered', 'J/kg/K')
        _add('temp_s', out.T_stag, 'staggered', 'K')
        _add('phi_s', out.phi_stag, 'staggered', '')
        _add('radius_s', out.r_stag / 1e3, 'staggered', 'km')
        _add('pres_s', out.P_stag / 1e9, 'staggered', 'GPa')
        _add('radius_b', out.r_basic / 1e3, 'basic', 'km')
        _add('log10visc_s', np.log10(np.maximum(out.visc_stag, 1e-10)), 'staggered', 'Pa s')
        _add('density_s', out.rho_stag, 'staggered', 'kg m-3')
        _add('Ftotal_b', out.heat_flux, 'basic', 'W m-2')
        _add('Htotal_s', out.heating, 'staggered', 'W kg-1')
        _add('mass_s', out.mass_stag, 'staggered', 'kg')

        ds.createVariable('time', np.float64)
        ds['time'][0] = float(time)
        ds['time'].units = 'yr'

        ds.createVariable('phi_global', np.float64)
        ds['phi_global'][0] = out.Phi_global

        ds.close()


def read_last_Sfield(output_dir: str, time: float):
    """Read the entropy field from the previous Aragog NetCDF output."""
    fpath = os.path.join(output_dir, 'data', '%d_int.nc' % time)
    ds = nc.Dataset(fpath)
    try:
        S_stag = np.array(ds['entropy_s'][:])
    except (KeyError, IndexError):
        # Fallback: older output format without entropy; read T and convert
        logger.warning('No entropy_s in %s; falling back to temp_s', fpath)
        S_stag = np.array(ds.get('temp_s', ds.get('temp_b', [3200.0]))[:])
    ds.close()
    return S_stag


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
