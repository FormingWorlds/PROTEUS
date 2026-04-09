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
from proteus.interior_energetics.wrapper import get_core_density, get_core_heatcap
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
            return next_step(
                config, dirs, hf_row, hf_all, step_sf, interior_o=interior_o,
            )

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
            AragogRunner._set_entropy_ic(config, interior_o, dirs['output'], hf_row)
            # Verify and correct IC if table resolution causes > 1% T mismatch
            AragogRunner._verify_entropy_ic(
                config,
                interior_o,
                dirs['output'],
                hf_row=hf_row,
            )
        else:
            if interior_o.ic == 1:
                AragogRunner.update_structure(config, hf_row, interior_o)
                # Preserve the evolved S field across equilibration resets.
                # Use the solver's entropy_staggered accessor instead of
                # reading sol.y directly: with the v4 Bower core BC the
                # state vector is N+1 long (entropy followed by T_core),
                # and the wrapper-side handover logic only wants the
                # entropy block. The accessor strips T_core when present.
                sol = interior_o.aragog_solver.solution
                if sol is not None and sol.y.size > 0:
                    S_block = interior_o.aragog_solver.entropy_staggered
                    S_last = S_block[:, -1] if S_block.ndim > 1 else S_block
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
            # Tier 4: rtol and atol (temperature-equivalent) are now
            # config-exposed. Aragog's state variable is entropy, but
            # users specify a temperature-scale atol because dT/dt is
            # easier to reason about than dS/dt. The default 0.01 K is
            # tight enough to resolve the ~0.3 K/yr magma-ocean cooling
            # rate without slowing the BDF solver excessively.
            atol=float(config.interior_energetics.aragog.atol_temperature_equivalent),
            rtol=float(config.interior_energetics.rtol),
        )

        # Surface boundary condition mode (matches SPIDER wrapper):
        # - 'flux' (default): outer_bc=4, consume hf_row['F_atm'] unchanged.
        # - 'grey_body': outer_bc=1, Aragog recomputes
        #   emissivity * sigma * (T_surf^4 - T_eqm^4) per CVode substep using
        #   the current top-cell T. emissivity=1 matches SPIDER's -emissivity0
        #   setting for parity runs, so both solvers follow the identical
        #   physical law.
        _aragog_outer_bc = 1 if config.interior_energetics.surface_bc_mode == 'grey_body' else 4
        # Core BC mode: thread config.interior_energetics.aragog.core_bc
        # through to the Aragog solver. Valid values:
        #   'quasi_steady' (default, v3 alpha-factor)
        #   'spider_bc'    (Path A SPIDER bit-parity, v5)
        #   'bower2018'    (EXPERIMENTAL tombstone, do not use)
        aragog_cfg = getattr(config.interior_energetics, 'aragog', None)
        core_bc_str = getattr(aragog_cfg, 'core_bc', 'quasi_steady')
        if core_bc_str not in ('quasi_steady', 'spider_bc', 'bower2018'):
            logger.warning(
                'Unknown core_bc=%r, falling back to quasi_steady',
                core_bc_str,
            )
            core_bc_str = 'quasi_steady'

        boundary_conditions = _BoundaryConditionsParameters(
            # 4 = prescribed heat flux (PROTEUS coupling mode, from hf_row['F_atm'])
            # 1 = native grey-body (emissivity * sigma * (T^4 - T_eqm^4))
            outer_boundary_condition=_aragog_outer_bc,
            # first guess surface heat flux [W/m2] (only used if outer_bc=4)
            outer_boundary_value=hf_row['F_atm'],
            # 1 = core cooling model
            # 2 = prescribed heat flux
            # 3 = prescribed temperature
            inner_boundary_condition=(1),
            # core temperature [K], if inner_boundary_condition = 3
            inner_boundary_value=(4000),
            # only used in gray body BC, outer_boundary_condition = 1
            emissivity=1,
            # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature=hf_row['T_eqm'],
            # used if inner_boundary_condition = 1
            core_heat_capacity=get_core_heatcap(config, hf_row),
            # core T_avg/T_cmb ratio from adiabatic gradient (Bower+2018 Table 2)
            tfac_core_avg=config.interior_energetics.core_tfac_avg,
            # ultra-thin boundary layer parameterization (Bower et al. 2018, Eq. 18)
            param_utbl=config.interior_energetics.param_utbl,
            param_utbl_const=config.interior_energetics.param_utbl_const,
            # core BC mode (v5 Path A adds 'spider_bc')
            core_bc=core_bc_str,
        )

        # Define the inner_radius for the mesh
        if config.interior_struct.module in ('spider', 'dummy'):
            inner_radius = config.interior_struct.core_frac * hf_row['R_int']
        elif config.interior_struct.module == 'zalmoxis':
            inner_radius = hf_row.get(
                'R_core', config.interior_struct.core_frac * hf_row['R_int']
            )
        else:
            raise ValueError(
                f"Aragog: unsupported interior_struct.module = '{config.interior_struct.module}'"
            )

        mesh = _MeshParameters(
            # planet radius [m]
            outer_radius=hf_row['R_int'],
            # core radius [m]
            inner_radius=inner_radius,
            # basic nodes
            number_of_nodes=config.interior_energetics.num_levels,
            mixing_length_profile=(
                'nearest_boundary'
                if config.interior_energetics.mixing_length == 'nearest'
                else 'constant'
            ),
            core_density=get_core_density(config, hf_row),
            eos_method=1,  # 1: Adams-Williamson / 2: User defined
            surface_density=config.interior_energetics.adams_williamson_rhos,
            gravitational_acceleration=hf_row['gravity'],  # [m/s-2]
            adiabatic_bulk_modulus=config.interior_energetics.adiabatic_bulk_modulus,
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
        if config.interior_struct.module in ('spider', 'dummy'):
            initial_condition_temperature_profile = 3
            init_file_temperature_profile = os.path.join(FWL_DATA_DIR, '')
        elif config.interior_struct.module == 'zalmoxis':
            _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
            if config.interior_struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES):
                # When using Zalmoxis with temperature-dependent silicate EOS, set initial condition to user-defined temperature field (from file) in Aragog
                initial_condition_temperature_profile = 2
                init_file_temperature_profile = os.path.join(
                    outdir, 'data', 'zalmoxis_output_temp.txt'
                )
            elif config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:'):
                # For PALEOS EOS with adiabatic IC: Aragog uses IC=3 with
                # entropy tables for its entropy-conserving adiabat. After
                # initialization, _verify_entropy_ic compares against an
                # independent PALEOS entropy inversion and corrects the IC
                # if the discrepancy exceeds 1% (table resolution effect).
                initial_condition_temperature_profile = 3
                init_file_temperature_profile = ''
            else:
                # Otherwise, use the initial condition from aragog config
                initial_condition_temperature_profile = 3
                init_file_temperature_profile = os.path.join(FWL_DATA_DIR, '')
        else:
            raise ValueError(
                f"Aragog IC: unsupported interior_struct.module = '{config.interior_struct.module}'"
            )

        # When initial_thermal_state = 'self_consistent', Zalmoxis computes
        # T_surface from accretion + differentiation energy (White+Li 2025)
        # and passes it via hf_row. This overrides the config tsurf_init.
        tsurf_init = config.planet.tsurf_init
        T_surface_computed = hf_row.get('T_surface_initial', 0)
        if T_surface_computed and T_surface_computed > 0:
            logger.info(
                'Overriding tsurf_init with self-consistent thermal state: %.0f K -> %.0f K',
                tsurf_init,
                T_surface_computed,
            )
            tsurf_init = T_surface_computed

        initial_condition = _InitialConditionParameters(
            # 1 = linear profile
            # 2 = user-defined profile
            # 3 = adiabatic profile
            initial_condition=initial_condition_temperature_profile,
            # initial top temperature (K)
            surface_temperature=tsurf_init,
            basal_temperature=7000.0,
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

            mass_tot = config.planet.mass_tot or 1.0
            P_max = min(200e9, 50e9 * mass_tot + 100e9)
            LOOK_UP_DIR = Path(outdir) / 'data' / 'aragog_pt'

            # Try PALEOS-2phase first (separate solid/liquid tables)
            twophase_entry = mat_dicts.get('PALEOS-2phase:MgSiO3', {})
            solid_eos = twophase_entry.get('solid_mantle', {}).get('eos_file', '')
            liquid_eos = twophase_entry.get('melted_mantle', {}).get('eos_file', '')
            has_2phase = (
                solid_eos
                and os.path.isfile(solid_eos)
                and liquid_eos
                and os.path.isfile(liquid_eos)
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
            if config.interior_struct.melting_dir is None:
                raise ValueError(
                    'interior_struct.melting_dir must be set for non-PALEOS EOS. '
                    'Provide a melting curve folder name (e.g. "Monteux-600").'
                )
            MELTING_DIR = FWL_DATA_DIR / 'interior_lookup_tables/Melting_curves/'
            solidus_path = MELTING_DIR / config.interior_struct.melting_dir / 'solidus_P-T.dat'
            liquidus_path = (
                MELTING_DIR / config.interior_struct.melting_dir / 'liquidus_P-T.dat'
            )

        # check data exist
        if not (LOOK_UP_DIR / 'heat_capacity_melt.dat').is_file():
            raise FileNotFoundError(f'Aragog lookup data not found at {LOOK_UP_DIR}')

        # Entropy tables (optional): enable entropy-conserving adiabatic IC.
        # Generated by Zalmoxis from the same PALEOS data as Cp/alpha/rho.
        entropy_melt = LOOK_UP_DIR / 'entropy_melt.dat'
        entropy_solid = LOOK_UP_DIR / 'entropy_solid.dat'
        entropy_melt_arg = str(entropy_melt) if entropy_melt.is_file() else ''
        entropy_solid_arg = str(entropy_solid) if entropy_solid.is_file() else ''
        if entropy_melt_arg:
            logger.info('Entropy tables found: enabling entropy-conserving adiabat IC')
        else:
            logger.info('No entropy tables: Aragog will use single-phase dTdPs adiabat')

        phase_liquid = _PhaseParameters(
            density=LOOK_UP_DIR / 'density_melt.dat',
            viscosity=10.0 ** float(config.interior_energetics.melt_log10visc),
            heat_capacity=LOOK_UP_DIR / 'heat_capacity_melt.dat',
            melt_fraction=1,
            thermal_conductivity=float(config.interior_energetics.melt_cond),
            thermal_expansivity=LOOK_UP_DIR / 'thermal_exp_melt.dat',
            entropy=entropy_melt_arg,
        )

        phase_solid = _PhaseParameters(
            density=LOOK_UP_DIR / 'density_solid.dat',
            viscosity=10.0 ** float(config.interior_energetics.solid_log10visc),
            heat_capacity=LOOK_UP_DIR / 'heat_capacity_solid.dat',
            melt_fraction=0,
            thermal_conductivity=float(config.interior_energetics.solid_cond),
            thermal_expansivity=LOOK_UP_DIR / 'thermal_exp_solid.dat',
            entropy=entropy_solid_arg,
        )

        phase_mixed = _PhaseMixedParameters(
            latent_heat_of_fusion=float(config.interior_energetics.latent_heat_of_fusion),
            rheological_transition_melt_fraction=config.interior_energetics.rfront_loc,
            rheological_transition_width=config.interior_energetics.rfront_wid,
            solidus=solidus_path,
            liquidus=liquidus_path,
            phase='mixed',
            phase_transition_width=float(config.interior_energetics.phase_transition_width),
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

            if config.interior_energetics.radio_Al > 0.0:
                _append_radnuc('al26', config.interior_energetics.radio_Al)

            if config.interior_energetics.radio_Fe > 0.0:
                _append_radnuc('fe60', config.interior_energetics.radio_Fe)

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
    def _set_entropy_ic(
        config: Config, interior_o: Interior_t, outdir: str, hf_row: dict | None = None
    ):
        """Set the entropy IC via the shared compute_initial_entropy helper.

        Delegates to ``proteus.interior_energetics.common.compute_initial_entropy``
        so that SPIDER and Aragog resolve the initial mantle entropy
        through the same code path. That helper handles:

        - ``temperature_mode = "isentropic"`` -> return
          ``planet.ini_entropy`` verbatim (no EOS lookup). This is the
          CHILI protocol path used by the R8 baseline.
        - Otherwise: invert the P-S temperature table at (P=1 bar,
          tsurf_init) using the Aragog EntropyEOS on the same tables the
          solver integrates with.
        - Fallback to the PALEOS adiabat via Zalmoxis if the P-S
          inversion path is unavailable.

        After obtaining the scalar ``S_target``, build the initial
        entropy profile on the solver's staggered nodes. The baseline
        is a uniform ``S_target`` profile; add a small linear
        perturbation ``ini_dsdr * (r - r_surf)`` when
        ``config.planet.ini_dsdr`` is non-zero. This matches SPIDER's
        ``-ic_dsdr`` flag, which is the numerical perturbation SPIDER's
        BDF solver needs for stability on a uniform isentropic IC
        (the R8 CHILI baseline sets ``ini_dsdr = -4.698e-6 J/kg/K/m``).

        Parameters
        ----------
        config : Config
            PROTEUS configuration.
        interior_o : Interior_t
            Interior object with initialized EntropySolver.
        outdir : str
            Output directory (unused; kept for API compatibility).
        hf_row : dict, optional
            Helpfile row. Passed through to compute_initial_entropy so
            it can honour Zalmoxis's accretion-mode T_surface_initial
            override.
        """
        from proteus.interior_energetics.common import compute_initial_entropy

        solver = interior_o.aragog_solver
        spider_eos_dir = interior_o._spider_eos_dir or None

        S_target = compute_initial_entropy(
            config,
            hf_row=hf_row,
            spider_eos_dir=spider_eos_dir,
        )

        # Build the initial entropy profile. Baseline is uniform
        # S_target on all staggered nodes. When ini_dsdr is non-zero,
        # add a linear radial perturbation anchored at the surface
        # (S_init[i] = S_target + ini_dsdr * (r_stag[i] - r_surf)) so the
        # perturbation is exactly zero at the top cell and grows toward
        # the CMB. This matches SPIDER's -ic_dsdr semantic — a small
        # perturbation used purely to break degeneracy in the BDF
        # solver; the physical magnitude is negligible
        # (~ -14 J/kg/K over 3e6 m for the R8 default of -4.698e-6).
        P_stag = solver._P_stag_flat
        N = len(P_stag)
        S_init = np.full(N, float(S_target))

        ini_dsdr = float(config.planet.ini_dsdr)
        if ini_dsdr != 0.0:
            r_basic = np.asarray(solver._r_basic_flat, dtype=float)
            # Staggered nodes sit between basic nodes; midpoints are exact
            # for uniform meshes and a good approximation for mass-
            # coordinate meshes. The perturbation magnitude is tiny so
            # any small discretisation error is irrelevant.
            r_stag = 0.5 * (r_basic[:-1] + r_basic[1:])
            if len(r_stag) == N:
                r_surf = float(r_basic[-1])
                S_init = S_init + ini_dsdr * (r_stag - r_surf)
                logger.info(
                    'Applied ini_dsdr perturbation: %.3e J/kg/K/m, '
                    'amplitude %.2f J/kg/K from surface to CMB',
                    ini_dsdr,
                    float(abs(S_init[0] - S_init[-1])),
                )
            else:
                logger.warning(
                    'ini_dsdr perturbation skipped: r_stag length %d != '
                    'staggered nodes %d. Using uniform S_target.',
                    len(r_stag),
                    N,
                )

        solver.set_initial_entropy(S_init)
        logger.info(
            'Aragog entropy IC set from compute_initial_entropy: '
            'S_target=%.1f J/kg/K (uniform baseline, %d staggered nodes)',
            float(S_target),
            N,
        )

    @staticmethod
    def _verify_entropy_ic(
        config: Config,
        interior_o: Interior_t,
        outdir: str,
        hf_row: dict | None = None,
    ):
        """Verify Aragog's IC against an independent PALEOS entropy adiabat.

        After ``_set_entropy_ic`` has written the entropy profile into the
        solver, this function independently computes a PALEOS adiabat via
        ``zalmoxis.eos_export.compute_entropy_adiabat`` and compares its T(P)
        against the T(P) derived from Aragog's initialized entropy via the
        P-S EOS tables. A mismatch > 1% triggers an override: the entropy
        profile is replaced with values inverted from the adiabat's T profile.
        A mismatch > 5% is raised as a ``RuntimeError`` (true code-path drift).

        Notes
        -----
        History: this function was dead code between commits 63df1b1d
        (when it was introduced) and the fix that references ``solver._S0``
        / ``solver._P_stag_flat``. The original implementation referenced
        ``solver.evaluator.initial_condition.temperature``, an attribute
        of the old T-based Aragog solver that does not exist on the
        lightweight ``_EntropyEvaluator`` from the entropy rewrite. The
        resulting ``AttributeError`` was silently swallowed by a broad
        ``except Exception``. Every run since 63df1b1d logged
        "Entropy IC verification failed: '_EntropyEvaluator' object has
        no attribute 'initial_condition'" as a warning and proceeded
        without any cross-check.

        Parameters
        ----------
        config : Config
            PROTEUS configuration.
        interior_o : Interior_t
            Interior object with initialized EntropySolver.
        outdir : str
            Output directory for diagnostic files.
        """
        if not (
            config.interior_struct.module == 'zalmoxis'
            and config.interior_struct.zalmoxis.mantle_eos.startswith('PALEOS:')
        ):
            logger.debug(
                'Entropy IC cross-check skipped: not zalmoxis+PALEOS '
                "(interior_struct.module='%s')",
                config.interior_struct.module,
            )
            return

        solver = interior_o.aragog_solver

        # Catch expected failures only. AttributeError and TypeError must
        # propagate so that future API drift fails loudly in tests.
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
                logger.debug(
                    'Entropy IC cross-check skipped: PALEOS file not found (%s)',
                    paleos_eos_file,
                )
                return

            melt_funcs = load_zalmoxis_solidus_liquidus_functions(
                config.interior_struct.zalmoxis.mantle_eos, config
            )
            sol_func = liq_func = None
            if melt_funcs is not None:
                sol_func, liq_func = melt_funcs

            # Prefer 2-phase tables (clean phase-specific entropy at melting curve)
            twophase = mat_dicts.get('PALEOS-2phase:MgSiO3', {})
            solid_eos = twophase.get('solid_mantle', {}).get('eos_file', '')
            liquid_eos = twophase.get('melted_mantle', {}).get('eos_file', '')
            solid_eos = solid_eos if solid_eos and os.path.isfile(solid_eos) else None
            liquid_eos = liquid_eos if liquid_eos and os.path.isfile(liquid_eos) else None

            # ---- Current Aragog IC (from the entropy profile just set) ----
            #
            # The EntropySolver stores:
            #   solver._S0           : staggered entropy profile [J/kg/K]
            #   solver._P_stag_flat  : staggered pressures [Pa]
            #   solver.entropy_eos   : EntropyEOS with temperature_scalar(P, S)
            #
            # Derive T(P) on Aragog's staggered mesh by looking up the EOS.
            P_stag = np.asarray(solver._P_stag_flat, dtype=float)
            S_stag = np.asarray(solver._S0, dtype=float)
            if P_stag.shape != S_stag.shape:
                raise RuntimeError(
                    f'Entropy IC shape mismatch: P_stag={P_stag.shape}, S_stag={S_stag.shape}'
                )
            T_stag_aragog = np.array(
                [
                    float(solver.entropy_eos.temperature_scalar(float(p), float(s)))
                    for p, s in zip(P_stag, S_stag)
                ]
            )

            # ---- Independent PALEOS adiabat ----
            # Use the same surface temperature that _set_entropy_ic used:
            # config.planet.tsurf_init unless hf_row carries an accretion
            # override (T_surface_initial from Zalmoxis White+Li mode).
            T_surf = float(config.planet.tsurf_init)
            if hf_row is not None:
                T_surface_computed = hf_row.get('T_surface_initial', 0)
                if T_surface_computed and T_surface_computed > 0:
                    T_surf = float(T_surface_computed)

            # Surface pressure: use 1 bar (same as _set_entropy_ic and
            # common.compute_initial_entropy) to keep the cross-check
            # comparing apples to apples.
            P_surf_adiabat = 1e5
            P_cmb_adiabat = float(P_stag[0])

            result = compute_entropy_adiabat(
                eos_file=paleos_eos_file,
                T_surface=T_surf,
                P_surface=P_surf_adiabat,
                P_cmb=P_cmb_adiabat,
                n_points=500,
                solidus_func=sol_func,
                liquidus_func=liq_func,
                solid_eos_file=solid_eos,
                liquid_eos_file=liquid_eos,
            )

            # Interpolate the independent adiabat onto Aragog's staggered mesh
            from scipy.interpolate import interp1d

            T_adiabat_interp = interp1d(
                result['P'],
                result['T'],
                bounds_error=False,
                fill_value='extrapolate',
            )(P_stag)

            # ---- Compare ----
            T_diff = np.abs(T_stag_aragog - T_adiabat_interp)
            T_rel_diff = T_diff / np.maximum(T_stag_aragog, 1.0) * 100.0
            max_diff = float(np.max(T_diff))
            max_rel = float(np.max(T_rel_diff))
            mean_rel = float(np.mean(T_rel_diff))

            WARN_PCT = 1.0
            FAIL_PCT = 5.0

            if max_rel <= WARN_PCT:
                verdict = 'PASS'
            elif max_rel <= FAIL_PCT:
                verdict = 'WARN'
            else:
                verdict = 'FAIL'

            logger.info(
                'Entropy IC cross-check (Aragog): max T diff = %.1f K (%.2f%%), '
                'mean = %.3f%%, verdict = %s',
                max_diff,
                max_rel,
                mean_rel,
                verdict,
            )

            # Save diagnostic data on every call so post-mortem is possible
            diag_dir = Path(outdir) / 'data' / 'entropy_ic_verification'
            diag_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                diag_dir / 'entropy_ic_comparison.npz',
                P_staggered=P_stag,
                S_aragog=S_stag,
                T_aragog=T_stag_aragog,
                T_adiabat=T_adiabat_interp,
                T_diff=T_diff,
                S_target=result['S_target'],
            )

            if verdict == 'WARN':
                logger.warning(
                    'Entropy IC full-profile cross-check > %.1f%% '
                    '(max %.1f K / %.2f%% at depth). Diagnostic only; '
                    'not overriding the IC. See pitfall 50 in memory: '
                    'PALEOS P-T and regenerated P-S tables disagree at '
                    'high P / high T because the table regeneration '
                    'involves bilinear interpolation across non-converged '
                    'cells. Up to ~10%% is expected at M>=2.0 Earth masses.',
                    WARN_PCT,
                    max_diff,
                    max_rel,
                )
            elif verdict == 'FAIL':
                # Do NOT raise. Log only. Production runs on Habrok
                # showed that this threshold fires on every Aragog case
                # at M>=2.0 Earth masses (CMB pressure >~250 GPa) due
                # to the same table-boundary effect noted above. This
                # is an EOS self-consistency drift, not a coupling bug.
                # If you want a stricter safety net, use the surface-only
                # scalar check in _set_entropy_ic (T_check log line at
                # line ~600) which is always reliable.
                logger.warning(
                    'Entropy IC full-profile cross-check > %.1f%% '
                    '(max %.1f K / %.2f%% at depth). Diagnostic only; '
                    'NOT raising because this reflects known PALEOS '
                    'P-T vs P-S table boundary drift at high mass, '
                    'not a coupling bug. The scalar surface cross-check '
                    'logged by _set_entropy_ic is the authoritative '
                    'IC sanity check.',
                    FAIL_PCT,
                    max_diff,
                    max_rel,
                )

        except (
            FileNotFoundError,
            ImportError,
            ModuleNotFoundError,
            KeyError,
            ValueError,
        ) as e:
            # Expected failures:
            # - FileNotFoundError / ImportError: missing PALEOS files or Zalmoxis
            #   not installed
            # - KeyError: missing config keys
            # - ValueError: scipy brentq inside compute_entropy_adiabat raises
            #   this when the adiabat's internal root-find lands on a NaN, which
            #   happens when the integrated T(P) exceeds the PALEOS table range
            #   (T > ~4900 K at deep pressures for tsurf_init >= ~3500 K). The
            #   SPIDER twin in common.py has the same guard; without this we
            #   crash hot-initial-condition runs (e.g. earthlike_*_dry at
            #   tsurf_init = 4000 K, which reaches T ~ 8000 K at the CMB).
            logger.warning('Entropy IC cross-check skipped (expected error: %s)', e)

    @staticmethod
    def update_solver(dt: float, hf_row: dict, interior_o: Interior_t, output_dir: str = None):
        """Update solver for next coupling step (entropy formulation)."""
        solver = interior_o.aragog_solver
        solver.parameters.solver.start_time = hf_row['Time']
        solver.parameters.solver.end_time = hf_row['Time'] + dt

        # Get entropy field from previous run.
        # With the v4 Bower core BC the solver state vector is N+1 long
        # (entropy block followed by T_core); use entropy_staggered to
        # strip T_core so the wrapper-side _last_entropy stays the
        # entropy-only profile.
        if output_dir is not None:
            S_field = read_last_Sfield(output_dir, hf_row['Time'])
        else:
            sol = solver.solution
            if sol is not None and sol.y.size > 0:
                S_block = solver.entropy_staggered
                S_field = S_block[:, -1] if S_block.ndim > 1 else S_block
            else:
                S_field = None

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

        if config.interior_struct.module == 'spider':
            solver.parameters.mesh.inner_radius = (
                config.interior_struct.core_frac * hf_row['R_int']
            )
        # For Zalmoxis: inner_radius is set at setup_solver from Zalmoxis output.
        # The EOS file (zalmoxis_output.dat) is refreshed by the Zalmoxis solver
        # before the interior step, so Aragog.reset() will re-read it.

    def run_solver(self, hf_row, interior_o, dirs, write_data: bool = True):
        # Dispatch to JAX solver if configured
        if self._use_jax:
            return self._jax_runner.run_solver(hf_row, interior_o, dirs, write_data=write_data)

        # Run scipy BDF Aragog solver
        self.aragog_solver.solve()

        # Get clean output via the public API
        out = self.aragog_solver.get_state()

        # Build PROTEUS helpfile output from SolverOutput
        output = self._build_helpfile_output(out, hf_row)

        # Store arrays on interior object for inter-module access.
        # Radius is stored in metres (SI), matching SPIDER's convention
        # (spider.py:1185 reads radius_b directly from the SPIDER JSON
        # in metres). Previously Aragog stored km, which silently broke
        # update_structure_from_interior (wrapper.py:795) whenever
        # Zalmoxis update_interval > 0: the r <= _R_cmb comparison and
        # the downstream np.interp would use mixed units.
        interior_o.phi = out.phi_stag
        interior_o.visc = out.visc_stag
        interior_o.density = out.rho_stag
        interior_o.radius = out.r_basic  # m
        interior_o.mass = out.mass_stag
        interior_o.temp = out.T_stag
        interior_o.pres = out.P_stag

        sim_time = self.aragog_solver.parameters.solver.end_time

        # Write output to a file (skipped when dt_write suppresses this step)
        if write_data:
            self._write_output_ncdf(dirs['output'], sim_time, out)

        return sim_time, output

    @staticmethod
    def _build_helpfile_output(out: SolverOutput, hf_row: dict) -> dict:
        """Build the PROTEUS helpfile dict from SolverOutput."""
        logger.info(
            'Aragog entropy: T_surf=%.0f K, T_cmb=%.0f K, Phi=%.3f, status=%d',
            out.T_magma,
            out.T_core,
            out.Phi_global,
            out.status,
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
