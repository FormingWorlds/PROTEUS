# JAX Aragog interior module
#
# Called by AragogRunner when config.interior_energetics.aragog.jax = True.
# Delegates the ODE solve to aragog.jax.solver (diffrax). Output
# uses the same helpfile dict contract as the scipy version.
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

from proteus.interior_energetics.common import Interior_t

jax.config.update('jax_enable_x64', True)

logger = logging.getLogger('fwl.' + __name__)

if TYPE_CHECKING:
    from proteus.config import Config


class AragogJAXRunner:
    """JAX-based Aragog entropy solver backend.

    Called by AragogRunner when config.interior_energetics.aragog.jax is True.
    All setup (config, EOS tables, mesh, IC) is done by AragogRunner;
    this class only handles JAX-specific components and the diffrax solve.
    """

    def __init__(
        self,
        config: 'Config',
        dirs: dict,
        hf_row: dict,
        hf_all,
        interior_o: Interior_t,
    ):
        # The numpy AragogRunner already ran setup (logger, time step, solver).
        # We just grab the solver reference and build JAX-specific components.
        self.aragog_solver = interior_o.aragog_solver
        self._config = config

        # Build JAX components (once, cached on interior_o)
        if not hasattr(interior_o, '_jax_eos') or interior_o._jax_eos is None:
            self._build_jax_components(config, interior_o)
        self._eos_jax = interior_o._jax_eos
        self._params_jax = interior_o._jax_params
        self._bc_jax = interior_o._jax_bc

        # Rebuild mesh arrays every step (mesh may change from Zalmoxis)
        self._mesh_jax = self._build_mesh_arrays()

    def _build_jax_components(self, config: Config, interior_o: Interior_t):
        """Build JAX EOS, params, and BCs from the numpy solver."""
        from aragog.jax.eos import EntropyEOS_JAX
        from aragog.jax.phase import PhaseParams
        from aragog.jax.solver import BoundaryParams

        # EOS: load from the same directory as numpy solver
        spider_eos_dir = interior_o._spider_eos_dir
        if spider_eos_dir and os.path.isdir(spider_eos_dir):
            eos_jax = EntropyEOS_JAX(spider_eos_dir)
        else:
            raise FileNotFoundError(
                f'PALEOS P-S tables not found for JAX solver: {spider_eos_dir}'
            )
        interior_o._jax_eos = eos_jax

        # Phase parameters from config
        interior_o._jax_params = PhaseParams(
            phi_rheo=config.interior_energetics.rfront_loc,
            phi_width=config.interior_energetics.rfront_wid,
            viscosity_solid=1e21,  # TODO: wire from config
            viscosity_liquid=1e-1,
            grain_size=config.interior_energetics.grain_size,
            k_solid=4.0,
            k_liquid=2.0,
            conduction=config.interior_energetics.trans_conduction,
            convection=config.interior_energetics.trans_convection,
            grav_sep=config.interior_energetics.trans_grav_sep,
            mixing=config.interior_energetics.trans_mixing,
            eddy_diff_thermal=1.0,
            eddy_diff_chemical=1.0,
            kappah_floor=config.interior_energetics.kappah_floor,
        )

        # Boundary conditions
        bc_cfg = self.aragog_solver.parameters.boundary_conditions
        interior_o._jax_bc = BoundaryParams(
            outer_bc_type=bc_cfg.outer_boundary_condition,
            outer_bc_value=bc_cfg.outer_boundary_value,
            emissivity=bc_cfg.emissivity,
            T_eq=bc_cfg.equilibrium_temperature,
            inner_bc_type=bc_cfg.inner_boundary_condition,
            inner_bc_value=bc_cfg.inner_boundary_value,
            core_density=self.aragog_solver.parameters.mesh.core_density,
            core_heat_capacity=bc_cfg.core_heat_capacity,
            tfac_core_avg=getattr(bc_cfg, 'tfac_core_avg', 1.147),
        )

        logger.info('JAX Aragog components built')

    def _build_mesh_arrays(self):
        """Convert numpy mesh to JAX arrays."""
        from aragog.jax.phase import MeshArrays
        return MeshArrays.from_numpy_mesh(self.aragog_solver.evaluator.mesh)

    def run_solver(self, hf_row, interior_o, dirs):
        """Run the JAX solver and return output in the same format."""
        from aragog.jax.solver import BoundaryParams, solve_entropy

        solver = self.aragog_solver

        # Get entropy IC
        if hasattr(interior_o, '_last_entropy') and interior_o._last_entropy is not None:
            S0 = jnp.asarray(interior_o._last_entropy)
        else:
            S0 = jnp.asarray(solver._S0)

        # Time interval
        t_start = solver.parameters.solver.start_time
        t_end = solver.parameters.solver.end_time
        if t_end <= t_start:
            t_end = t_start + 1.0  # minimum 1 yr step

        # Update BC value (F_atm from atmosphere)
        bc = self._bc_jax
        if bc.outer_bc_type == 4:  # prescribed flux
            bc = BoundaryParams(
                outer_bc_type=bc.outer_bc_type,
                outer_bc_value=float(hf_row['F_atm']),
                emissivity=bc.emissivity,
                T_eq=bc.T_eq,
                inner_bc_type=bc.inner_bc_type,
                inner_bc_value=bc.inner_bc_value,
                core_density=bc.core_density,
                core_heat_capacity=bc.core_heat_capacity,
                tfac_core_avg=bc.tfac_core_avg,
            )

        # Heating (radionuclide + tidal, computed from numpy solver's config)
        n_stag = len(S0)
        heating_np = np.zeros(n_stag)
        if self._config.interior_energetics.heat_radiogenic:
            for r in solver.parameters.radionuclides:
                heating_np += r.get_heating(t_start)
        if self._config.interior_energetics.heat_tidal:
            tides = getattr(interior_o, 'tides', [0.0])
            if len(tides) == 1:
                heating_np += tides[0]
            elif len(tides) == n_stag:
                heating_np += np.array(tides)
        heating = jnp.asarray(heating_np)
        self._last_heating = heating_np  # store for _extract_output

        # Solve
        atol = max(self._config.interior_energetics.num_tolerance, 0.01)
        rtol = self._config.interior_energetics.num_tolerance

        logger.info(
            'JAX Aragog: integrating [%.2e, %.2e] yr, N=%d',
            t_start, t_end, n_stag,
        )

        result = solve_entropy(
            S0, t_start, t_end,
            self._eos_jax, self._params_jax, self._mesh_jax, bc, heating,
            atol=atol, rtol=rtol, max_steps=100_000,
            method='tsit5',
        )

        if not result.success:
            logger.error('JAX Aragog solver failed (steps=%d)', result.n_steps)

        # Save entropy for next step
        interior_o._last_entropy = np.asarray(result.S_final)

        # Also run numpy solver's get_state for the output contract
        # (temporarily set the numpy solver's solution to match JAX result)
        # Instead, compute output directly from JAX results
        out = self._extract_output(result, hf_row, interior_o)
        sim_time = result.t_final

        # Write NetCDF
        self._write_ncdf(dirs['output'], sim_time, result)

        return sim_time, out

    def _extract_output(self, result, hf_row, interior_o):
        """Build PROTEUS helpfile dict from JAX solve result."""
        eos = self._eos_jax
        mesh = self._mesh_jax

        S = result.S_final
        P = mesh.P_stag
        T = np.asarray(eos.temperature(P, S))
        phi = np.asarray(eos.melt_fraction(P, S))
        rho = np.asarray(eos.density(P, S))
        vol = np.asarray(mesh.volume)

        T_magma = float(T[-1])
        T_core = float(T[0])
        mass = rho * vol
        M_mantle = float(mass.sum())
        Phi_global = float(np.dot(phi, vol) / vol.sum())

        # Rheological front
        r_basic = np.asarray(mesh.radii_basic)
        phi_rheo = self._config.interior_energetics.rfront_loc
        if Phi_global > 0.99:
            rf = float(r_basic[0])
        elif Phi_global < 0.01:
            rf = float(r_basic[-1])
        else:
            phi_basic = np.asarray(mesh.quantity_matrix @ jnp.asarray(phi))
            idx = np.argmin(np.abs(phi_basic - phi_rheo))
            rf = float(r_basic[idx])
        R_outer = float(r_basic[-1])
        RF_depth = 1.0 - rf / R_outer if R_outer > 0 else 0.0

        # Compute phase properties for stored arrays and Cp_eff
        from aragog.jax.phase import evaluate_phase
        props = evaluate_phase(eos, self._params_jax, P, S)
        visc = np.asarray(props.viscosity)
        cap = np.asarray(props.capacitance)  # rho * T
        Cp_eff = float(np.sum(cap * vol)) / max(M_mantle, 1.0)

        # Thermal energy (sensible, with reference Cp=1200 matching numpy)
        CP_REF = 1200.0
        E_th = float(np.sum(mass * CP_REF * T))

        # Heating flux (radionuclide + tidal combined)
        heating_np = np.asarray(self._last_heating) if hasattr(self, '_last_heating') else np.zeros_like(T)
        area_surf = 4 * np.pi * float(r_basic[-1]) ** 2
        F_heat_total = float(np.dot(heating_np, mass)) / area_surf

        # Store arrays on interior_o
        interior_o.phi = phi
        interior_o.visc = visc
        interior_o.density = rho
        interior_o.radius = r_basic / 1e3
        interior_o.mass = mass
        interior_o.temp = T
        interior_o.pres = np.asarray(P)

        logger.info(
            'JAX Aragog: T_surf=%.0f K, T_cmb=%.0f K, Phi=%.3f, steps=%d',
            T_magma, T_core, Phi_global, result.n_steps,
        )

        return {
            'M_mantle': M_mantle,
            'T_magma': T_magma,
            'Phi_global': Phi_global,
            'RF_depth': RF_depth,
            'F_int': hf_row['F_atm'],
            'M_mantle_liquid': float(np.sum(phi * mass)),
            'M_mantle_solid': float(M_mantle - np.sum(phi * mass)),
            'Phi_global_vol': Phi_global,  # simplified (same as mass-weighted)
            'T_pot': T_magma,
            'T_core': T_core,
            'E_th_mantle': E_th,
            'Cp_eff': Cp_eff,
            'F_radio': F_heat_total,
            'F_tidal': 0.0,
        }

    def _write_ncdf(self, output_dir: str, time: float, result):
        """Write JAX solver output to NetCDF."""
        eos = self._eos_jax
        mesh = self._mesh_jax

        S = result.S_final
        P = mesh.P_stag
        T = np.asarray(eos.temperature(P, S))
        phi = np.asarray(eos.melt_fraction(P, S))
        rho = np.asarray(eos.density(P, S))

        fpath = os.path.join(output_dir, 'data', '%d_int.nc' % time)
        ds = nc.Dataset(fpath, mode='w')
        ds.description = 'Aragog JAX entropy solver output'

        n_stag = len(np.asarray(S))
        n_basic = len(np.asarray(mesh.radii_basic))
        ds.createDimension('staggered', n_stag)
        ds.createDimension('basic', n_basic)

        def _add(name, data, dim, units=''):
            v = ds.createVariable(name, np.float64, (dim,))
            v[:] = np.asarray(data)
            v.units = units

        _add('entropy_s', S, 'staggered', 'J/kg/K')
        _add('temp_s', T, 'staggered', 'K')
        _add('phi_s', phi, 'staggered', '')
        _add('radius_s', np.asarray(mesh.radii_stag) / 1e3, 'staggered', 'km')
        _add('pres_s', np.asarray(P) / 1e9, 'staggered', 'GPa')
        _add('radius_b', np.asarray(mesh.radii_basic) / 1e3, 'basic', 'km')
        _add('density_s', rho, 'staggered', 'kg m-3')
        _add('mass_s', rho * np.asarray(mesh.volume), 'staggered', 'kg')

        # Viscosity (compute from phase evaluator)
        from aragog.jax.phase import evaluate_phase
        props = evaluate_phase(self._eos_jax, self._params_jax, P, S)
        _add('log10visc_s', np.log10(np.maximum(np.asarray(props.viscosity), 1e-10)), 'staggered', 'Pa s')

        # Heat flux and heating (recompute from full flux computation)
        from aragog.jax.phase import compute_fluxes
        flux_out = compute_fluxes(S, time, self._eos_jax, self._params_jax,
                                  self._mesh_jax, jnp.asarray(self._last_heating))
        _add('Ftotal_b', np.asarray(flux_out.heat_flux), 'basic', 'W m-2')
        _add('Htotal_s', np.asarray(flux_out.heating), 'staggered', 'W kg-1')

        ds.createVariable('time', np.float64)
        ds['time'][0] = float(time)
        ds['time'].units = 'yr'

        vol = np.asarray(mesh.volume)
        ds.createVariable('phi_global', np.float64)
        ds['phi_global'][0] = float(np.dot(phi, vol) / vol.sum())

        ds.close()
