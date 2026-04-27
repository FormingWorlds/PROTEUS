from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from proteus.atmos_clim.common import Atmos_t
from proteus.interior.common import Interior_t
from proteus.interior.timestep import next_step
from proteus.utils.constants import (
    M_earth,
    const_G,
    const_R,
    radnuc_data,
    secs_per_year,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

class BoundaryRunner():

    def __init__(self, config: Config, dirs: dict, hf_row: dict, hf_all:
                 pd.DataFrame, interior_o: Interior_t, atmos_o: Atmos_t):

        self.curr_time = hf_row["Time"] * secs_per_year
        self.dt        = self.compute_time_step(config, dirs, hf_row, hf_all, interior_o) * secs_per_year
        self.iteration = 1 if hf_all is None else len(hf_all)

        self.planet_radius = hf_row["R_int"]
        self.planet_mass   = config.struct.mass_tot * M_earth
        self.core_mass     = hf_row["M_core"]
        self.m_atm         = hf_row["M_atm"]
        self.f_atm         = hf_row["F_atm"]

        cp_layer = getattr(getattr(atmos_o, "_atm", None), "layer_cp", None)
        if cp_layer is not None:
            cp_arr = np.asarray(cp_layer, dtype=float).ravel()
            valid = np.isfinite(cp_arr)
            if np.any(valid):
                # Use profile-mean atmospheric heat capacity across all valid layers.
                self.atmosphere_heat_capacity = float(np.mean(cp_arr[valid]))  # J/kg/K
            else:
                self.atmosphere_heat_capacity = 1.7e4  # J/kg/K for H2 at 2000K
        else:
            self.atmosphere_heat_capacity = 1.7e4  # J/kg/K for H2 at 2000K

        # Prefer Zalmoxis-derived CMB radius when available.
        # Fall back to corefrac-based radius for non-Zalmoxis runs.
        self.core_radius = float(hf_row.get("R_core", config.struct.corefrac * self.planet_radius))
        self.core_frac   = self.core_radius / self.planet_radius

        self.mantle_radius   = self.planet_radius - self.core_radius
        self.mantle_volume   = (4/3) * np.pi * (self.planet_radius**3 - self.core_radius**3)
        self.mantle_mass     = (self.planet_mass - self.core_mass)
        self.bulk_density    = self.mantle_mass / self.mantle_volume
        self.surface_gravity = const_G * self.planet_mass / self.planet_radius**2

        self.rtol = config.interior.boundary.rtol
        self.atol = config.interior.boundary.atol

        if interior_o.ic == 2 or config.struct.module == "zalmoxis":
            self.T_p_0    = hf_row.get("T_magma")
            self.T_surf_0 = hf_row.get("T_surf")
        else:
            self.T_p_0    = config.interior.boundary.T_p_0
            self.T_surf_0 = self.T_p_0

        if self.T_surf_0 > self.T_p_0:
            self.T_surf_0 = self.T_p_0 - 1.0  # Ensure initial surface temperature does not exceed potential temperature

        self.T_solidus                = config.interior.boundary.T_solidus
        self.T_liquidus               = config.interior.boundary.T_liquidus
        self.critical_melt_fraction   = config.interior.boundary.critical_melt_fraction
        self.Tsurf_event_change       = config.interior.boundary.Tsurf_event_change

        # Material constants
        self.critical_rayleigh_number = config.interior.boundary.critical_rayleigh_number  # dimensionless
        self.heat_fusion_silicate     = config.interior.boundary.heat_fusion_silicate  # J/kg
        self.nusselt_exponent         = config.interior.boundary.nusselt_exponent  # dimensionless
        self.silicate_heat_capacity   = config.interior.boundary.silicate_heat_capacity  # J/kg/K
        self.thermal_conductivity     = config.interior.boundary.thermal_conductivity  # W/m/K
        self.thermal_diffusivity      = config.interior.boundary.thermal_diffusivity  # m^2/s
        self.thermal_expansivity      = config.interior.boundary.thermal_expansivity  # 1/K
        self.const_R                  = const_R  # Gas constant [J/(mol·K)]

        # Viscosity parameterisation selection
        # 1 = constant viscosity
        # 2 = aggregate viscosity (smooth transition between solid and melt)
        # 3 = Arrhenius temperature-dependent viscosity
        self.viscosity_model = config.interior.boundary.viscosity_model

        # Constant viscosity model parameters
        self.eta_constant = config.interior.boundary.eta_constant  # Pa s

        # Aggregate viscosity parameters
        self.transition_width = config.interior.boundary.transition_width  # dimensionless
        self.eta_solid_const  = config.interior.boundary.eta_solid_const  # Pa s
        self.eta_melt_const   = config.interior.boundary.eta_melt_const  # Pa s

        # Arrhenius solid mantle parameters
        self.dynamic_viscosity = config.interior.boundary.dynamic_viscosity  # Pa s
        self.activation_energy = config.interior.boundary.activation_energy  # J/mol
        self.creep_parameter   = config.interior.boundary.creep_parameter  # dimensionless, for non-Arrhenius power-law creep (not currently used)

        # Arrhenius magma ocean parameters (Vogel-Fulcher-Tammann)
        self.viscosity_prefactor = config.interior.boundary.viscosity_prefactor  # Pa s
        self.viscosity_activation_temp = config.interior.boundary.viscosity_activation_temp  # K

        # Radioactive heating parameters
        self.use_radiogenic_heating = config.interior.radiogenic_heat
        self.age_ini                = config.star.age_ini
        self.radio_tref             = config.delivery.radio_tref # in Gyr
        self.U_abun                 = config.delivery.radio_U * 1e-6  # Convert ppm to kg/kg
        self.Th_abun                = config.delivery.radio_Th * 1e-6  # Convert ppm to kg/kg
        self.K_abun                 = config.delivery.radio_K * 1e-6  # Convert ppm to kg/kg

        # use tidal heating from orbit module if enabled, otherwise zero
        self.use_tidal_heating = config.interior.tidal_heat
        self.tidal_term        = interior_o.tides[0] if self.use_tidal_heating else 0.0

        # Logging setup
        self.logging = config.interior.boundary.logging

    @staticmethod
    def compute_time_step(config: Config, dirs: dict, hf_row: dict,
                          hf_all: pd.DataFrame, interior_o: Interior_t) -> float:
        if interior_o.ic == 1:
            return 0.0
        else:
            step_sf = 1.0  # dt scale factor
            return next_step(config, dirs, hf_row, hf_all, step_sf)

    def viscosity_aggregate_model(self, phi: float) -> float:
        """
        Calculate the aggregate viscosity using a smooth transition function.

        This function blends between solid and magma ocean viscosities using a
        hyperbolic tangent transition function centered at the critical melt fraction.

        Parameters
        ----------
        phi : float
            Melt fraction [0-1]

        Returns
        -------
        float
            Aggregate dynamic viscosity [Pa s]
        """
        # Use constant viscosities from config
        eta_solid = self.eta_solid_const
        eta_magma = self.eta_melt_const

        # Calculate transition parameter
        y = (phi - self.critical_melt_fraction) / self.transition_width

        # Calculate transition function (0 to 1)
        z = 0.5 * (1 + np.tanh(y))

        # Calculate aggregate viscosity using logarithmic interpolation
        log_eta = z * np.log10(eta_magma) + (1 - z) * np.log10(eta_solid)
        eta = 10**log_eta

        return eta

    def viscosity_arrhenius(self, T_p: float, phi: float) -> float:
        """
        Calculate viscosity using an Arrhenius temperature-dependent model.

        Selects between solid mantle (Arrhenius) and magma ocean (Vogel-Fulcher-Tammann)
        viscosity based on melt fraction relative to critical melt fraction.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        phi : float
            Melt fraction [0-1]

        Returns
        -------
        float
            Temperature-dependent dynamic viscosity [Pa s]
        """
        if phi < self.critical_melt_fraction:
            # Solid mantle: use Arrhenius equation
            eta = self.dynamic_viscosity * np.exp(
                self.activation_energy / (self.const_R * T_p)
            )

            eta = eta *np.exp(-self.creep_parameter * phi)  # Adjust for melt weakening

        else:
            # Magma ocean: use Vogel-Fulcher-Tammann relation adjusted for crystal fraction
            # Viscosity of liquid magma
            eta_l = self.viscosity_prefactor * np.exp(
                self.viscosity_activation_temp / (T_p - 1000.0)
            )

            # Adjust viscosity based on melt fraction
            eta = eta_l / (1.0 - (1.0 - phi) / (1.0 - self.critical_melt_fraction))**2.5

        return eta

    def viscosity(self, T_p: float, T_surf: float, phi: float) -> float:
        """
        Calculate viscosity using the selected parameterisation model.

        Dispatcher method that selects between constant, aggregate, or Arrhenius
        viscosity models based on the configuration.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        T_surf : float
            Surface temperature of the planet [K]
        phi : float
            Melt fraction [0-1]

        Returns
        -------
        float
            Dynamic viscosity [Pa s]
        """
        if self.viscosity_model == 1:
            return self.eta_constant
        elif self.viscosity_model == 2:
            return self.viscosity_aggregate_model(phi)
        elif self.viscosity_model == 3:
            return self.viscosity_arrhenius(T_p, phi)
        else:
            log.warning(f"Unknown viscosity model {self.viscosity_model}, defaulting to aggregate (2)")
            return self.viscosity_aggregate_model(phi)

    def rayleigh_number(self, T_p: float, T_surf: float, phi: float) -> float:
        """
        Calculate the Rayleigh number for mantle convection.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        T_surf : float
            Surface temperature of the planet [K]
        phi : float
            Melt fraction [0-1]

        Returns
        -------
        float
            Rayleigh number [dimensionless]
        """
        # Determine viscosity based on configuration
        eta = self.viscosity(T_p, T_surf, phi)

        # Calculate Rayleigh number
        Ra = ((self.bulk_density * self.surface_gravity * self.thermal_expansivity *
               np.abs(T_p - T_surf) * self.mantle_radius**3) /
              (eta * self.thermal_diffusivity))

        return Ra

    def q_m(self, T_p: float, T_surf: float, phi: float) -> float:
        """
        Calculate the convective heat flux from the mantle.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        T_surf : float
            Surface temperature of the planet [K]
        phi : float
            Melt fraction [0-1]

        Returns
        -------
        float
            Convective heat flux [W/m^2]
        """
        Ra = self.rayleigh_number(T_p, T_surf, phi)

        # Nusselt number scaling relation
        Nu = (Ra / self.critical_rayleigh_number)**self.nusselt_exponent

        # Calculate convective heat flux
        q_m_val = Nu * self.thermal_conductivity * np.abs(T_p - T_surf) / self.mantle_radius

        return q_m_val

    def radioactive_heating(self, t: float) -> float:
        """
        Calculate the volumetric radioactive heating rate as a function of time.

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        float
            Radioactive heating rate per unit volume [W/kg]
        """
        # Get radioactive constants from radnuc_data

        radio_t0 = (self.radio_tref-self.age_ini) * 1e9 * secs_per_year

        # Radioactive isotope properties from radnuc_data
        # Convert half-lives to decay constants: lambda = ln(2) / half-life
        K_decay_constant  = np.log(2) / (radnuc_data['k40']['halflife'] * secs_per_year)  # 1/s
        K_heat_production = radnuc_data['k40']['heatprod']  # W/kg of K-40
        K40_abun = self.K_abun * radnuc_data['k40']['abundance']  # kg(K-40)/kg(mantle)

        Th_decay_constant  = np.log(2) / (radnuc_data['th232']['halflife'] * secs_per_year)  # 1/s
        Th_heat_production = radnuc_data['th232']['heatprod']  # W/kg of Th-232
        Th232_abun = self.Th_abun * radnuc_data['th232']['abundance']  # kg(Th-232)/kg(mantle)

        U238_decay_constant  = np.log(2) / (radnuc_data['u238']['halflife'] * secs_per_year)  # 1/s
        U238_heat_production = radnuc_data['u238']['heatprod']  # W/kg of U-238
        U238_abun = self.U_abun * radnuc_data['u238']['abundance']  # kg(U-238)/kg(mantle)

        U235_decay_constant  = np.log(2) / (radnuc_data['u235']['halflife'] * secs_per_year)  # 1/s
        U235_heat_production = radnuc_data['u235']['heatprod']  # W/kg of U-235
        U235_abun = self.U_abun * radnuc_data['u235']['abundance']  # kg(U-235)/kg(mantle)

        # Calculate heating contributions from each isotope
        K_term = K40_abun * K_heat_production * np.exp(
            K_decay_constant * (radio_t0 - t)
        )
        U238_term = U238_abun * U238_heat_production * np.exp(
            U238_decay_constant * (radio_t0 - t)
        )
        U235_term = U235_abun * U235_heat_production * np.exp(
            U235_decay_constant * (radio_t0 - t)
        )
        Th_term = Th232_abun * Th_heat_production * np.exp(
            Th_decay_constant * (radio_t0 - t)
        )
        H_total = K_term + U238_term + U235_term + Th_term

        if self.use_radiogenic_heating:
            return H_total
        else:
            return 0.0

    def melt_fraction(self, T_p: float) -> float:
        """
        Calculate the melt fraction from potential temperature.

        Parameters
        ----------
        T_p : float
            Potential temperature [K]

        Returns
        -------
        float
            Melt fraction [0-1]
        """
        phi = (T_p - self.T_solidus) / (self.T_liquidus - self.T_solidus)
        return np.clip(phi, 0.0, 1.0)

    def r_s(self, T_p) -> float:
        """
        Calculate the solidification radius directly from the potential temperature.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]

        Returns
        -------
        float
            Solidification radius [m]
        """
        phi = self.melt_fraction(T_p)

        if phi >= 1.0:
            return self.core_radius  # Fully molten, solidification radius at core-mantle boundary
        elif phi <= 0.0:
            return self.planet_radius  # Fully solid, solidification radius at surface
        else:
            # Linear interpolation between core and surface based on melt fraction
            return (self.planet_radius**3 - phi * (self.planet_radius**3 - self.core_radius**3))**(1/3)

    def drs_dTp(self, T_p: float) -> float:
        """
        Calculate the derivative of solidification radius with respect to potential temperature.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]

        Returns
        -------
        float
            Derivative dr_s/dT_p [m/K]
        """

        # Compute r_s
        r_s = self.r_s(T_p)

        # Compute dr_s/dT_p using the chain rule as derived in drs_dt
        volume_diff = self.planet_radius**3 - self.core_radius**3
        T_range = self.T_liquidus - self.T_solidus

        dr_s_dT_p = -(volume_diff) / (3 * T_range * r_s**2)

        return dr_s_dT_p

    def dT_pdt(self, T_p: float, T_surf: float, t: float) -> float:
        """
        Calculate the rate of change of potential temperature of the mantle.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        T_surf : float
            Surface temperature of the planet [K]
        t : float
            Time [s]

        Returns
        -------
        float
            Rate of change of potential temperature [K/s]
        """
        # Calculate melt fraction
        phi = self.melt_fraction(T_p)

        # Calculate convective heat flux
        q_m_val = self.q_m(T_p, T_surf, phi)

        # Calculate radioactive heating
        Q_val = self.radioactive_heating(t)

        # Energy balance numerator: heat loss - radiogenic heating
        numerator = (-4 * np.pi * self.planet_radius**2 * q_m_val +
                     self.mantle_mass * (Q_val + self.tidal_term))

        r_s_val = self.r_s(T_p)  # Update r_s_val based on current T_p

        # Calculate dr_s/dT_p for the latent heat term
        dr_s_dT_p = self.drs_dTp(T_p)

        # Energy balance denominator: sensible heat + latent heat
        if r_s_val < self.planet_radius:
            denominator = ((4/3) * np.pi * self.bulk_density * self.silicate_heat_capacity *
                           (self.planet_radius**3 - r_s_val**3) -
                           4 * np.pi * r_s_val**2 * self.bulk_density * self.heat_fusion_silicate * dr_s_dT_p)
        else:
            denominator = self.silicate_heat_capacity * self.mantle_mass

        dT_pdt_val = numerator / denominator

        return dT_pdt_val

    def dT_surfdt(self, T_p: float, T_surf: float) -> float:
        """
        Calculate the rate of change of surface temperature of the planet.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]
        T_surf : float
            Surface temperature of the planet [K]

        Returns
        -------
        dT_surfdt_val : float
            Rate of change of surface temperature [K/s]
        """
        phi = self.melt_fraction(T_p)
        q_m_val = self.q_m(T_p, T_surf, phi)

        if T_p==T_surf:
            delta = 1e-3  # Small temperature difference to avoid division by zero
        else:
            delta = self.thermal_conductivity * (T_p - T_surf) / q_m_val

        numerator = 4 * np.pi * self.planet_radius**2 * (q_m_val - self.f_atm)
        denominator = self.atmosphere_heat_capacity * self.m_atm + \
            (4/3) * np.pi * self.silicate_heat_capacity * self.bulk_density * (self.planet_radius**3 - (self.planet_radius-delta)**3)

        dT_surfdt_val = numerator / denominator

        return dT_surfdt_val

    def thermal_rhs(self, t: float, y: list) -> list:
        """
        Right-hand side function for the coupled thermal evolution ODEs.
        like scipy.integrate.solve_ivp.

        Parameters
        ----------
        t : float
            Time [s]
        y : list or array-like
            State vector containing [T_p, T_surf, r_s] where:
            - T_p is the mantle potential temperature [K]
            - T_surf is the surface temperature [K]

        Returns
        -------
        list
            Time derivatives [dT_p/dt, dT_surf/dt, drs/dt] where:
            - dT_p/dt is the rate of change of potential temperature [K/s]
            - dT_surf/dt is the rate of change of surface temperature [K/s]
        """
        T_p, T_surf = y

        if T_surf>T_p:
            T_surf = T_p-1.0  # Ensure surface temperature does not exceed potential temperature

        dTp = self.dT_pdt(T_p, T_surf, t)
        dTs = self.dT_surfdt(T_p, T_surf)

        return [dTp, dTs]

    def run_solver(self, hf_row: dict, interior_o: Interior_t, dirs: dict) -> tuple:
        """
        Run the thermal evolution solver for a single timestep.

        Parameters
        ----------
        interior_o : Interior_t
            Interior model object containing structural and thermodynamic state
        dirs : dict
            Dictionary of directory paths for output files

        Returns
        -------
        tuple
            A tuple containing:
            - sim_time : float
                Final simulation time after integration [s]
            - output : dict
                Dictionary containing thermal evolution results
        """
        # Set up CSV logging for step diagnostics.
        if self.logging:
            output_dir = dirs.get('output', '.')
            csv_log_file = f"{output_dir}/boundary_solver_debug.csv"

            csv_columns = [
                "Time_years",
                "T_p_K",
                "T_surf_K",
                "q_m_val_W/m2",
                "viscosity_Pa_s",
                "rayleigh_number",
                "atm_heat_flux_W/m2",
            ]
            csv_needs_header = not pd.io.common.file_exists(csv_log_file)

        def tsurf_change_event(t: float, y: list) -> float:
            """
            Trigger when the surface temperature differs from the initial value
            by the configured threshold.
            """
            _ = t
            T_surf = y[1]
            return np.abs(T_surf - self.T_surf_0) - self.Tsurf_event_change

        tsurf_change_event.terminal = True
        tsurf_change_event.direction = 1

        y0 = [self.T_p_0, self.T_surf_0]
        t_span = (self.curr_time, self.curr_time + self.dt)

        sol = solve_ivp(
            self.thermal_rhs,
            t_span,
            y0,
            method='BDF',
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
            events=tsurf_change_event,
        )

        # Extract results
        T_p_final     = sol.y[0, -1]
        T_surf_final  = sol.y[1, -1]
        t_final       = sol.t[-1]
        phi_final     = self.melt_fraction(T_p_final)
        r_s_final     = self.r_s(T_p_final)
        sim_time      = t_final/secs_per_year  # convert back to years
        phi_final     = self.melt_fraction(T_p_final)
        visc_final    = self.viscosity(T_p_final, T_surf_final, phi_final)
        f_radio_final = self.radioactive_heating(t_final) * self.mantle_mass

        # Log final timestep values to CSV
        if self.logging:
            q_m_val = self.q_m(T_p_final, T_surf_final, phi_final)
            ra_final = self.rayleigh_number(T_p_final, T_surf_final, phi_final)

            with open(csv_log_file, mode='a', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_columns)
                if csv_needs_header:
                    writer.writeheader()

                writer.writerow(
                    {
                        "Time_years": sim_time,
                        "T_p_K": T_p_final,
                        "T_surf_K": T_surf_final,
                        "q_m_val_W/m2": q_m_val,
                        "viscosity_Pa_s": visc_final,
                        "rayleigh_number": ra_final,
                        "atm_heat_flux_W/m2": self.f_atm,
                    }
                )

        m_liquid = (4/3) * np.pi * self.bulk_density * (self.planet_radius**3 - r_s_final**3)
        m_solid  = self.mantle_mass - m_liquid

        if T_surf_final > T_p_final:
            T_surf_final = T_p_final - 1.0  # Ensure surface temperature does not exceed potential temperature

        output = {
            "T_magma": T_p_final,
            "T_pot": T_p_final,
            "T_surf": T_surf_final,
            "F_int": self.f_atm,
            "Phi_global": phi_final,
            "Phi_global_vol": phi_final,
            "F_radio": f_radio_final/(4*np.pi*self.planet_radius**2),
            "RF_depth": phi_final * (1.0 - self.core_frac),
            "M_mantle_liquid": m_liquid,
            "M_mantle_solid": m_solid,
            "F_tidal": self.tidal_term*self.mantle_mass/(4*np.pi*self.planet_radius**2) if self.use_tidal_heating else 0.0,
            "M_mantle": self.mantle_mass,
        }

        # Store arrays
        interior_o.phi = np.array([output['Phi_global']])
        interior_o.mass = np.array([output['M_mantle']])
        interior_o.visc = np.array([visc_final])
        interior_o.density = np.array([self.bulk_density])
        interior_o.temp = np.array([output['T_magma']])
        interior_o.pres = np.array([hf_row['P_surf']])
        interior_o.radius = np.array([self.core_radius, hf_row['R_int']])

        return sim_time, output
