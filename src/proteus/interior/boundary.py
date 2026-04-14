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
    radnuc_data,
    secs_per_year,
)

if TYPE_CHECKING:
    from proteus.config import Config


class LineLimitFilter(logging.Filter):
    def __init__(self, log_file: str, max_lines: int) -> None:
        super().__init__()
        self.log_file = log_file
        self.max_lines = max_lines
        self.line_count = self._count_lines()

    def _count_lines(self) -> int:
        try:
            with open(self.log_file, 'r', encoding='utf-8') as handle:
                return sum(1 for _ in handle)
        except FileNotFoundError:
            return 0

    def filter(self, record: logging.LogRecord) -> bool:
        if self.line_count >= self.max_lines:
            return False
        message = record.getMessage()
        lines_to_add = message.count("\n") + 1
        if self.line_count + lines_to_add > self.max_lines:
            return False
        self.line_count += lines_to_add
        return True

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
        if cp_layer is not None and len(cp_layer) > 2 and not pd.isna(cp_layer[2]):
            self.atmosphere_heat_capacity = cp_layer[2]  # J/kg/K
        else:
            self.atmosphere_heat_capacity = 1.7e4 # J/kg/K for H2 at 2000K

        self.core_radius   = config.struct.corefrac * self.planet_radius

        self.mantle_radius = self.planet_radius - self.core_radius
        self.mantle_volume = (4/3) * np.pi * (self.planet_radius**3 - self.core_radius**3)
        self.mantle_mass   = (self.planet_mass - self.core_mass)
        self.bulk_density  = self.mantle_mass / self.mantle_volume

        self.surface_gravity     = const_G * self.planet_mass / self.planet_radius**2
        self.scale_length_mantle = self.mantle_radius / 3

        self.rtol = config.interior.boundary.rtol
        self.atol = config.interior.boundary.atol

        if interior_o.ic == 2:
            self.T_p_0    = hf_row.get("T_magma", 0.0)
            self.T_surf_0 = hf_row.get("T_surf", 0.0)
        else:
            self.T_p_0    = config.interior.boundary.T_p_0
            self.T_surf_0 = config.interior.boundary.T_surf_0

        if self.T_surf_0 > self.T_p_0:
            self.T_surf_0 = self.T_p_0 - 1.0  # Ensure initial surface temperature does not exceed potential temperature

        self.T_solidus                = config.interior.boundary.T_solidus
        self.T_liquidus               = config.interior.boundary.T_liquidus
        self.critical_melt_fraction   = config.interior.boundary.critical_melt_fraction

        # Material constants
        self.critical_rayleigh_number = config.interior.boundary.critical_rayleigh_number  # dimensionless
        self.heat_fusion_silicate     = config.interior.boundary.heat_fusion_silicate  # J/kg
        self.nusselt_exponent         = config.interior.boundary.nusselt_exponent  # dimensionless
        self.silicate_heat_capacity   = config.interior.boundary.silicate_heat_capacity  # J/kg/K
        self.thermal_conductivity     = config.interior.boundary.thermal_conductivity  # W/m/K
        self.thermal_diffusivity      = config.interior.boundary.thermal_diffusivity  # m^2/s
        self.thermal_expansivity      = config.interior.boundary.thermal_expansivity  # 1/K

        # Aggregate viscosity parameters
        self.transition_width = config.interior.boundary.transition_width  # dimensionless
        self.eta_solid_const  = config.interior.boundary.eta_solid_const  # Pa s
        self.eta_melt_const   = config.interior.boundary.eta_melt_const  # Pa s

        # Radioactive heating parameters
        self.use_radiogenic_heating = config.interior.radiogenic_heat
        self.radio_tref             = config.delivery.radio_tref
        self.U_abun                 = config.delivery.radio_U * 1e-6  # Convert ppm to kg/kg
        self.Th_abun                = config.delivery.radio_Th * 1e-6  # Convert ppm to kg/kg
        self.K_abun                 = config.delivery.radio_K * 1e-6  # Convert ppm to kg/kg

    @staticmethod
    def compute_time_step(config: Config, dirs: dict, hf_row: dict,
                          hf_all: pd.DataFrame, interior_o: Interior_t) -> float:
        if interior_o.ic == 1:
            return 0.0
        else:
            step_sf = 1.0  # dt scale factor
            return next_step(config, dirs, hf_row, hf_all, step_sf)

    def viscosity_aggregate(self, phi: float) -> float:
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
        eta = self.viscosity_aggregate(phi)

        # Calculate Rayleigh number
        Ra = ((self.bulk_density * self.surface_gravity * self.thermal_expansivity *
               np.abs(T_p - T_surf) * self.scale_length_mantle**3) /
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
        q_m_val = Nu * self.thermal_conductivity * np.abs(T_p - T_surf) / self.scale_length_mantle

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
            Radioactive heating rate per unit volume [W/m^3]
        """
        # Get radioactive constants from radnuc_data
        secs_per_year = 3.154e7  # s

        # Radioactive isotope properties from radnuc_data
        # Convert half-lives to decay constants: lambda = ln(2) / half-life
        K_decay_constant  = np.log(2) / (radnuc_data['k40']['halflife'] * secs_per_year)  # 1/s
        K_heat_production = radnuc_data['k40']['heatprod']  # W/kg

        Th_decay_constant  = np.log(2) / (radnuc_data['th232']['halflife'] * secs_per_year)  # 1/s
        Th_heat_production = radnuc_data['th232']['heatprod']  # W/kg

        Ur_decay_constant  = np.log(2) / (radnuc_data['u238']['halflife'] * secs_per_year)  # 1/s
        Ur_heat_production = radnuc_data['u238']['heatprod']  # W/kg

        # Calculate heating contributions from each isotope
        K_term = self.K_abun * K_heat_production * np.exp(
            K_decay_constant * (self.radio_tref * secs_per_year - t)
        )
        Ur_term = self.U_abun * Ur_heat_production * np.exp(
            Ur_decay_constant * (self.radio_tref * secs_per_year - t)
        )
        Th_term = self.Th_abun * Th_heat_production * np.exp(
            Th_decay_constant * (self.radio_tref * secs_per_year - t)
        )
        H_total = K_term + Ur_term + Th_term

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
                     (4/3) * np.pi * self.bulk_density * Q_val *
                     (self.planet_radius**3 - self.core_radius**3))

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
        output_dir = dirs.get('output', '.')
        csv_log_file = f"{output_dir}/boundary_solver_debug.csv"
        self._run_solver_call_count = self.iteration

        csv_columns = [
            "call_index",
            "step_index",
            "n_steps",
            "time_s",
            "dt_s",
            "T_p_K",
            "T_surf_K",
            "dT_p_dt_K_per_s",
            "dT_surf_dt_K_per_s",
            "convective_heat_loss_term_W",
            "radiogenic_heating_term_W",
            "F_atmosphere_W_per_m2",
        ]
        csv_needs_header = not pd.io.common.file_exists(csv_log_file)

        y0 = [self.T_p_0, self.T_surf_0]
        t_span = (self.curr_time, self.curr_time + self.dt)
        sol = solve_ivp(self.thermal_rhs, t_span, y0, method='BDF', rtol=self.rtol, atol=self.atol, dense_output=True)

        with open(csv_log_file, mode='a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_columns)
            if csv_needs_header:
                writer.writeheader()

            for i, t in enumerate(sol.t):
                timestep = t - sol.t[i-1] if i > 0 else t - self.curr_time
                T_p = sol.y[0, i]
                T_surf = sol.y[1, i]
                dT_pdt_val = self.dT_pdt(T_p, T_surf, t)
                dT_surfdt_val = self.dT_surfdt(T_p, T_surf)
                q_m_val = self.q_m(T_p, T_surf, self.melt_fraction(T_p))
                q_m_term = 4 * np.pi * self.planet_radius**2 * q_m_val
                q_rad = self.radioactive_heating(t)
                q_rad_term = (4 / 3) * np.pi * self.bulk_density * q_rad * (self.planet_radius**3 - self.core_radius**3)

                writer.writerow(
                    {
                        "call_index": self._run_solver_call_count,
                        "step_index": i + 1,
                        "n_steps": len(sol.t),
                        "time_s": t,
                        "dt_s": timestep,
                        "T_p_K": T_p,
                        "T_surf_K": T_surf,
                        "dT_p_dt_K_per_s": dT_pdt_val,
                        "dT_surf_dt_K_per_s": dT_surfdt_val,
                        "convective_heat_loss_term_W": q_m_term,
                        "radiogenic_heating_term_W": q_rad_term,
                        "F_atmosphere_W_per_m2": self.f_atm,
                    }
                )

        # Extract results
        T_p_final     = sol.y[0, -1]
        T_surf_final  = sol.y[1, -1]
        r_s_final     = self.r_s(T_p_final)
        sim_time      = (self.curr_time + self.dt)/secs_per_year  # convert back to years
        q_m_final     = self.q_m(T_p_final, T_surf_final, self.melt_fraction(T_p_final))
        phi_final     = self.melt_fraction(T_p_final)
        f_radio_final = self.radioactive_heating(self.curr_time + self.dt)

        m_liquid = (4/3) * np.pi * self.bulk_density * (self.planet_radius**3 - r_s_final**3)
        m_solid  = self.mantle_mass - m_liquid

        if T_surf_final > T_p_final:
            T_surf_final = T_p_final - 1.0  # Ensure surface temperature does not exceed potential temperature

        output = {
            "T_magma": T_p_final,
            "T_pot": T_p_final,
            "T_surf": T_surf_final,
            "F_int": q_m_final,
            "Phi_global": phi_final,
            "Phi_global_vol": phi_final,
            "F_radio": f_radio_final/(4*np.pi*self.planet_radius**2),
            "RF_depth": r_s_final/self.planet_radius,
            "M_mantle_liquid": m_liquid,
            "M_mantle_solid": m_solid,
            "F_tidal": 0.0,
            "M_mantle": self.mantle_mass,
            "M_core": self.core_mass
        }

        return sim_time, output
