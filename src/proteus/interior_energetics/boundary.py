from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from proteus.atmos_clim.common import Atmos_t
from proteus.interior_energetics.common import (
    Interior_t,
    d_solidification_radius_dT,
    eval_rheoparam,
    melt_fraction_linear,
    solidification_radius,
)
from proteus.interior_energetics.radiogenic import radiogenic_heating, radiogenic_inputs
from proteus.interior_energetics.resolved_mantle import (
    get_melt_curve,
    mantle_shells,
    publish_shells,
    tidal_power,
)
from proteus.interior_energetics.timestep import next_step
from proteus.interior_energetics.wrapper import get_core_heatcap
from proteus.utils.constants import (
    const_G,
    const_R,
    secs_per_year,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

ATM_MASS_MIN = 1.0  # min atmosphere mass
DELTA_MIN = 1e-3  # min CBL thickness


def atmosphere_specific_heat(config: Config, atmos_o: Atmos_t | None) -> tuple[float, str]:
    """
    Specific heat capacity of the atmosphere carried by the surface temperature.

    With ``interior_energetics.boundary.atm_heat_capacity_const`` this is the
    configured ``atm_heat_capacity``. Otherwise it is the layer-mass-weighted
    mean of the climate module's layer heat capacities at constant pressure,
    ``sum(cp_i sigma_i) / sum(sigma_i)``, from the column of the last
    atmosphere solve (AGNI's ``layer_cp`` [J kg-1 K-1] and ``layer_σ``
    [kg m-2]). Layers with a non-finite or non-positive value are skipped.
    The configured value stands in, and the returned source says why, when
    there is no column yet (no atmosphere object, or one without these
    arrays), when the two arrays differ in shape, when no layer is usable, or
    when both arrays still hold AGNI's allocation values (all ones), which a
    solve overwrites.

    Parameters
    ----------
    config : Config
        Configuration object.
    atmos_o : Atmos_t or None
        Atmosphere object whose ``_atm`` is the climate module's column.

    Returns
    -------
    cp : float
        Specific heat capacity [J kg-1 K-1].
    source : str
        'constant', 'atmosphere column', or 'configured value: <reason>'.
    """
    const = float(config.interior_energetics.boundary.atm_heat_capacity)
    if config.interior_energetics.boundary.atm_heat_capacity_const:
        return const, 'constant'

    column = getattr(atmos_o, '_atm', None)
    cp_layer = getattr(column, 'layer_cp', None)
    mass_layer = getattr(column, 'layer_σ', None)
    if cp_layer is None or mass_layer is None:
        return const, 'configured value: no atmosphere column yet'

    cp_arr = np.asarray(cp_layer, dtype=float)
    mass_arr = np.asarray(mass_layer, dtype=float)
    if cp_arr.ndim != 1 or cp_arr.shape != mass_arr.shape:
        return const, 'configured value: layer arrays differ in shape'

    valid = np.isfinite(cp_arr) & np.isfinite(mass_arr) & (cp_arr > 0.0) & (mass_arr > 0.0)
    if not np.any(valid):
        return const, 'configured value: no usable layer'
    if np.all(cp_arr == 1.0) and np.all(mass_arr == 1.0):
        return const, 'configured value: column not solved yet'

    return float(np.average(cp_arr[valid], weights=mass_arr[valid])), 'atmosphere column'


class BoundaryRunner:
    def __init__(
        self,
        config: Config,
        dirs: dict,
        hf_row: dict,
        hf_all: pd.DataFrame,
        interior_o: Interior_t,
        atmos_o: Atmos_t,
    ):
        self.curr_time = hf_row['Time'] * secs_per_year
        self.dt = (
            self.compute_time_step(config, dirs, hf_row, hf_all, interior_o) * secs_per_year
        )
        self.iteration = 1 if hf_all is None else len(hf_all)

        # The structure module writes R_int, R_core, M_int and M_core into hf_row.
        self.planet_radius = hf_row['R_int']
        self.planet_mass = float(hf_row['M_int'])
        self.core_mass = hf_row['M_core']
        self.m_atm = hf_row.get('M_atm', ATM_MASS_MIN)  # Atmospheric mass
        self.f_atm = hf_row['F_atm']

        # Specific heat of the atmosphere reservoir that T_surf carries [J kg-1 K-1]
        self.atmosphere_heat_capacity, cp_source = atmosphere_specific_heat(config, atmos_o)
        if not config.interior_energetics.boundary.atm_heat_capacity_const:
            log.info(
                'Atmosphere heat capacity for the surface reservoir: %.4g J kg-1 K-1 (%s)',
                self.atmosphere_heat_capacity,
                cp_source,
            )

        self.core_radius = float(hf_row['R_core'])
        self.core_frac = self.core_radius / self.planet_radius

        self.mantle_radius = self.planet_radius - self.core_radius
        self.mantle_volume = (4 / 3) * np.pi * (self.planet_radius**3 - self.core_radius**3)

        self.mantle_mass = self.planet_mass - self.core_mass
        if not self.mantle_mass > 0.0:
            raise ValueError(
                'The boundary interior needs M_int > M_core from the structure module; '
                f'got M_int = {self.planet_mass:.4e} kg and M_core = {self.core_mass:.4e} kg.'
            )
        self.mantle_bulk_density = self.mantle_mass / self.mantle_volume

        log.debug(
            f'Initialized BoundaryRunner with planet mass {self.planet_mass:.2e} kg, bulk density {self.mantle_bulk_density:.2f} kg/m^3, mantle mass {self.mantle_mass:.2e} kg'
        )
        self.surface_gravity = const_G * self.planet_mass / self.planet_radius**2

        self.rtol = config.interior_energetics.rtol
        self.atol = config.interior_energetics.atol

        if interior_o.ic == 2:
            self.T_p_0 = hf_row.get('T_magma')
            self.T_surf_0 = hf_row.get('T_surf')
        else:
            self.T_p_0 = config.planet.tsurf_init
            self.T_surf_0 = self.T_p_0

        self.T_solidus = config.interior_energetics.boundary.T_solidus
        self.T_liquidus = config.interior_energetics.boundary.T_liquidus

        # How the melt fraction follows from the potential temperature. The
        # adiabat model integrates T_p down the structure's mantle profile and
        # compares each level with the pressure-dependent melting curves, and
        # hands the tides modules the mantle as shells. The linear model treats
        # the mantle as one isothermal, uniform-density layer between the
        # constant solidus and liquidus above.
        self.melt_model = config.interior_energetics.boundary.melt_model
        if self.melt_model == 'adiabat':
            self.melt_curve = get_melt_curve(config, dirs, hf_row, interior_o)
            self.shells = mantle_shells(
                self.melt_curve.model.profile, interior_o.nlev_s, self.mantle_mass
            )
            shell_mass = self.shells.mass
        else:
            self.melt_curve = None
            self.shells = None
            shell_mass = np.array([self.mantle_mass])
        self.critical_melt_fraction = config.interior_energetics.rfront_loc
        self.Tsurf_event_change = config.interior_energetics.tmagma_atol

        # Choices the dummy backend makes differently (InteriorBoundary). The
        # core, when included, is lumped with the mantle and cools with T_p.
        self.sensible_heat = config.interior_energetics.boundary.sensible_heat
        self.tidal_viscosity = config.interior_energetics.boundary.tidal_viscosity
        self.viscosity_phi = config.interior_energetics.boundary.viscosity_phi
        if self.viscosity_phi == 'top':
            if self.melt_curve is None:
                raise ValueError('viscosity_phi = "top" needs melt_model = "adiabat"')
            # Melting curves at the structure's top level, the top of the mantle.
            self.T_solidus_top = float(self.melt_curve.model.tmp_sol[-1])
            self.T_liquidus_top = float(self.melt_curve.model.tmp_liq[-1])
        self.core_heat_capacity = (
            get_core_heatcap(config, hf_row) * float(self.core_mass)
            if config.interior_energetics.boundary.include_core_heat_capacity
            else 0.0
        )  # J/K

        # Material constants
        self.critical_rayleigh_number = (
            config.interior_energetics.boundary.critical_rayleigh_number
        )  # dimensionless
        self.heat_fusion_silicate = config.interior_energetics.latent_heat_of_fusion  # J/kg
        self.nusselt_exponent = (
            config.interior_energetics.boundary.nusselt_exponent
        )  # dimensionless
        self.silicate_heat_capacity = (
            config.interior_energetics.boundary.silicate_heat_capacity
        )  # J/kg/K
        self.thermal_conductivity = (
            config.interior_energetics.boundary.thermal_conductivity
        )  # W/m/K
        self.thermal_diffusivity = (
            config.interior_energetics.boundary.thermal_diffusivity
        )  # m^2/s
        self.thermal_expansivity = (
            config.interior_energetics.boundary.thermal_expansivity
        )  # 1/K
        self.const_R = const_R  # Gas constant [J/(mol·K)]

        # Viscosity parameterisation selection
        # 1 = constant viscosity
        # 2 = aggregate viscosity (smooth transition between solid and melt)
        # 3 = Arrhenius temperature-dependent viscosity
        self.viscosity_model = config.interior_energetics.boundary.viscosity_model

        # Constant viscosity model parameters
        self.eta_constant = 10**config.interior_energetics.const_log10visc  # Pa s

        # Aggregate viscosity parameters
        self.transition_width = (
            config.interior_energetics.phase_transition_width
        )  # dimensionless
        self.eta_solid_const = 10**config.interior_energetics.solid_log10visc  # Pa s
        self.eta_melt_const = 10**config.interior_energetics.melt_log10visc  # Pa s

        # Arrhenius solid mantle parameters
        self.dynamic_viscosity = config.interior_energetics.boundary.dynamic_viscosity  # Pa s
        self.activation_energy = config.interior_energetics.boundary.activation_energy  # J/mol
        self.creep_parameter = (
            config.interior_energetics.boundary.creep_parameter
        )  # dimensionless, for non-Arrhenius power-law creep (not currently used)

        # Arrhenius magma ocean parameters (Vogel-Fulcher-Tammann)
        self.viscosity_prefactor = (
            config.interior_energetics.boundary.viscosity_prefactor
        )  # Pa s
        self.viscosity_activation_temp = (
            config.interior_energetics.boundary.viscosity_activation_temp
        )  # K

        # Radioactive heating parameters
        self.use_radiogenic_heating = config.interior_energetics.heat_radiogenic
        # (isotope, element ppmw, isotope mass ratio) at model time radio_t0_yr [yr]
        self.radio_inventory, self.radio_t0_yr = (
            radiogenic_inputs(config) if self.use_radiogenic_heating else ([], 0.0)
        )

        # Tidal heating from the orbit module if enabled, otherwise zero. The
        # tides module returns a rate per shell, so the heat it deposits is
        # summed over the shells and expressed per unit mantle mass [W kg-1].
        self.use_tidal_heating = config.interior_energetics.heat_tidal
        self.tidal_term = (
            tidal_power(interior_o.tides, shell_mass) / self.mantle_mass
            if self.use_tidal_heating
            else 0.0
        )

        # Logging setup
        self.logging = config.interior_energetics.boundary.logging

    @staticmethod
    def compute_time_step(
        config: Config, dirs: dict, hf_row: dict, hf_all: pd.DataFrame, interior_o: Interior_t
    ) -> float:
        if interior_o.ic == 1:
            return 0.0
        else:
            step_sf = 1.0  # dt scale factor
            return next_step(config, dirs, hf_row, hf_all, step_sf, interior_o=interior_o)

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
            eta = self.dynamic_viscosity * np.exp(self.activation_energy / (self.const_R * T_p))

            eta = eta * np.exp(-self.creep_parameter * phi)  # Adjust for melt weakening

        else:
            # Magma ocean: use Vogel-Fulcher-Tammann relation adjusted for crystal fraction
            # Viscosity of liquid magma
            eta_l = self.viscosity_prefactor * np.exp(
                self.viscosity_activation_temp / (T_p - 1000.0)
            )

            # Adjust viscosity based on melt fraction
            eta = eta_l / (1.0 - (1.0 - phi) / (1.0 - self.critical_melt_fraction)) ** 2.5

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
            log.warning(
                f'Unknown viscosity model {self.viscosity_model}, defaulting to aggregate (2)'
            )
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
        Ra = (
            self.mantle_bulk_density
            * self.surface_gravity
            * self.thermal_expansivity
            * np.abs(T_p - T_surf)
            * self.mantle_radius**3
        ) / (eta * self.thermal_diffusivity)

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
        Nu = (Ra / self.critical_rayleigh_number) ** self.nusselt_exponent

        # Calculate convective heat flux
        q_m_val = Nu * self.thermal_conductivity * np.abs(T_p - T_surf) / self.mantle_radius

        return q_m_val

    def boundary_layer_thickness(self, T_p: float, T_surf: float, phi: float) -> float:
        """
        Calculate the thermal boundary layer thickness.

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
            Thermal boundary layer thickness [m]
        """

        # calculate heat flux
        q_m_val = self.q_m(T_p, T_surf, phi)
        q_m_val = max(q_m_val, 1e-6)

        # calculate boundary layer thickness using Fourier's law
        delta = self.thermal_conductivity * (T_p - T_surf) / q_m_val
        if delta < DELTA_MIN or np.isnan(delta) or np.isinf(delta):
            delta = DELTA_MIN

        return delta

    def radioactive_heating(self, t: float) -> float:
        """
        Calculate the radioactive heating rate per unit mantle mass at time t.

        Parameters
        ----------
        t : float
            Time [s]

        Returns
        -------
        float
            Radioactive heating rate per unit mantle mass [W/kg]
        """
        if not self.use_radiogenic_heating:
            return 0.0
        return radiogenic_heating(self.radio_inventory, self.radio_t0_yr, t / secs_per_year)

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
            Melt fraction [0-1], by mass under the adiabat model
        """
        if self.melt_curve is not None:
            return self.melt_curve.phi(T_p)
        return melt_fraction_linear(T_p, self.T_solidus, self.T_liquidus)

    def top_melt_fraction(self, T_p: float) -> float:
        """
        Melt fraction at the top of the mantle on the adiabat.

        The adiabat starts at T_p at the structure's top level, so this is the
        level's melt fraction at T_p, linear between its own solidus and
        liquidus as in `adiabat.melt_profile`.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]

        Returns
        -------
        float
            Melt fraction [0-1] of the topmost mantle level
        """
        span = self.T_liquidus_top - self.T_solidus_top
        if span <= 0.0:
            return 1.0 if T_p >= self.T_liquidus_top else 0.0
        return float(np.clip((T_p - self.T_solidus_top) / span, 0.0, 1.0))

    def convective_melt_fraction(self, T_p: float) -> float:
        """
        Melt fraction that sets the viscosity of the convective heat flux.

        The mass-averaged melt fraction with ``viscosity_phi = "global"``, the
        top-of-mantle one with ``"top"``.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]

        Returns
        -------
        float
            Melt fraction [0-1]
        """
        if self.viscosity_phi == 'top':
            return self.top_melt_fraction(T_p)
        return self.melt_fraction(T_p)

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
        return solidification_radius(
            T_p, self.T_solidus, self.T_liquidus, self.planet_radius, self.core_radius
        )

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

        return d_solidification_radius_dT(
            T_p, self.T_solidus, self.T_liquidus, self.planet_radius, self.core_radius
        )

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
        # Melt fraction that sets the convective viscosity
        phi = self.convective_melt_fraction(T_p)

        # Calculate convective heat flux
        q_m_val = self.q_m(T_p, T_surf, phi)

        # Calculate radioactive heating
        Q_val = self.radioactive_heating(t)

        # Energy balance numerator: heat loss - radiogenic heating
        numerator = -4 * np.pi * self.planet_radius**2 * q_m_val + self.mantle_mass * (
            Q_val + self.tidal_term
        )

        dT_pdt_val = numerator / self.heat_capacity(T_p)

        return dT_pdt_val

    def heat_capacity(self, T_p: float) -> float:
        """
        Effective heat capacity of the mantle, sensible plus latent, plus the core's.

        With ``sensible_heat = "molten"`` only the molten part of a partially
        molten mantle cools with the potential temperature; with ``"whole"``
        the whole mantle does. Either way the crystallisation front releases
        latent heat as it advances. A wholly molten mantle has no front to
        advance, and a wholly solid one cools as a whole, so neither draws on
        the latent budget. With ``include_core_heat_capacity`` the core's heat
        capacity is added throughout.

        Under the adiabat model the molten part is the mass melt fraction of
        the adiabat and the latent term is L M_mantle dphi/dT_p. With no
        adiabatic gradient, pressure-independent melting curves and a uniform
        density this reduces exactly to the linear model's solidification-front
        form.

        Parameters
        ----------
        T_p : float
            Potential temperature of the mantle [K]

        Returns
        -------
        float
            Heat capacity [J/K], positive
        """
        whole = self.silicate_heat_capacity * self.mantle_mass
        phi = self.melt_fraction(T_p)
        if not 0.0 < phi < 1.0:
            return whole + self.core_heat_capacity

        if self.melt_curve is not None:
            molten = self.silicate_heat_capacity * phi * self.mantle_mass
            latent = (
                self.heat_fusion_silicate * self.mantle_mass * self.melt_curve.dphi_dtmp(T_p)
            )
        else:
            r_s_val = self.r_s(T_p)  # Update r_s_val based on current T_p

            # Calculate dr_s/dT_p for the latent heat term
            dr_s_dT_p = self.drs_dTp(T_p)

            molten = (
                (4 / 3)
                * np.pi
                * self.mantle_bulk_density
                * self.silicate_heat_capacity
                * (self.planet_radius**3 - r_s_val**3)
            )
            latent = (
                -4
                * np.pi
                * r_s_val**2
                * self.mantle_bulk_density
                * self.heat_fusion_silicate
                * dr_s_dT_p
            )

        sensible = whole if self.sensible_heat == 'whole' else molten
        return sensible + latent + self.core_heat_capacity

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
        phi = self.convective_melt_fraction(T_p)
        q_m_val = self.q_m(T_p, T_surf, phi)

        # Calculate thickness of boundary layer
        delta = self.boundary_layer_thickness(T_p, T_surf, phi)

        numerator = 4 * np.pi * self.planet_radius**2 * (q_m_val - self.f_atm)
        denominator = self.atmosphere_heat_capacity * self.m_atm + (
            4 / 3
        ) * np.pi * self.silicate_heat_capacity * self.mantle_bulk_density * (
            self.planet_radius**3 - (self.planet_radius - delta) ** 3
        )

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
            csv_log_file = f'{output_dir}/boundary_solver_debug.csv'

            csv_columns = [
                'Time_years',
                'T_p_K',
                'T_surf_K',
                'q_m_val_W/m2',
                'viscosity_Pa_s',
                'rayleigh_number',
                'atm_heat_flux_W/m2',
                'atm_mass_kg',
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

        # time-step
        t_final = sol.t[-1]
        dt_actual = t_final - t_span[0]
        sim_time = t_final / secs_per_year  # convert back to years

        # Check reason for solver termination
        if sol.status == 1:
            log.warning(
                f'Tsurf event truncated timestep: dt={dt_actual / secs_per_year:.1e} years'
            )
        elif sol.status == -1:
            log.warning('Boundary layer interior failed to integrate - solver error')
            log.warning(f'    solver msg: {sol.message}')

        # Extract results
        T_p_final = sol.y[0, -1]
        T_surf_final = sol.y[1, -1]
        if self.melt_curve is not None:
            # The exact melt state at the end of the step, rather than the
            # interpolated curve the solver stepped on, supplies the reported
            # melt fractions, the front and the profiles handed on.
            state = self.melt_curve.model.state(T_p_final)
            phi_final = state.phi_mass
            phi_vol_final = state.phi_vol
            # Depth of the rheological front as a fraction of the planet
            # radius, the convention of the dummy, Aragog and SPIDER backends.
            rf_depth = 1.0 - state.r_liq / self.planet_radius
            m_liquid = phi_final * self.mantle_mass
        else:
            state = None
            phi_final = self.melt_fraction(T_p_final)
            phi_vol_final = phi_final
            rf_depth = phi_final * (1.0 - self.core_frac)
            r_s_fin = self.r_s(T_p_final)
            m_liquid = (
                (4 / 3)
                * np.pi
                * self.mantle_bulk_density
                * (self.planet_radius**3 - r_s_fin**3)
            )
        # The convective quantities use the melt fraction that sets their
        # viscosity, which is phi_final unless viscosity_phi = "top".
        phi_conv_final = self.convective_melt_fraction(T_p_final)
        if self.viscosity_phi == 'top':
            log.info(
                f'Top-of-mantle melt fraction for the convective viscosity: '
                f'{phi_conv_final:.4f} (mantle average {phi_final:.4f})'
            )
        visc_final = self.viscosity(T_p_final, T_surf_final, phi_conv_final)
        f_radio_final = self.radioactive_heating(t_final) * self.mantle_mass

        # Log final timestep values to CSV
        if self.logging:
            q_m_val = self.q_m(T_p_final, T_surf_final, phi_conv_final)
            ra_final = self.rayleigh_number(T_p_final, T_surf_final, phi_conv_final)

            with open(csv_log_file, mode='a', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=csv_columns)
                if csv_needs_header:
                    writer.writeheader()

                writer.writerow(
                    {
                        'Time_years': sim_time,
                        'T_p_K': T_p_final,
                        'T_surf_K': T_surf_final,
                        'q_m_val_W/m2': q_m_val,
                        'viscosity_Pa_s': visc_final,
                        'rayleigh_number': ra_final,
                        'atm_heat_flux_W/m2': self.f_atm,
                        'atm_mass_kg': self.m_atm,
                    }
                )

        m_solid = self.mantle_mass - m_liquid

        # Calculate boundary layer thickness
        delta = self.boundary_layer_thickness(T_p_final, T_surf_final, phi_conv_final)

        output = {
            'T_magma': T_p_final,
            'T_pot': T_p_final,
            'T_surf': T_surf_final,
            'F_int': self.f_atm,
            'Phi_global': phi_final,
            'Phi_global_vol': phi_vol_final,
            'F_radio': f_radio_final / (4 * np.pi * self.planet_radius**2),
            'RF_depth': rf_depth,
            'M_mantle_liquid': m_liquid,
            'M_mantle_solid': m_solid,
            'F_tidal': self.tidal_term * self.mantle_mass / (4 * np.pi * self.planet_radius**2)
            if self.use_tidal_heating
            else 0.0,
            'M_mantle': self.mantle_mass,
            'boundary_layer_thickness': delta,
            # Effective specific heat capacity, including the latent heat the
            # crystallisation front releases per kelvin of cooling.
            # The core's share, when included, is left out so Cp_eff stays the
            # mantle's own, as the dummy backend reports it.
            'Cp_eff': (self.heat_capacity(T_p_final) - self.core_heat_capacity)
            / self.mantle_mass,
        }

        # Store arrays
        if state is not None:
            self.publish_resolved_mantle(interior_o, state, T_surf_final)
        else:
            interior_o.phi = np.array([output['Phi_global']])
            interior_o.mass = np.array([output['M_mantle']])
            interior_o.visc = np.array([self.tidal_layer_viscosity(visc_final, phi_final)])
            interior_o.density = np.array([self.mantle_bulk_density])
            interior_o.temp = np.array([output['T_magma']])
            interior_o.pres = np.array([hf_row['P_surf']])
            interior_o.radius = np.array([self.core_radius, hf_row['R_int']])
            # The single isothermal layer is already the mantle's profile.
            interior_o.temp_profile = None

        return sim_time, output

    def tidal_layer_viscosity(self, own: float, phi: float) -> float:
        """
        Viscosity of one mantle layer as handed to the tides modules.

        Parameters
        ----------
        own : float
            This backend's own viscosity of the layer, from `viscosity_model` [Pa s]
        phi : float
            Melt fraction of the layer [0-1]

        Returns
        -------
        float
            `own` with ``tidal_viscosity = "boundary"``; with ``"kervazo"`` the
            Kervazo et al. (2021) melt-fraction rheology the dummy backend hands
            the tides modules, matching the shear and bulk moduli [Pa s]
        """
        if self.tidal_viscosity == 'kervazo':
            return float(eval_rheoparam(phi, 'visc'))
        return float(own)

    def publish_resolved_mantle(self, interior_o: Interior_t, state, T_surf: float):
        """
        Publish the adiabat-resolved mantle for the tides modules and the structure.

        The shells carry the mass-weighted melt fraction and temperature of the
        adiabat over them, and each takes the viscosity this backend's own
        rheology assigns to that melt fraction and temperature. The adiabat
        itself is handed to the structure re-solve on pressure, which a shell
        keeps when the re-solve moves its radius.

        Parameters
        ----------
        interior_o : Interior_t
            Interior model object receiving the layer arrays
        state : adiabat.MeltState
            Melt state of the mantle at the end of the step
        T_surf : float
            Surface temperature of the planet [K]
        """
        profile = self.melt_curve.model.profile
        publish_shells(interior_o, self.shells, profile, state)
        interior_o.visc = np.array(
            [
                self.tidal_layer_viscosity(self.viscosity(tmp, T_surf, phi), phi)
                for tmp, phi in zip(interior_o.temp, interior_o.phi)
            ],
            dtype=float,
        )
        interior_o.temp_profile = (
            np.array(profile.pressure, dtype=float),
            np.array(state.temperature, dtype=float),
        )
