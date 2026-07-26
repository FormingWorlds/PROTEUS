# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t, get_all_m_hansen
from proteus.utils.constants import const_G, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def evolve_orbit_star(hf_row: dict, config: Config, tides_o: Tides_t, interior_o: Interior_t):
    """Evolve the planet's orbital parameters.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : dict
            Dictionary of configuration options
        tides_o : Tides_t
            Tides object containing tidal interactions
        interior_o : Interior_t
            Interior object containing interior arrays
    """
    model = config.orbit.star_planet_model

    # Update orbit
    if model == 'sp0d':
        sp0d(hf_row, interior_o.dt)

    elif model == 'sp1d':
        # Compute planet principal moment of inertia (C_planet)
        from proteus.orbit.common import get_C_planet
        get_C_planet(hf_row, config, interior_o)

        # Call the sp1d function to evolve the satellite's orbital parameters
        sp1d(hf_row, tides_o, interior_o.dt)


def sp0d(hf_row: dict, dt: float):
    """Evolve the planet's orbital parameters module.

    Updates the semi-major axis and eccentricity.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        tides_o : Tides_t
            Tides object containing tidal interactions
        dt : float
            Time interval over which escape is occuring [yr]
    """

    def de_dt(a, e, params):
        """
        ODE describing evolution of orbital eccentricity based on Eq. 16 of
        Driscoll and Barnes (2015), Astrobiology 15, 739 (DOI 10.1089/ast.2015.1325).

        Sign convention note: in the paper, Im(k2) is negative for tidal
        dissipation (Eq. 4 expresses -Im(k2) as the positive dissipation
        efficiency). The current PROTEUS callers (dummy and lovepy backends)
        feed a positive Imk2, which under the formula below produces a
        positive de/dt and so EXPANDS the orbit rather than circularizing it.
        The paper convention would require Imk2 < 0 to obtain the physical
        circularization direction. Treat the sign as a known science item;
        do not invert it without first checking every Imk2 producer
        (proteus.orbit.dummy, proteus.orbit.lovepy, and any Imk2-dependent
        test) so the change propagates consistently.
        """
        Imk2, Mst, G, Rpl, Mpl = params
        return (21 / 2) * Imk2 * Mst**1.5 * G**0.5 * Rpl**5 / (Mpl * a**6.5) * e


    def da_dt(a, e, params):
        """
        ODE describing evolution of semimajor axis based on Eq. 15 of
        Driscoll and Barnes (2015), Astrobiology 15, 739.
        """
        return 2 * a * e * de_dt(a, e, params)


    def orbitals(t, z, params):
        """
        Helper function for solving coupled ODEs.
        """
        a, e = z
        return [da_dt(a, e, params), de_dt(a, e, params)]


    Imk2 = hf_row['Imk2']

    Rpl = hf_row['R_int']
    Mpl = hf_row['M_int']
    Mst = hf_row['M_star']

    sma = float(hf_row['semimajorax'])
    ecc = float(hf_row['eccentricity'])

    # Convert time to seconds
    dt = float(dt) * secs_per_year

    # Collect system parameters at previous_time
    params = (Imk2, Mst, const_G, Rpl, Mpl)

    # Find new semimajor axis and eccentricity using RK5(4) integration method
    log.debug('Integrate sma and ecc with solve_ivp')
    sol = solve_ivp(orbitals, [0, dt], [sma, ecc], args=(params,))

    # Update semimajor axis and eccentricity
    hf_row['semimajorax'] = sol.y[0][-1]
    hf_row['eccentricity'] = sol.y[1][-1]


def sp1d(hf_row, tides_o, dt):
    """Evolve the Planets's orbital parameters module.

    Updates the semi-major axis and primary rotation
    frequency based on angular momentum conservation.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        tides_o : Tides_t
            Tides object containing tidal interactions
        dt : float
            Time interval over which escape is occuring [yr]
    """

    # Convert time to seconds
    dt = float(dt) * secs_per_year

    # Orbital parameters from helpfile
    axial_p = 2 * np.pi / float(hf_row['axial_period'])
    sma = float(hf_row['semimajorax'])
    ecc = float(hf_row['eccentricity'])

    # Setup Initial State and Parameters
    y0 = [
        axial_p,
        sma,
        ecc
    ]

    params = {
        'M_p': hf_row['M_planet'], 'M_s': hf_row['M_star'],
        'R_p': hf_row['R_int'],    'R_s': hf_row['R_star'],
        'C_p': hf_row['C_planet']
    }

    # Retrieve tidal mode information from tides_o object
    nmk_p = tides_o.get(primary="planet", perturber="star").nmk
    LNk_p = tides_o.get(primary="planet", perturber="star").LNk

    # Convert arrays into a dictionary mapping (n, m, k) -> Complex Love Number
    love_dict_p = {tuple(nmk): ln for nmk, ln in zip(nmk_p, LNk_p)}

    kmin, kmax = np.min(nmk_p[:,2]), np.max(nmk_p[:,2])


    def domega_dt(I_j, C_j, sum_dOmega):
        """Planar secular tidal spin"""
        return -(3.0 * I_j / (2.0 * C_j)) * sum_dOmega


    def da_dt(a, E_p, sum_da_p):
        """Semimajor axis"""
        return a * (E_p / 2.0) * sum_da_p


    def de_dt(e, e_safe, E_p, sum_de_p):
        """Eccentricity"""
        sqrt_term = np.sqrt(1.0 - e**2)

        # Standard tidal dissipation (always active)
        de_tide_p = (E_p * sqrt_term / (4.0 * e_safe)) * sum_de_p

        return de_tide_p


    def smooth_sign(sigma, scale=1e-12):
        """Smooth approximation to sign(sigma) using tanh to avoid solver kinks."""
        return np.tanh(sigma / scale)


    def dE_dt(z, p):
        """Tidal energy dissipation rate"""

        Omega_p, Omega_s, a, e = z
        e_safe = max(e, 1e-12)

        # Basic Orbital and Physical Parameters
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)

        I_p = (const_G * p['M_s']**2 * p['R_p']**5) / a**6

        # Accumulators
        sums = {
            'dE_orb_p': 0.0,
            'dE_rot_p': 0.0
        }

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)  # 'n = 2' is fixed for this orbital evolution model

        for si, s in enumerate(k):

            # Fetch Hansen coefficients per mode 'm'
            X_0  = X_all[0][si]
            X_2  = X_all[2][si]

            # Fetch complex Love number components (A = Real, K = -Imaginary)
            # Look up the complex values, defaulting to 0.0 + 0j if not found
            val_p0 = love_dict_p.get((2, 0, s), 0.0 + 0.0j)
            val_p2 = love_dict_p.get((2, 2, s), 0.0 + 0.0j)

            # Extract real and imaginary parts
            K_p0 = -val_p0.imag
            K_p2 = -val_p2.imag

            # Accumulate
            X0_sq = X_0**2
            X2_sq = X_2**2

            sums['dE_orb_p'] += s * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)
            sums['dE_rot_p'] += K_p2 * X2_sq

        dE_orb_p = I_p * n_mm * sums['dE_orb_p'] / 4

        dE_rot_p = -I_p * 3 * Omega_p * sums['dE_rot_p'] / 2

        return -(dE_orb_p + dE_rot_p)


    def orbitals(t, z, p):
        Omega_p, Omega_s, a, e = z
        e_safe = max(e, 1e-12)

        # Basic Orbital and Physical Parameters
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)

        # Tidal scaling factors
        E_p = n_mm * (p['M_s'] / p['M_p']) * (p['R_p'] / a)**5
        I_p = (const_G * p['M_s']**2 * p['R_p']**5) / a**6

        # Accumulators
        sums = {
            'dOmega_p': 0.0,
            'da_p': 0.0,
            'de_p': 0.0
        }

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)  # 'n = 2' is fixed for this orbital evolution model

        # Retrieve Hansen/Love properties for this specific (a, e) state
        # Assuming tides_o and orbit_o handle the known caching efficiently
        for si, s in enumerate(k):

            # Fetch Hansen coefficients per mode 'm'
            X_0  = X_all[0][si]
            X_2  = X_all[2][si]

            # A small fraction of mean motion (e.g., 1e-4 * n_mm) or absolute threshold (1e-12) works well.
            sig_scale = max(1e-12, 1e-4 * n_mm)

            # Compute forcing frequencies
            sigma_0 = - s*n_mm
            sigma_p2 = 2*Omega_p - s*n_mm

            # Fetch complex Love number components (A = Real, K = -Imaginary)
            # Look up the complex values, defaulting to 0.0 + 0j if not found
            val_p0 = love_dict_p.get((2, 0, s), 0.0 + 0.0j)
            val_p2 = love_dict_p.get((2, 2, s), 0.0 + 0.0j)

            # Dynamically scale dissipation continuously through zero-crossing
            K_p0 = np.abs(val_p0.imag) * smooth_sign(sigma_0, sig_scale)
            K_p2 = np.abs(val_p2.imag) * smooth_sign(sigma_p2, sig_scale)

            # Accumulate
            X0_sq = X_0**2
            X2_sq = X_2**2
            sqrt_e = np.sqrt(1.0 - e_safe**2)

            sums['dOmega_p'] += K_p2 * X2_sq

            sums['da_p'] += s * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)

            sums['de_p'] += K_p0 * X0_sq * s * sqrt_e - 3.0 * K_p2 * X2_sq * (2.0 - s * sqrt_e)


        # Compute the eccentricity derivative using the same filter value
        de_dot = de_dt(
            e, e_safe,
            E_p,
            sums['de_p']
        )

        # Final Evaluation using distinct functions
        return [
            domega_dt(I_p, p['C_p'], sums['dOmega_p']),
            da_dt(a, E_p, sums['da_p']),
            de_dot
        ]


    # Integration
    log.debug("Integrating the ps1d_evec orbital model with solve_ivp")
    sol = solve_ivp(
        fun=lambda t, y: orbitals(t, y, params),
        t_span=(0, dt),
        y0=y0,
        method='Radau',
        rtol=1e-6,
        atol=1e-9
    )

    # Compute total angular momentum at the end of the integration
    L_final = params['C_p'] * sol.y[0][-1] + \
              (params['M_s'] * params['M_s']) / (params['M_p'] + params['M_s']) * \
              np.sqrt(const_G * (params['M_p'] + params['M_s']) * sol.y[2][-1] * \
              (1 - sol.y[3][-1]**2))

    # Compute total energy dissipated by tides over the time step
    dE_tide_p, _ = dE_dt(y0, params)

    # log energy per surface area for debugging
    energy_per_area = dE_tide_p / (4 * np.pi * params['R_p']**2)
    log.info(f"Total tidal power: {dE_tide_p:.3e} W, Energy per unit area: {energy_per_area:.3e} W/m^2")

    # Update semimajor axis and axial period
    hf_row['axial_period']     = 2 * np.pi / sol.y[0][-1]
    hf_row['semimajorax_sat']  = sol.y[1][-1]
    hf_row['eccentricity_sat'] = sol.y[2][-1]
    hf_row['plan_sat_am']      = L_final

    pass
