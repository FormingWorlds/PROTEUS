# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t, get_all_m_hansen
from proteus.utils.constants import const_G

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def evolve_orbit_satellite(hf_row: dict, config: Config, tides_o: Tides_t, interior_o: Interior_t):
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
    model = config.orbit.planet_satellite_model

    if model == 'ps0d':
        # Call the ps0d function to evolve the satellite's orbital parameters
        ps0d(hf_row, config, interior_o.dt)

    elif model == 'ps1d_evec':
        # Compute planet principal moment of inertia (C_planet)
        get_C_planet(hf_row, config, interior_o)

        # Call the ps1d_evec function to evolve the satellite's orbital parameters
        ps1d_evec(hf_row, config, tides_o, interior_o.dt)

    pass


def get_C_planet(hf_row: dict, config: Config, interior_o: Interior_t):
    """Compute the planet's principal moment of inertia (C_planet) based on the interior structure.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : Config
            Model configuration.
        interior_o : Interior_t
            Interior object containing interior arrays
    """
    # Calculate the planet's principal moment of inertia (C_planet)
    # Assuming a spherically symmetric mass distribution, we can use the formula:
    # C = (8/3) * pi * integral_0^R (rho(r) * r^4 dr)
    # where rho(r) is the density profile and R is the radius of the planet.

    # Get the radial grid and density profile from the interior object
    arr_keys = ("density", "radius")
    lov = {k:np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == "spider":
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    r_edges = lov["radius"]      # length N+1
    rho = lov["density"]         # length N

    r0 = r_edges[:-1]
    r1 = r_edges[1:]

    integral = np.sum(
        rho * (r1**5 - r0**5) / 5.0
    )

    C_planet = (8*np.pi/3.0) * integral

    # Store C_planet in the helpfile row for later use
    hf_row['C_planet'] = C_planet

    # Check if C_planet is physically reasonable
    C_factor_planet = C_planet / (hf_row['M_int'] * hf_row['R_int']**2)
    log.info(f"Computed C_planet: {C_planet:.3e} kg.m^2, C_factor_planet: {C_factor_planet:.3f}")


def ps0d(hf_row, config, dt):
    """Evolve the Satellite's orbital parameters module.

    Updates the semi-major axis and primary rotation
    frequency based on angular momentum conservation.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : dict
            Dictionary of configuration options
        dt : float
            Time interval over which escape is occuring [yr]
    """


    def Ltot(ω, a, params):
        """Total angular momentum of the planet plus satellite system.

        Implements Korenaga (2023) Icarus 400, 115564, Eq. 60:

            L = I_E * Omega + M_M * sqrt(G * (M_E + M_M) * a)        (Eq. 60)

        where I_E and Omega are the planet's moment of inertia and rotation
        frequency, M_M is the satellite mass, M_E is the planet mass, G is
        Newton's constant, and a is the planet-satellite semi-major axis.

        Derivation
        ----------
        The first term is the planet's spin angular momentum, I_E * Omega.

        The second term is the orbital angular momentum of the planet-
        satellite two-body problem. The textbook expression for a two-body
        orbital angular momentum about the system barycenter is

            L_orb = mu * v_rel * a                                   (textbook)

        with reduced mass mu = M_E * M_M / (M_E + M_M) and orbital speed
        v_rel = sqrt(G * (M_E + M_M) / a) (vis-viva at a circular orbit).
        Substituting,

            L_orb = mu * sqrt(G * (M_E + M_M) * a)

        Korenaga (2023) replaces mu by M_M, which is the limit of mu as
        M_M / M_E -> 0:

            mu = M_E M_M / (M_E + M_M) = M_M / (1 + M_M / M_E) -> M_M.

        For the Earth-Moon system the relative error of this substitution is
        M_M / M_E ~ 1/81 ~ 1.2%; for any heavier-satellite system the
        approximation would degrade, but PROTEUS's satellite module is
        currently targeted at the Earth-Moon regime, so we keep Korenaga's
        form verbatim.

        Sign convention: positive angular momentum corresponds to a prograde
        Moon (counter-clockwise from the planet's north pole). The integration
        constant L produced here is consumed by ``dω_dt`` and ``da_dt`` below,
        so any change to this formula MUST be paired with sanity checks on
        the time-evolution equations (Eqs. 58 + 59).
        """
        I, _, G, Mpl, Msa, _ = params
        # Korenaga (2023) Eq. 60: the orbital prefactor is the SATELLITE mass
        # M_M, which is the M_M << M_E limit of the textbook reduced-mass
        # formula. Substituting M_planet here inflates L by M_planet/M_sat
        # (~80x for Earth-Moon); see the reference-pinned test in
        # tests/orbit/test_satellite.py for the discriminating numeric guard.
        return I * ω + Msa * (G * (Mpl + Msa) * a) ** 0.5


    def dω_dt(a, ω, params):
        """Right-hand side of the planet-rotation ODE.

        Implements Korenaga (2023) Icarus 400, 115564, Eq. 58:

            dOmega/dt = -E_tide_dot / (I_E * Omega + G * M_E * M_M * I_E
                                        / (a * (L - I_E * Omega)))   (Eq. 58)

        where E_tide_dot is the tidal heat flux dissipated in the planet
        (positive, in W). The minus sign in front of E_tide_dot ensures the
        spin slows whenever tidal energy is being dissipated, matching the
        physical expectation that dissipation transfers angular momentum
        from the planet's spin to the satellite's orbit.

        The denominator is the partial derivative of the system's total
        energy with respect to Omega, evaluated at constant L (the
        integration constant set up by ``Ltot`` above). The bracketed second
        term is the orbital contribution; for the Earth-Moon system its
        magnitude is comparable to the spin term once the Moon recedes past
        a few Earth radii.

        See Korenaga (2023) Section 2.7 ("Orbital evolution") for the full
        derivation; the formulation closely follows Zahnle et al. (2015).
        """
        I, L, G, Mpl, Msa, dE_tidal = params
        return -dE_tidal / (I * ω + (G * Mpl * Msa * I) / (a * (L - I * ω)))


    def da_dt(a, ω, params):
        """Right-hand side of the satellite semi-major-axis ODE.

        Implements Korenaga (2023) Icarus 400, 115564, Eq. 59:

            da/dt = -2 * I_E * a / (L - I_E * Omega) * dOmega/dt      (Eq. 59)

        This is a direct consequence of differentiating the angular-momentum
        closure ``L = I_E * Omega + M_M * sqrt(G * (M_E + M_M) * a)`` (Eq. 60)
        with respect to time at constant L and solving for da/dt. Whenever the
        planet's spin slows (dOmega/dt < 0), the satellite's orbit expands
        (da/dt > 0) provided L > I_E * Omega, which is the prograde-Moon
        regime PROTEUS targets.
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

    # Calculate moment of inertia of planet (assuming solid sphere)
    I = 2 / 5 * Mpl * Rpl**2  # kg.m-1

    # Time step
    current_time = float(hf_row['Time'])

    # On the first run of this orbital module, instantiate the system angular-momentum
    if current_time <= 10 and L == 0:
        # Calculate the system angular-momentum integration constant
        # via the dedicated ``Ltot`` helper above, which implements
        # Korenaga (2023) Eq. 60 with the satellite-mass prefactor in
        # the orbital sqrt. Using the helper avoids duplicating the
        # formula and keeps any future revision in one place.
        L = Ltot(omega, sma, (hf_row['C_planet'], 0, const_G, Mpl, Msa, 0))
        hf_row['plan_sat_am'] = L

    # Find previous_time from which to evolve orbit to current_time
    previous_time = current_time - dt

    # Collect system parameters at previous_time
    params = (I, L, const_G, Mpl, Msa, dE_tidal)

    # Find new satellite semimajor axis and axial frequency using RK5(4) integration method
    log.debug("Integrating the ps0d orbital model with solve_ivp")
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, omega], args=(params,))

    # Update semimajor axis and axial period
    hf_row['semimajorax_sat'] = sol.y[0][-1]
    hf_row['axial_period'] = 2 * np.pi / sol.y[1][-1]

    pass


def ps1d_evec(hf_row, config, tides_o, dt):
    """Evolve the Satellite's orbital parameters module.

    Updates the semi-major axis and primary rotation
    frequency based on angular momentum conservation.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        config : dict
            Dictionary of configuration options
        tides_o : Tides_t
            Tides object containing tidal interactions
        dt : float
            Time interval over which escape is occuring [yr]
    """

    # Orbital parameters from helpfile
    axial_p = 2 * np.pi / float(hf_row['axial_period'])
    axial_s = 2 * np.pi / float(hf_row['axial_period_sat'])
    sma = float(hf_row['semimajorax_sat'])
    ecc = float(hf_row['eccentricity_sat'])
    aps_prec_angle = float(hf_row['aps_prec_angle'])

    # Setup Initial State and Parameters
    y0 = [
        axial_p,
        axial_s,
        sma,
        ecc,
        aps_prec_angle # evec_angle
    ]

    # Mean motion of star-planet system
    n_star = np.sqrt(const_G * (hf_row['M_star'] + hf_row['M_planet']) / hf_row['semimajorax']**3)

    params = {
        'M_p': hf_row['M_planet'], 'M_s': hf_row['M_sat'],
        'R_p': hf_row['R_int'],    'R_s': hf_row['R_sat'],
        'C_p': hf_row['C_planet'], 'C_s': hf_row['C_sat'],
        'n_star': n_star,
        'J_struc': 0.313          # <-- How is this computed? ~0.3 - 0.5 depending on structure
    }


    def domega_dt(I_j, C_j, sum_dOmega):
        """Planar secular tidal spin"""
        return -(3.0 * I_j / (2.0 * C_j)) * sum_dOmega


    def da_dt(a, E_p, E_s, sum_da_p, sum_da_s):
        """Semimajor axis"""
        return a * ((E_p / 2.0) * sum_da_p + (E_s / 2.0) * sum_da_s)


    def de_dt(e_safe, n_mm, n_star, phi, E_p, E_s, sum_de_p, sum_de_s):
        """Eccentricity (Tides + Solar Resonance)"""
        sqrt_term = np.sqrt(1.0 - e_safe**2)

        de_tide_p = (E_p * sqrt_term / (4.0 * e_safe)) * sum_de_p
        de_tide_s = (E_s * sqrt_term / (4.0 * e_safe)) * sum_de_s
        de_res    = (15.0 / 4.0) * e_safe * sqrt_term * (n_star**2 / n_mm) * np.sin(2.0 * phi)

        return de_tide_p + de_tide_s + de_res


    def dw_dt(e_safe, n_mm, n_star, phi, dw_J2, E_p, E_s, sum_dw_p, sum_dw_s):
        """Apsidal precession / Evection Angle"""
        prefactor = 1.0 / (e_safe**2 * np.sqrt(1.0 - e_safe**2))

        dw_tide_p = E_p * prefactor * sum_dw_p
        dw_tide_s = E_s * prefactor * sum_dw_s

        # Calculate d(phi)/dt
        dphi = dw_J2 + dw_tide_p + dw_tide_s - n_star + \
               (0.75) * (n_star**2 / n_mm) * np.sqrt(1.0 - e_safe**2) * (1.0 + 5.0 * np.cos(2.0 * phi))
        return dphi


    def orbitals(t, z, p):
        Omega_p, Omega_s, a, e, phi = z
        e_safe = max(e, 1e-12)

        # Basic Orbital and Physical Parameters
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)

        # Tidal scaling factors
        E_p = n_mm * (p['M_s'] / p['M_p']) * (p['R_p'] / a)**5
        I_p = (const_G * p['M_s']**2 * p['R_p']**5) / a**6

        E_s = n_mm * (p['M_p'] / p['M_s']) * (p['R_s'] / a)**5
        I_s = (const_G * p['M_p']**2 * p['R_s']**5) / a**6

        # J2 Apsidal Precession
        Omega_b = np.sqrt(const_G * p['M_p'] / p['R_p']**3)
        J2 = p['J_struc'] * (Omega_p / Omega_b)**2
        dw_J2 = 1.5 * J2 * n_mm * (p['R_p'] / a)**2 / (1.0 - e_safe**2)**2

        # Accumulators
        sums = {
            'dOmega_p': 0.0, 'dOmega_s': 0.0,
            'da_p': 0.0, 'da_s': 0.0,
            'de_p': 0.0, 'de_s': 0.0,
            'dw_p': 0.0, 'dw_s': 0.0
        }

        # Retrieve tidal mode information from tides_o object
        nmk_p = tides_o.get(primary="planet", perturber="satellite").nmk
        LNk_p = tides_o.get(primary="planet", perturber="satellite").LNk

        nmk_s = tides_o.get(primary="satellite", perturber="planet").nmk
        LNk_s = tides_o.get(primary="satellite", perturber="planet").LNk

        kmin, kmax = np.min(nmk_p[:,2]), np.max(nmk_p[:,2])
        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)

        # Convert your arrays into a dictionary mapping (n, m, k) -> Complex Love Number
        love_dict_p = {tuple(nmk): ln for nmk, ln in zip(nmk_p, LNk_p)}
        love_dict_s = {tuple(nmk): ln for nmk, ln in zip(nmk_s, LNk_s)}

        # Retrieve Hansen/Love properties for this specific (a, e) state
        # Assuming tides_o and orbit_o handle the known caching efficiently
        for si, s in enumerate(k):

            # Fetch Hansen modes
            X_0  = X_all[0][si]
            X_2  = X_all[2][si]
            X_m1 = X_all[-1][si]
            X_1  = X_all[1][si]
            X_m2 = X_all[-2][si]

            # Fetch complex Love number components (A = Real, K = -Imaginary)
            # Look up the complex values, defaulting to 0.0 + 0j if not found
            val_p0 = love_dict_p.get((2, 0, s), 0.0 + 0.0j)
            val_p2 = love_dict_p.get((2, 2, s), 0.0 + 0.0j)
            val_s0 = love_dict_s.get((2, 0, s), 0.0 + 0.0j)
            val_s2 = love_dict_s.get((2, 2, s), 0.0 + 0.0j)

            # Extract real and imaginary parts
            A_p0, K_p0 = val_p0.real, -val_p0.imag
            A_p2, K_p2 = val_p2.real, -val_p2.imag
            A_s0, K_s0 = val_s0.real, -val_s0.imag
            A_s2, K_s2 = val_s2.real, -val_s2.imag

            # Accumulate
            X0_sq = X_0**2
            X2_sq = X_2**2
            sqrt_e = np.sqrt(1.0 - e_safe**2)

            sums['dOmega_p'] += K_p2 * X2_sq
            sums['dOmega_s'] += K_s2 * X2_sq

            sums['da_p'] += s * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)
            sums['da_s'] += s * (K_s0 * X0_sq + 3.0 * K_s2 * X2_sq)

            sums['de_p'] += K_p0 * X0_sq * s * sqrt_e - 3.0 * K_p2 * X2_sq * (2.0 - s * sqrt_e)
            sums['de_s'] += K_s0 * X0_sq * s * sqrt_e - 3.0 * K_s2 * X2_sq * (2.0 - s * sqrt_e)

            term0 = 2.0*e_safe**2 * X0_sq + e_safe**2 * X_0 * (X_m2 + X_2) + 2.0*e_safe * X_0 * (X_m1 + X_1)
            term2 = (12.0*(2.0 - s*sqrt_e**3) - 9.0*e_safe**2) * X2_sq + 3.0*e_safe**2 * X_2 * X_m2 + \
                    (4.0*s*sqrt_e**3 - 6.0*e_safe**2) * X_0 * X_2 + 6.0*e_safe * X_2 * (X_m1 + X_1)

            sums['dw_p'] += (3.0/16.0) * A_p0 * term0 - (1.0/16.0) * A_p2 * term2
            sums['dw_s'] += (3.0/16.0) * A_s0 * term0 - (1.0/16.0) * A_s2 * term2

        # Final Evaluation using distinct functions
        return [
            domega_dt(I_p, p['C_p'], sums['dOmega_p']),
            domega_dt(I_s, p['C_s'], sums['dOmega_s']),
            da_dt(a, E_p, E_s, sums['da_p'], sums['da_s']),
            de_dt(e_safe, n_mm, p['n_star'], phi, E_p, E_s, sums['de_p'], sums['de_s']),
            dw_dt(e_safe, n_mm, p['n_star'], phi, dw_J2, E_p, E_s, sums['dw_p'], sums['dw_s'])
        ]

    # Integration
    log.debug("Integrating the ps1d_evec orbital model with solve_ivp")
    sol = solve_ivp(
        fun=lambda t, y: orbitals(t, y, params),
        t_span=(0, dt),
        y0=y0,
        method='Radau',
        rtol=1e-8,
        atol=1e-10
    )

    # Update semimajor axis and axial period
    hf_row['axial_period']     = 2 * np.pi / sol.y[0][-1]
    hf_row['axial_period_sat'] = 2 * np.pi / sol.y[1][-1]
    hf_row['semimajorax_sat']  = sol.y[2][-1]
    hf_row['eccentricity_sat'] = sol.y[3][-1]
    hf_row['aps_prec_angle']   = sol.y[4][-1]

    pass
