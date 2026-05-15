# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from proteus.utils.constants import const_G, secs_per_hour

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


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


def update_satellite(hf_row: dict, config: Config, dt: float):
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
    Rpl = hf_row['R_int']
    Mpl = hf_row['M_int']

    # Calculate bulk tidal power
    dE_tidal = hf_row['F_tidal'] * 4 * np.pi * Rpl**2  # Js-1

    # Calculate moment of inertia of planet (assuming solid sphere)
    I = 2 / 5 * Mpl * Rpl**2  # kg.m-1

    # Time step
    current_time = float(hf_row['Time'])

    # Use config parameters as initial guess
    if current_time <= 1:
        # Set satellite semimajor axis, satellite mass, and planet rotation frequency from config.
        hf_row['semimajorax_sat'] = float(config.orbit.semimajoraxis_sat)  # m
        hf_row['M_sat'] = float(config.orbit.mass_sat)  # kg

        Msa = hf_row['M_sat']

        if config.orbit.axial_period is None:
            # set by user to 'none', use 1:1 SOR
            hf_row['axial_period'] = float(hf_row['orbital_period'])
        else:
            hf_row['axial_period'] = float(config.orbit.axial_period) * secs_per_hour

        # Calculate the system angular-momentum integration constant
        # via the dedicated ``Ltot`` helper above, which implements
        # Korenaga (2023) Eq. 60 with the satellite-mass prefactor in
        # the orbital sqrt. Using the helper avoids duplicating the
        # formula and keeps any future revision in one place.
        sma = float(hf_row['semimajorax_sat'])
        omega = 2 * np.pi / float(hf_row['axial_period'])

        am_params = (I, None, const_G, Mpl, Msa, None)
        hf_row['plan_sat_am'] = Ltot(omega, sma, am_params)
        log.info('    sys.am = %.5f kg.m2.s-1' % (hf_row['plan_sat_am']))

        return
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Set semimajor axis, rotation frequency, satellite mass, and system AM from config.
    sma = float(hf_row['semimajorax_sat'])
    omega = 2 * np.pi / float(hf_row['axial_period'])
    Msa = hf_row['M_sat']

    # Could be allowed to vary to mimic resonance effects
    L = hf_row['plan_sat_am']

    # Collect system parameters at previous_time
    params = (I, L, const_G, Mpl, Msa, dE_tidal)

    # Find new satellite semimajor axis and axial frequency using RK5(4) integration method
    log.debug("Integrate satellite's sma and planet's omega with solve_ivp")
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, omega], args=(params,))

    # Update semimajor axis and axial period
    hf_row['semimajorax_sat'] = sol.y[0][-1]
    hf_row['axial_period'] = 2 * np.pi / sol.y[1][-1]
