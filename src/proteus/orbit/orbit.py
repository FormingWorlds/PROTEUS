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

    Angular momentum is deliberately NOT tracked or logged by this
    model: Driscoll and Barnes (2015), Astrobiology 15, 739, Eq. 14-16
    (the equations implemented below) evolve only (a, e), assuming no
    dissipation in the star and no rotation dynamics for the planet
    (the paper lists "variable rotation rates" as a future extension,
    not part of this model; "angular momentum" is not mentioned
    anywhere in the paper). There is consequently no citable
    conserved angular-momentum quantity for this model: orbital
    angular momentum alone (~sqrt(a(1-e^2))) is not conserved by the
    da/dt = 2*a*e*de/dt relation below (it decreases monotonically
    under the sign convention documented in de_dt's docstring), and
    constructing a "total" angular momentum under an assumed
    synchronous-rotation closure (orbital + I_planet*n(a)) does not
    fix this either: since a planet's spin angular momentum is orders
    of magnitude smaller than its orbital angular momentum, that total
    still drifts at essentially the same rate as the orbital term
    alone. Reaching a genuine angular-momentum-conserving model
    requires an explicit spin state and torque balance, which this
    model does not have (contrast sp1d, which does track planetary
    spin).

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

    This model is identical to ps1d, however here we
    assume no stellar tides, hence the governing
    equations are simplified to only include the
    planetary tides.

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
        ecc,
    ]

    params = {
        'M_p': hf_row['M_int'],
        'M_s': hf_row['M_star'],
        'R_p': hf_row['R_int'],
        'R_s': hf_row['R_star'],
        'C_p': hf_row['C_planet'],
    }

    # Retrieve tidal mode information from tides_o object
    nmk_p = tides_o.get(primary='planet', perturber='star').nmk
    LNk_p = tides_o.get(primary='planet', perturber='star').LNk

    kmin, kmax = int(np.min(nmk_p[:, 2])), int(np.max(nmk_p[:, 2]))
    n_k = kmax - kmin + 1

    def _dense_love(nmk, LNk, m_target):
        # Sparse-mode-safe (real tidal data need not have a row for every
        # integer s in [kmin, kmax]) AND folds in the m=0/s<0 modes that
        # Obliqua's own emission only supplies for s>=0. For a real,
        # causal linear response, k(-sigma) = k(sigma)* -- and for m=0,
        # sigma_0(-s) = -sigma_0(s) -- so the s<0 half is the complex
        # conjugate of the s>0 half, and (per Obliqua's own internal
        # heating-amplitude weighting, which doubles s>0 and keeps s=0
        # single for exactly this reason) contributes EQUALLY, not
        # negligibly. That doubling is applied inside Obliqua's own
        # P_T_blk/P_T_prf calculation but never reaches the exported
        # knms_total/nmk arrays this function consumes -- mirrored here
        # instead of at the source.
        mask = (nmk[:, 1] == m_target) & (nmk[:, 2] >= kmin) & (nmk[:, 2] <= kmax)
        dense = np.zeros(n_k, dtype=complex)
        dense[(nmk[mask, 2] - kmin).astype(int)] = LNk[mask]
        if m_target == 0:
            pos_mask = mask & (nmk[:, 2] > 0)
            s_pos = nmk[pos_mask, 2].astype(int)
            neg_idx = -s_pos - kmin
            valid = (neg_idx >= 0) & (neg_idx < n_k)
            dense[neg_idx[valid]] = np.conj(LNk[pos_mask][valid])
        return dense

    LNk_p_m0 = _dense_love(nmk_p, LNk_p, 0)
    LNk_p_m2 = _dense_love(nmk_p, LNk_p, 2)

    def domega_dt(I_j, C_j, sum_dOmega):
        """Planar secular tidal spin"""
        return -(3.0 * I_j / (2.0 * C_j)) * sum_dOmega

    def smooth_sign(sigma, scale=1e-12):
        """Smooth approximation to sign(sigma) using tanh to avoid solver kinks."""
        return np.tanh(sigma / scale)

    def dE_dt(z, p):
        """Tidal energy dissipation rate"""

        Omega_p, a, e = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        # Basic Orbital and Physical Parameters
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)
        s_arr = k.astype(float)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X0_sq = X_0**2
        X2_sq = X_2**2

        K_p0 = -LNk_p_m0.imag
        K_p2 = -LNk_p_m2.imag

        dE_orb_p = I_p * n_mm * np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq)) / 4

        dE_rot_p = -I_p * 3 * Omega_p * np.sum(K_p2 * X2_sq) / 2

        return -(dE_orb_p + dE_rot_p)

    def orbitals(t, z, p):
        Omega_p, a, e = z
        e_safe = min(
            max(e, 1e-12), 1.0 - 1e-9
        )  # symmetric: also guards e briefly exceeding 1 during a solver trial

        # Basic Orbital and Physical Parameters
        n_mm = np.sqrt(const_G * (p['M_p'] + p['M_s']) / a**3)

        # Tidal scaling factors
        E_p = n_mm * (p['M_s'] / p['M_p']) * (p['R_p'] / a) ** 5
        I_p = (const_G * p['M_s'] ** 2 * p['R_p'] ** 5) / a**6

        k, X_all = get_all_m_hansen(e_safe, 2, kmin, kmax)
        s_arr = k.astype(float)
        sig_scale = max(1e-12, 1e-4 * n_mm)
        sigma_0 = -s_arr * n_mm
        sigma_p2 = 2 * Omega_p - s_arr * n_mm

        K_p0 = np.abs(LNk_p_m0.imag) * smooth_sign(sigma_0, sig_scale)
        K_p2 = np.abs(LNk_p_m2.imag) * smooth_sign(sigma_p2, sig_scale)

        X_0 = X_all[0]
        X_2 = X_all[2]
        X0_sq = X_0**2
        X2_sq = X_2**2
        sqrt_e = np.sqrt(1.0 - e_safe**2)

        dOmega_p = np.sum(K_p2 * X2_sq)
        da_p = np.sum(s_arr * (K_p0 * X0_sq + 3.0 * K_p2 * X2_sq))
        de_p = np.sum(
            K_p0 * X0_sq * s_arr * sqrt_e - 3.0 * K_p2 * X2_sq * (2.0 - s_arr * sqrt_e)
        )

        sqrt_term = np.sqrt(1.0 - e_safe**2)
        da_dt_p = a * (E_p / 2.0) * da_p
        de_dt_p = (E_p * sqrt_term / (4.0 * e_safe)) * de_p

        return [
            domega_dt(I_p, p['C_p'], dOmega_p),
            da_dt_p,
            de_dt_p,
        ]

    # Integration
    log.debug('Integrating the sp1d orbital model with solve_ivp')
    sol = solve_ivp(
        fun=lambda t, y: orbitals(t, y, params),
        t_span=(0, dt),
        y0=y0,
        method='Radau',
        rtol=1e-6,
        atol=1e-9,
    )

    # Compute total angular momentum at the end of the integration
    L_final = params['C_p'] * sol.y[0][-1] + (params['M_s'] * params['M_s']) / (
        params['M_p'] + params['M_s']
    ) * np.sqrt(
        const_G * (params['M_p'] + params['M_s']) * sol.y[1][-1] * (1 - sol.y[2][-1] ** 2)
    )

    # Compute total energy dissipated by tides over the time step
    dE_tide_p = dE_dt(y0, params)

    # log energy per surface area for debugging
    energy_per_area = dE_tide_p / (4 * np.pi * params['R_p'] ** 2)
    log.debug(
        f'Total tidal power: {dE_tide_p:.3e} W, Energy per unit area: {energy_per_area:.3e} W/m^2'
    )

    # Exact, solver-consistent split of the changes accumulated over this step
    da_planet_tide = sol.y[1][-1] - sol.y[1][0]
    de_planet_tide = sol.y[2][-1] - sol.y[2][0]

    # Update semimajor axis and axial period
    hf_row['sma_dot_planet'] = da_planet_tide / dt  # m/s, planet-raised tide
    hf_row['ecc_dot_planet'] = de_planet_tide / dt  # 1/s

    # Update semimajor axis and axial period
    hf_row['axial_period'] = 2 * np.pi / sol.y[0][-1]
    hf_row['semimajorax'] = sol.y[1][-1]
    # Circularization (e -> 0) is a valid terminal state, but the ODE can
    # cross exactly zero and land on a floating-point-noise-scale negative
    # value.
    hf_row['eccentricity'] = max(sol.y[2][-1], 0.0)
    hf_row['plan_star_am'] = L_final

    pass
