# Orbit evolution module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scipy.integrate import solve_ivp

from proteus.orbit.common import Tides_t
from proteus.utils.constants import AU, const_G

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def evolve_orbit_star(hf_row: dict, config: Config, tides_o: Tides_t, dt: float):
    """Evolve the planet's orbital parameters.

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

    model = config.orbit.star_planet_model

    # Update orbit
    if model == 'sp0d':
        sp0d(hf_row, config, dt)

    # elif model == 'sp1d':
        # sp1d(hf_row, config, tides_o, orbit_o, dt)


def sp0d(hf_row: dict, config: Config, dt: float):
    """Evolve the planet's orbital parameters module.

    Updates the semi-major axis and eccentricity.

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

    # Time step
    current_time = float(hf_row['Time'])

    # Use config parameters as initial guess
    if current_time <= 1:
        # Set semimajor axis and eccentricity from config.
        hf_row['semimajorax'] = config.orbit.semimajoraxis * AU
        hf_row['eccentricity'] = config.orbit.eccentricity
        return
    else:
        # Find previous_time from which to evolve orbit to current_time
        previous_time = current_time - dt

    # Collect system parameters at previous_time
    params = (Imk2, Mst, const_G, Rpl, Mpl)

    # Find new semimajor axis and eccentricity using RK5(4) integration method
    log.debug('Integrate sma and ecc with solve_ivp')
    sol = solve_ivp(orbitals, [previous_time, current_time], [sma, ecc], args=(params,))

    # Update semimajor axis and eccentricity
    hf_row['semimajorax'] = sol.y[0][-1]
    hf_row['eccentricity'] = sol.y[1][-1]
