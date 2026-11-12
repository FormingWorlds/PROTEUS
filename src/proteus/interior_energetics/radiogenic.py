# Radiogenic heating inputs shared by the interior modules
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import RADIO_HEATING_MAX, radnuc_data

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def mass_ratio_from_atom_ratio(
    isotope: str, atom_ratio: float, element_molar_mass: float
) -> float:
    """Convert an isotope/element atom ratio to the mass ratio the config takes.

    ``X = R * m_iso / M_el``, with ``m_iso`` the isotope molar mass from
    ``radnuc_data``.

    Parameters
    ----------
    isotope : str
        Isotope name, a key of ``radnuc_data``.
    atom_ratio : float
        Atoms of the isotope per atom of the element [mol/mol].
    element_molar_mass : float
        Mean molar mass of the element at the same time [kg/mol].

    Returns
    -------
    float
        Isotope/element mass ratio [kg/kg].
    """
    return atom_ratio * radnuc_data[isotope]['mass'] / element_molar_mass


def radionuclide_inventory(config: Config) -> list[tuple[str, float, float]]:
    """List the heat-producing isotopes set in the config.

    Each entry is ``(isotope, element concentration [ppmw of mantle],
    isotope/element mass ratio [kg/kg])``, both at the stellar age
    ``interior_energetics.radio_tref``. The isotope name is a key of
    ``radnuc_data``.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.

    Returns
    -------
    list of (str, float, float)
        Isotope name, element concentration [ppmw], mass ratio [kg/kg].
    """
    ie = config.interior_energetics
    candidates = (
        ('al26', ie.radio_Al, ie.radio_Al26_abund),
        ('fe60', ie.radio_Fe, ie.radio_Fe60_abund),
        ('k40', ie.radio_K, ie.radio_K40_abund),
        ('th232', ie.radio_Th, ie.radio_Th232_abund),
        ('u235', ie.radio_U, ie.radio_U235_abund),
        ('u238', ie.radio_U, ie.radio_U238_abund),
    )
    return [
        (iso, float(cnc), float(abund))
        for iso, cnc, abund in candidates
        if float(cnc) * float(abund) > 0.0
    ]


def radio_t0_years(config: Config) -> float:
    """Model time at which the radionuclide inputs apply.

    Model time 0 is the stellar age ``star.age_ini``, so inputs given before
    the model start (``radio_tref < age_ini``) have a negative time.

    Parameters
    ----------
    config : Config
        PROTEUS configuration; reads ``interior_energetics.radio_tref`` and
        ``star.age_ini`` [Gyr].

    Returns
    -------
    float
        Model time of the radionuclide inputs [yr].
    """
    return (config.interior_energetics.radio_tref - config.star.age_ini) * 1e9


def radiogenic_heating(
    inventory: list[tuple[str, float, float]],
    t0_years: float,
    time_years: float | np.ndarray,
) -> float | np.ndarray:
    """Radiogenic heat production per unit mantle mass [W/kg].

    ``H(t) = sum_i C_i * 1e-6 * X_i * h_i * 2**((t0 - t) / t_half_i)``, with
    ``C_i`` the element concentration [ppmw], ``X_i`` the isotope mass ratio
    [kg/kg], ``h_i`` the isotope heat production [W/kg] and ``t_half_i`` the
    half-life [yr] from ``radnuc_data``.

    Parameters
    ----------
    inventory : list of (str, float, float)
        Output of ``radionuclide_inventory``.
    t0_years : float
        Model time at which the inventory applies [yr].
    time_years : float or ndarray
        Model time [yr].

    Returns
    -------
    float or ndarray
        Heat production [W/kg].
    """
    total = 0.0 * np.asarray(time_years, dtype=float)
    for iso, cnc, abund in inventory:
        data = radnuc_data[iso]
        arg = np.log(2.0) * (t0_years - np.asarray(time_years)) / data['halflife']
        total = total + cnc * 1e-6 * abund * data['heatprod'] * np.exp(arg)
    return total if np.ndim(total) else float(total)


def limit_radiogenic_heating(
    inventory: list[tuple[str, float, float]],
    t0_years: float,
    h_max: float = RADIO_HEATING_MAX,
) -> tuple[list[tuple[str, float, float]], float]:
    """Clip the heating at the model start to ``h_max``.

    Heating only decays after the model start (time 0), so its value there is
    the largest the run sees. When that value is above ``h_max`` [W/kg] or
    overflows, a warning is logged and the inputs are moved to model time 0
    and scaled so the heating starts at ``h_max`` and decays from there. The
    interior modules evaluate the decay law themselves, so the clip acts on
    their inputs. The sum is taken in log space to avoid the overflow.

    Parameters
    ----------
    inventory : list of (str, float, float)
        Output of ``radionuclide_inventory``.
    t0_years : float
        Model time at which the inventory applies [yr].
    h_max : float
        Largest heating allowed at the model start [W/kg].

    Returns
    -------
    inventory : list of (str, float, float)
        The input inventory, or the scaled one referred to model time 0.
    t0_years : float
        Model time of the returned inventory [yr].
    """
    if not inventory:
        return inventory, t0_years
    log_terms = np.array(
        [
            np.log(cnc * 1e-6 * abund * radnuc_data[iso]['heatprod'])
            + np.log(2.0) * t0_years / radnuc_data[iso]['halflife']
            for iso, cnc, abund in inventory
        ]
    )
    log_h0 = float(np.logaddexp.reduce(log_terms))
    if log_h0 <= np.log(h_max):
        return inventory, t0_years
    log.warning(
        'Radiogenic heating at the model start is unphysical: 10^%.1f W/kg'
        '\n    Check `interior_energetics.radio_tref` and isotope ratios.',
        log_h0 / np.log(10.0),
    )
    # Concentration at model time 0, scaled by h_max / H(0); X is kept, so C * X is right
    shift = np.log(h_max) - log_h0
    limited = [
        (
            iso,
            float(
                np.exp(
                    np.log(cnc) + np.log(2.0) * t0_years / radnuc_data[iso]['halflife'] + shift
                )
            ),
            abund,
        )
        for iso, cnc, abund in inventory
    ]
    return limited, 0.0


def radiogenic_inputs(config: Config) -> tuple[list[tuple[str, float, float]], float]:
    """The radionuclide inventory and its model time, clipped at the model start.

    The inventory of ``radionuclide_inventory`` and the time of
    ``radio_t0_years``, passed through ``limit_radiogenic_heating``. Aragog,
    SPIDER and the boundary interior all take their inputs from here.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.

    Returns
    -------
    inventory : list of (str, float, float)
        Isotope name, element concentration [ppmw], mass ratio [kg/kg].
    t0_years : float
        Model time at which the inventory applies [yr].
    """
    return limit_radiogenic_heating(radionuclide_inventory(config), radio_t0_years(config))
