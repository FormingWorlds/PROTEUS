# Radiogenic heating inputs shared by the interior modules
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import radnuc_data

if TYPE_CHECKING:
    from proteus.config import Config


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
    """Model time at which the radionuclide inputs apply [yr].

    Model time 0 is the stellar age ``star.age_ini``.
    """
    return (config.interior_energetics.radio_tref - config.star.age_ini) * 1e9


def radiogenic_heating(
    inventory: list[tuple[str, float, float]], t0_years: float, time_years
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
    time_years : float or array
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
