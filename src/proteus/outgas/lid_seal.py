"""Volatiles held in the melt under a closed crust, and their leak to the atmosphere.

With ``outgas.lid_seal.enabled``, each step after the initialisation stage tests
whether a crust seals the mantle melt from the atmosphere: the crust is at least
``delta_close`` thick and the top mesh cell holds less melt than ``phi_close``.
The thickness is the Aragog sub-grid lid ``lid_delta`` while that lid is active
(``lid_weight > 0``), and otherwise the depth of the top run of solid cells of
the mesh. When the seal closes, after the chemistry solve of that step, the
dissolved inventory moves from ``{x}_kg_liquid`` to ``{x}_kg_sealed``. While
sealed, the chemistry is frozen, escape draws on the atmosphere alone, and each
sealed species leaks to the atmosphere at

    dM_i/dt = -M_i / tau - F_volc,i,    F_volc,i = chi_d Mdot_c X_melt,i,
    X_melt,i = M_i / (M_melt + D_i M_solid),

with ``tau = tau_leak``, ``Mdot_c = crust_rate`` and ``chi_d`` the degassing
efficiency of Schaefer & Sasselov (2015, ApJ 801:40, Eq. 14), whose plate
spreading term ``rho_m F_melt D_melt S`` is replaced by the crust production
rate. The law is linear in ``M_i``; its solution over a step is exact. The seal
opens once the top cell holds ``phi_open`` or more melt, the crust is thinner
than ``delta_open``, or a giant impact remelts the mantle; the sealed mass then
returns to the melt and the chemistry solve of the same step takes it.

The leaked gas enters the atmosphere as the species it was sealed as, with no
chemistry solve: each species keeps the molar mass of its sealed moles, and the
partial pressures follow from the masses as in CALLIOPE (``P_surf = M_atm g / A``
with ``A = 4 pi R_int**2``, partial pressures by mole fraction).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.common import element_masses_from_species
from proteus.utils.constants import (
    gas_list,
    noble_gases,
    vol_element_list,
    vol_gas_list,
    vol_list,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

SEALED_SPECIES = (*vol_list, *noble_gases)
SEALED_ELEMENTS = tuple(vol_element_list)
LEAK_ELEMENTS = (*vol_element_list, *noble_gases)

# Relative round-off allowed in the ledger checks of one step.
_LEDGER_RTOL = 1e-12
# Mass fraction of each sealed element in each sealed species [1].
_WEIGHTS = {
    e: {s: element_masses_from_species({s: 1.0}).get(e, 0.0) for s in SEALED_SPECIES}
    for e in SEALED_ELEMENTS
}


def seal_enabled(config: Config) -> bool:
    """Whether the run uses the lid seal; ``False`` for a config built without it."""
    seal = getattr(config.outgas, 'lid_seal', None)
    return getattr(seal, 'enabled', False) is True


def crust_thickness(phi: np.ndarray, radius: np.ndarray, phi_solid: float) -> float:
    """Depth of the top run of solid mesh cells [m].

    Parameters
    ----------
    phi : numpy.ndarray
        Melt fraction of the mesh cells [1], core to surface.
    radius : numpy.ndarray
        Radii of the cell faces [m], core to surface, one more than ``phi``.
    phi_solid : float
        Melt fraction below which a cell counts as crust [1].

    Returns
    -------
    float
        Surface radius minus the radius of the lower face of the lowest cell in
        the top contiguous run of cells with ``phi < phi_solid``; 0 when the top
        cell is not solid.
    """
    phi = np.asarray(phi, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if radius.size != phi.size + 1 or phi.size == 0:
        raise ValueError(
            f'crust_thickness needs one more face than cells: {radius.size} faces, '
            f'{phi.size} cells'
        )
    if not np.all(np.diff(radius) > 0.0):
        raise ValueError('crust_thickness needs face radii that increase from core to surface')
    liquid = np.flatnonzero(~(phi < phi_solid))
    base = liquid[-1] + 1 if liquid.size else 0
    return float(radius[-1] - radius[base])


def closure_state(config: Config, hf_row: dict, interior_o) -> tuple[float, float]:
    """Crust thickness [m] and top-cell melt fraction [1] that the seal reads.

    The thickness is ``hf_row['lid_delta']`` while the Aragog sub-grid lid is
    active, else the mesh crust of :func:`crust_thickness`.

    Raises
    ------
    KeyError
        With an active sub-grid lid and no ``lid_delta`` in the row.
    ValueError
        If the thickness or the melt fraction is not finite.
    """
    phi_top = float(np.asarray(interior_o.phi, dtype=float)[-1])
    if float(hf_row.get('lid_weight', 0.0)) > 0.0:
        if 'lid_delta' not in hf_row:
            raise KeyError('The lid seal needs lid_delta [m] while the sub-grid lid is active')
        delta = float(hf_row['lid_delta'])
    else:
        delta = crust_thickness(
            interior_o.phi, interior_o.radius, config.outgas.lid_seal.phi_solid
        )
    if not (np.isfinite(delta) and np.isfinite(phi_top)):
        raise ValueError(
            f'Lid seal: crust thickness {delta} m or top-cell melt fraction {phi_top} '
            'is not finite'
        )
    return delta, phi_top


def _checked(value, label: str) -> float:
    """A reservoir mass that a seal move reads: finite and >= 0, else an error."""
    kg = float(value)
    if not np.isfinite(kg) or kg < 0.0:
        raise RuntimeError(f'Lid seal: {label} = {kg} cannot be moved')
    return kg


def close_seal(hf_row: dict) -> float:
    """Move the dissolved inventory into the sealed reservoir; return the species kg moved.

    Every column is checked before any mass moves, so a refused row is left as it was.
    """
    names = (*SEALED_SPECIES, *SEALED_ELEMENTS)
    kg = {n: _checked(hf_row.get(f'{n}_kg_liquid', 0.0), f'{n}_kg_liquid') for n in names}
    mol = {
        n: _checked(hf_row.get(f'{n}_mol_liquid', 0.0), f'{n}_mol_liquid')
        for n in SEALED_SPECIES
    }
    for n in names:
        hf_row[f'{n}_kg_sealed'] = float(hf_row.get(f'{n}_kg_sealed', 0.0)) + kg[n]
        hf_row[f'{n}_kg_liquid'] = 0.0
    for n in SEALED_SPECIES:
        hf_row[f'{n}_mol_sealed'] = float(hf_row.get(f'{n}_mol_sealed', 0.0)) + mol[n]
        hf_row[f'{n}_mol_liquid'] = 0.0
    return sum(kg[n] for n in SEALED_SPECIES)


def open_seal(hf_row: dict) -> float:
    """Return the sealed inventory to the melt; return the species kg moved.

    Every column is checked before any mass moves, so a refused row is left as it was.
    """
    names = (*SEALED_SPECIES, *SEALED_ELEMENTS)
    kg = {n: _checked(hf_row.get(f'{n}_kg_sealed', 0.0), f'{n}_kg_sealed') for n in names}
    mol = {
        n: _checked(hf_row.get(f'{n}_mol_sealed', 0.0), f'{n}_mol_sealed')
        for n in SEALED_SPECIES
    }
    for n in names:
        hf_row[f'{n}_kg_liquid'] = float(hf_row.get(f'{n}_kg_liquid', 0.0)) + kg[n]
        hf_row[f'{n}_kg_sealed'] = 0.0
    for n in SEALED_SPECIES:
        hf_row[f'{n}_mol_liquid'] = float(hf_row.get(f'{n}_mol_liquid', 0.0)) + mol[n]
        hf_row[f'{n}_mol_sealed'] = 0.0
    hf_row['seal_open_kg'] = sum(kg[n] for n in SEALED_SPECIES)
    return hf_row['seal_open_kg']


def leak_rates(config: Config, hf_row: dict, dt: float) -> tuple[dict, float, bool]:
    """Relaxation rate and volcanic rate per sealed species [yr-1].

    Parameters
    ----------
    config : Config
        Model configuration; reads ``outgas.lid_seal`` and ``outgas.D_const_*``.
    hf_row : dict
        Row with ``M_mantle`` [kg] and ``Phi_global`` [1].
    dt : float
        Step length [yr].

    Returns
    -------
    rates : dict
        ``{species: (1 / tau, k_i)}`` [yr-1].
    mdot : float
        Crust production rate applied over the step [kg yr-1].
    capped : bool
        Whether ``crust_rate dt`` exceeded the melt mass and was cut to it.
    """
    seal = config.outgas.lid_seal
    m_mantle = float(hf_row['M_mantle'])
    phi = min(max(float(hf_row['Phi_global']), 0.0), 1.0)
    m_melt, m_solid = phi * m_mantle, (1.0 - phi) * m_mantle
    mdot = float(seal.crust_rate) if phi >= seal.phi_solid else 0.0
    capped = mdot * dt > m_melt
    if capped:
        mdot = m_melt / dt
    inv_tau = 1.0 / float(seal.tau_leak)
    rates = {}
    for s in SEALED_SPECIES:
        reservoir = m_melt + float(getattr(config.outgas, f'D_const_{s}', 0.0)) * m_solid
        k = seal.chi_d * mdot / reservoir if mdot > 0.0 else 0.0
        rates[s] = (inv_tau, k)
    return rates, mdot, capped


def _refresh_atmosphere(hf_row: dict) -> None:
    """Partial pressures, mixing ratios and column masses from the species masses."""
    area = 4.0 * np.pi * float(hf_row['R_int']) ** 2
    m_atm = sum(float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in gas_list)
    mol = {s: float(hf_row.get(f'{s}_mol_atm', 0.0)) for s in gas_list}
    n_atm = sum(mol.values())
    p_surf = m_atm * float(hf_row['gravity']) / area * 1e-5
    for s in gas_list:
        vmr = mol[s] / n_atm if n_atm > 0.0 else 0.0
        hf_row[f'{s}_vmr'] = vmr
        hf_row[f'{s}_bar'] = vmr * p_surf
    hf_row['P_surf'] = p_surf
    hf_row['P_vol'] = p_surf
    hf_row['M_atm'] = m_atm
    hf_row['M_vol_atm'] = sum(float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in vol_gas_list)
    hf_row['atm_kg_per_mol'] = m_atm / n_atm if n_atm > 0.0 else 0.0


def _check(cond: bool, message: str) -> None:
    """Raise when a ledger invariant of the sealed step fails."""
    if not cond:
        raise RuntimeError(f'Lid seal ledger: {message}')


def leak_step(config: Config, hf_row: dict, dt: float) -> None:
    """Advance the sealed reservoir by ``dt`` [yr] with the exact step solution.

    Each species decays as ``M exp(-dt (1/tau + k_i))``; the leaked mass splits
    into its relaxation and volcanic parts in the ratio ``(1/tau) : k_i``. Each
    element of a species leaves in the same mass fraction of its sealed total as
    the species weights of that element give, so the element and species ledgers
    stay apart from the stoichiometry of the chemistry that filled them.
    """
    rates, mdot, capped = leak_rates(config, hf_row, dt)
    hf_row['seal_crust_rate'] = mdot
    hf_row['seal_volc_capped'] = float(capped)
    tau_part, volc_part, before = {}, {}, {}
    for s in SEALED_SPECIES:
        m = before[s] = float(hf_row.get(f'{s}_kg_sealed', 0.0))
        inv_tau, k = rates[s]
        total = inv_tau + k
        if m <= 0.0 or total <= 0.0:
            tau_part[s] = volc_part[s] = 0.0
            continue
        keep = np.exp(-dt * total)
        leaked = m * -np.expm1(-dt * total)
        tau_part[s], volc_part[s] = leaked * inv_tau / total, leaked * k / total
        mol = float(hf_row.get(f'{s}_mol_sealed', 0.0))
        hf_row[f'{s}_kg_sealed'] = m * keep
        hf_row[f'{s}_mol_sealed'] = mol * keep
        hf_row[f'{s}_kg_atm'] = float(hf_row.get(f'{s}_kg_atm', 0.0)) + leaked
        hf_row[f'{s}_mol_atm'] = float(hf_row.get(f'{s}_mol_atm', 0.0)) + mol * (1.0 - keep)

    leak_kg = 0.0
    for e, weights in _WEIGHTS.items():
        held = sum(w * before[s] for s, w in weights.items())
        sealed_e = float(hf_row.get(f'{e}_kg_sealed', 0.0)) / held if held > 0.0 else 0.0
        hf_row[f'{e}_kg_leak_tau'] = sealed_e * sum(w * tau_part[s] for s, w in weights.items())
        hf_row[f'{e}_kg_leak_volc'] = sealed_e * sum(
            w * volc_part[s] for s, w in weights.items()
        )
    for g in noble_gases:
        hf_row[f'{g}_kg_leak_tau'] = tau_part[g]
        hf_row[f'{g}_kg_leak_volc'] = volc_part[g]
    for e in SEALED_ELEMENTS:
        out = hf_row[f'{e}_kg_leak_tau'] + hf_row[f'{e}_kg_leak_volc']
        before = float(hf_row.get(f'{e}_kg_sealed', 0.0))
        _check(out <= before * (1.0 + _LEDGER_RTOL), f'{e} leaks {out} kg of {before} kg')
        hf_row[f'{e}_kg_sealed'] = max(before - out, 0.0)
        hf_row[f'{e}_kg_atm'] = float(hf_row.get(f'{e}_kg_atm', 0.0)) + out
        leak_kg += out
    leak_kg += sum(tau_part[g] + volc_part[g] for g in noble_gases)
    hf_row['seal_leak_kg_cum'] = float(hf_row.get('seal_leak_kg_cum', 0.0)) + leak_kg
    if leak_kg > 0.0:
        _refresh_atmosphere(hf_row)


def assert_sealed_row(hf_row: dict) -> None:
    """Invariants of a sealed row: no melt inventory, sealed masses finite and >= 0."""
    for name in (*SEALED_SPECIES, *SEALED_ELEMENTS):
        sealed = float(hf_row.get(f'{name}_kg_sealed', 0.0))
        _check(np.isfinite(sealed) and sealed >= 0.0, f'{name}_kg_sealed = {sealed} kg')
        liquid = float(hf_row.get(f'{name}_kg_liquid', 0.0))
        _check(liquid == 0.0, f'{name}_kg_liquid = {liquid} kg while sealed')


def sealed_mass(hf_row: dict, element: str) -> float:
    """Mass of one element [kg] held in the sealed reservoir, 0 when unreadable."""
    mass = float(hf_row.get(f'{element}_kg_sealed', 0.0))
    return mass if np.isfinite(mass) and mass > 0.0 else 0.0


def _reset_step_columns(hf_row: dict) -> None:
    """Zero the per-step seal columns before a step writes them."""
    hf_row['seal_open_kg'] = 0.0
    hf_row['seal_crust_rate'] = 0.0
    hf_row['seal_volc_capped'] = 0.0
    for e in LEAK_ELEMENTS:
        hf_row[f'{e}_kg_leak_tau'] = 0.0
        hf_row[f'{e}_kg_leak_volc'] = 0.0


def run_lid_seal_step(
    dirs: dict,
    config: Config,
    hf_row: dict,
    interior_o,
    sealed: bool,
    *,
    first_iter: bool,
    init_stage: bool,
    dt: float,
) -> bool:
    """Volatile exchange of one step with the lid seal on; return whether it is sealed.

    An open seal runs the chemistry solve and then closes when the crust allows
    it (never in the initialisation stage or on an impact step). A sealed step
    opens first when the band is crossed or an impact remelts the mantle;
    otherwise escape is applied to the atmosphere as on a crystallized mantle and
    the sealed reservoir leaks over ``dt`` [yr].
    """
    from proteus.outgas.wrapper import (
        check_ic_oxygen_budget,
        run_crystallized,
        run_outgassing_and_vapourisation,
    )

    seal = config.outgas.lid_seal
    _reset_step_columns(hf_row)
    delta, phi_top = closure_state(config, hf_row, interior_o)
    hf_row['seal_delta'], hf_row['seal_phi_top'] = delta, phi_top
    impact = bool(getattr(interior_o, 'impact_reset', False))
    if sealed and (impact or phi_top >= seal.phi_open_value or delta < seal.delta_open_value):
        log.info(
            'Lid seal opens (crust %.3e m, top-cell melt fraction %.3f, impact %s); '
            '%.3e kg returns to the melt',
            delta,
            phi_top,
            impact,
            open_seal(hf_row),
        )
        sealed = False
    if sealed:
        run_crystallized(config, hf_row, dt)
        leak_step(config, hf_row, dt)
        assert_sealed_row(hf_row)
    else:
        run_outgassing_and_vapourisation(dirs, config, hf_row, first_iter)
        check_ic_oxygen_budget(config, hf_row)
        if (
            not (init_stage or impact)
            and delta >= seal.delta_close
            and phi_top < seal.phi_close
        ):
            moved = close_seal(hf_row)
            hf_row['seal_t_close'] = float(hf_row['Time'])
            log.info(
                'Lid seal closes (crust %.3e m, top-cell melt fraction %.3f); %.3e kg sealed',
                delta,
                phi_top,
                moved,
            )
            sealed = True
    hf_row['seal_active'] = float(sealed)
    return sealed
