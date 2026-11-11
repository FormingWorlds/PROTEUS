"""Volatiles held in the melt under a closed crust, and their leak to the atmosphere.

With ``outgas.lid_seal.enabled``, each step after the initialisation stage tests
whether a crust seals the mantle melt from the atmosphere: the crust is at least
``delta_close`` thick and the top mesh cell holds less melt than ``phi_close``.
The thickness is the Aragog sub-grid lid ``lid_delta`` while that lid is active
(``lid_weight > 0``), and otherwise the depth of the top run of solid cells of
the mesh. When the seal closes, after the chemistry solve of that step, the
dissolved inventory moves from ``{x}_kg_liquid`` to ``{x}_kg_sealed`` and the
magma temperature of that solve is kept as ``seal_Tmagma_close``. While sealed,
the chemistry is frozen, escape draws on the atmosphere alone, and each sealed
species leaks to the atmosphere down to its dissolved equilibrium,

    dM_i/dt = -(1/tau + k_i) max(M_i - M_eq,i, 0),    k_i = chi_d Mdot_c / R_i,
    M_eq,i = X_eq,i(p) R_i,    R_i = M_melt + D_i M_solid,

with ``tau = tau_leak``, ``Mdot_c = crust_rate`` and ``chi_d`` the degassing
efficiency of Schaefer & Sasselov (2015, ApJ 801:40, Eq. 14), whose plate
spreading term ``rho_m F_melt D_melt S`` is replaced by the crust production
rate. The volcanic part is their ``chi_d Mdot_c X_melt,i`` with ``X_melt,i =
M_i / R_i``, limited here to the excess over the solubility, ``X_melt,i -
X_eq,i``, which their Eq. 14 does not have. ``X_eq,i`` is the CALLIOPE
solubility law of the species at its partial pressure in the atmosphere, at
the temperature and IW offset of the closing solve; a species without a law has
``M_eq = 0``. Gas does not dissolve back through the crust, so a species below
its floor stays as it is. Each step takes the exponential step of the law with
``M_eq`` at the end state, since the leak raises the partial pressure that sets
it; the step is exact for a constant ``M_eq`` and first order where it moves. The
seal opens once the top cell holds ``phi_open`` or more melt, the crust is
thinner than ``delta_open``, or a giant impact remelts the mantle; the sealed
mass then returns to the melt and the chemistry solve of the same step takes it.

The leaked gas enters the atmosphere as the species it was sealed as, with no
chemistry solve: each species keeps the molar mass of its sealed moles, and the
partial pressures follow from the masses as in CALLIOPE (``P_surf = M_atm g / A``
with ``A = 4 pi R_int**2``, partial pressures by mole fraction).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from calliope.solubility import (
    SolubilityCH4,
    SolubilityCO,
    SolubilityCO2,
    SolubilityH2O,
    SolubilityN2,
    SolubilityNobleGas,
    SolubilityS2,
)
from scipy.optimize import brentq

from proteus.outgas.common import element_masses_from_species, set_element_ratios
from proteus.utils.constants import (
    gas_list,
    noble_gases,
    soluble_gases,
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

# seal_path: the crust a closed seal reads; 0 while open.
PATH_OPEN, PATH_MESH, PATH_LID = 0, 1, 2
# Mass fraction of each sealed element in each sealed species [1].
_WEIGHTS = {
    e: {s: element_masses_from_species({s: 1.0}).get(e, 0.0) for s in SEALED_SPECIES}
    for e in SEALED_ELEMENTS
}
# The solubility laws of the CALLIOPE solve, with its arguments [ppmw].
_LAW = {
    'H2O': SolubilityH2O(),
    'CO2': SolubilityCO2(),
    'CO': SolubilityCO(),
    'CH4': SolubilityCH4(),
    'N2': SolubilityN2('dasgupta'),
    'S2': SolubilityS2(),
    **{g: SolubilityNobleGas(g) for g in noble_gases},
}
# S2 pressure [bar] below which the CALLIOPE S2 law returns 0.
S2_CUTOFF = 1.0e-20
# Leaks at which the end-state check samples the step residual of a species.
SCAN_POINTS = 17
# Gauss-Seidel sweeps over the species of a leak step: relative change and limit.
SWEEP_TOL, SWEEP_MAX = 1.0e-14, 100


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


def closure_state(
    config: Config, hf_row: dict, interior_o, path: int = 0
) -> tuple[float, float, int]:
    """Crust thickness [m], top-cell melt fraction [1] and the path that gives them.

    Path 2 is the Aragog sub-grid lid ``hf_row['lid_delta']``, read while the lid is
    active (``lid_weight > 0``); path 1 is the mesh crust of :func:`crust_thickness`.
    An open seal (``path = 0``) reads the lid when it is active and the mesh crust
    below ``phi_solid`` otherwise. A closed seal keeps the path that closed it: a
    lid-closed seal reads its lid while it is active, and every other closed seal
    reads the mesh crust below ``phi_open``. The mesh crust is 0 or whole cells, so
    that threshold is the band on the mesh.

    Raises
    ------
    KeyError
        With an active sub-grid lid and no ``lid_delta`` in the row.
    ValueError
        If ``lid_weight``, the thickness or the melt fraction is not finite.
    """
    seal = config.outgas.lid_seal
    phi_top = float(np.asarray(interior_o.phi, dtype=float)[-1])
    weight = float(hf_row.get('lid_weight', 0.0))
    if not np.isfinite(weight):
        raise ValueError(f'Lid seal: lid_weight = {weight} is not finite')
    lid = weight > 0.0
    if lid and path in (PATH_OPEN, PATH_LID):
        if 'lid_delta' not in hf_row:
            raise KeyError('The lid seal needs lid_delta [m] while the sub-grid lid is active')
        delta, used = float(hf_row['lid_delta']), 2
    else:
        phi_crust = seal.phi_solid if path == PATH_OPEN else seal.phi_open_value
        delta, used = crust_thickness(interior_o.phi, interior_o.radius, phi_crust), PATH_MESH
    if not (np.isfinite(delta) and np.isfinite(phi_top)):
        raise ValueError(
            f'Lid seal: crust thickness {delta} m or top-cell melt fraction {phi_top} '
            'is not finite'
        )
    return delta, phi_top, used


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

    Raises
    ------
    ValueError
        If ``M_mantle`` or ``Phi_global`` is not finite.
    """
    seal = config.outgas.lid_seal
    reservoir, phi, m_melt = reservoirs(config, hf_row)
    mdot = float(seal.crust_rate) if phi >= seal.phi_solid else 0.0
    capped = mdot * dt > m_melt
    if capped:
        mdot = m_melt / dt
    inv_tau = 1.0 / float(seal.tau_leak)
    rates = {}
    for s in SEALED_SPECIES:
        k = seal.chi_d * mdot / reservoir[s] if mdot > 0.0 else 0.0
        rates[s] = (inv_tau, k)
    return rates, mdot, capped


def reservoirs(config: Config, hf_row: dict) -> tuple[dict, float, float]:
    """Reservoir ``R_i = M_melt + D_i M_solid`` [kg] of each sealed species.

    Returns the reservoirs, the global melt fraction clipped to [0, 1] and the
    melt mass [kg]. Raises ``ValueError`` if ``M_mantle`` or ``Phi_global`` is
    not finite.
    """
    m_mantle, phi = float(hf_row['M_mantle']), float(hf_row['Phi_global'])
    for key, value in (('M_mantle', m_mantle), ('Phi_global', phi)):
        if not np.isfinite(value):
            raise ValueError(f'Lid seal: {key} = {value} is not finite')
    phi = min(max(phi, 0.0), 1.0)
    m_melt, m_solid = phi * m_mantle, (1.0 - phi) * m_mantle
    d = {s: float(getattr(config.outgas, f'D_const_{s}', 0.0)) for s in SEALED_SPECIES}
    return {s: m_melt + d[s] * m_solid for s in SEALED_SPECIES}, phi, m_melt


def _pressures(kg: dict, mol: dict, gravity: float, r_int: float) -> tuple[dict, dict, float]:
    """Mixing ratios [1], partial pressures [bar] and P_surf = M_atm g / A [bar], A = 4 pi R^2."""
    area = 4.0 * np.pi * r_int**2
    m_atm, n_atm = sum(kg.values()), sum(mol.values())
    p_surf = m_atm * gravity / area * 1e-5
    vmr = {s: n / n_atm if n_atm > 0.0 else 0.0 for s, n in mol.items()}
    return vmr, {s: x * p_surf for s, x in vmr.items()}, p_surf


def _atmosphere(hf_row: dict) -> tuple[dict, dict]:
    """Species masses [kg] and moles [mol] of the atmosphere of a row."""
    kg = {s: float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in gas_list}
    return kg, {s: float(hf_row.get(f'{s}_mol_atm', 0.0)) for s in gas_list}


def _refresh_atmosphere(hf_row: dict) -> None:
    """Partial pressures, mixing ratios, element ratios and column masses from the masses."""
    kg, mol = _atmosphere(hf_row)
    vmr, bar, p_surf = _pressures(kg, mol, float(hf_row['gravity']), float(hf_row['R_int']))
    m_atm = sum(kg.values())
    n_atm = sum(mol.values())
    for s in gas_list:
        hf_row[f'{s}_vmr'], hf_row[f'{s}_bar'] = vmr[s], bar[s]
    hf_row['P_surf'] = p_surf
    hf_row['P_vol'] = p_surf
    hf_row['M_atm'] = m_atm
    hf_row['M_vol_atm'] = sum(float(hf_row.get(f'{s}_kg_atm', 0.0)) for s in vol_gas_list)
    hf_row['atm_kg_per_mol'] = m_atm / n_atm if n_atm > 0.0 else 0.0
    set_element_ratios(hf_row)


def _check(cond: bool, message: str) -> None:
    """Raise when a ledger invariant of the sealed step fails."""
    if not cond:
        raise RuntimeError(f'Lid seal ledger: {message}')


def equilibrium_fractions(
    config: Config, bar: dict, p_tot: float, temp: float, fo2_shift: float
) -> dict:
    """Dissolved mass fraction ``X_eq,i`` [1] of each species of ``soluble_gases``.

    The laws and arguments of the CALLIOPE solve: H2O (Sossi et al. 2023), CO2
    (Dixon et al. 1995) at ``temp`` [K], CO (Armstrong et al. 2015) and CH4
    (Ardia et al. 2013) at the total pressure ``p_tot`` [bar], N2 (Dasgupta et
    al. 2022) and S2 (Gaillard et al. 2022) at ``temp`` and the IW offset
    ``fo2_shift`` [log10], and the noble gases by Henry's law (Jambon et al.
    1986), each at its partial pressure in ``bar`` [bar]. CO, CH4 and the noble
    gases dissolve only where ``outgas.calliope.include_*`` is set, and every
    fraction is 0 with ``outgas.calliope.solubility`` off. Below
    ``S2_CUTOFF``, where the CALLIOPE S2 law returns 0, the S2 fraction follows
    the same square-root law in its partial pressure, so the floor stays continuous.
    """
    calliope = config.outgas.calliope
    x = dict.fromkeys(soluble_gases, 0.0)
    if not calliope.solubility:
        return x
    args = {'CO2': (temp,), 'CO': (p_tot,), 'CH4': (p_tot,), 'N2': (p_tot, temp, fo2_shift)}
    args['S2'] = (temp, fo2_shift)
    for s in soluble_gases:  # every law is 0 at p = 0, where the CO law warns
        p = bar[s]
        if calliope.is_included(s) and p > 0.0:
            q = max(p, S2_CUTOFF) if s == 'S2' else p
            x[s] = 1e-6 * float(_LAW[s](q, *args.get(s, ()))) * np.sqrt(p / q)
    return x


def floor_leak(mass: float, frac: float, floor_of) -> float:
    """Leak [kg] of one sealed species over a step toward the floor of its end state.

    Solves ``L = frac max(mass - floor_of(L), 0)`` with ``frac = 1 - exp(-lam dt)``,
    the exact step of ``dM/dt = -lam max(M - M_eq, 0)`` for a constant ``M_eq``,
    with ``M_eq = floor_of(L)`` [kg] taken at the state after the leak ``L``. The
    residual is <= 0 at ``L = 0`` and >= 0 at ``L = frac mass``, so the root is
    bracketed for any continuous floor. Raises ``ValueError`` on a floor that is
    negative or not finite.
    """
    floor = floor_of(0.0)
    if not (np.isfinite(floor) and floor >= 0.0):
        raise ValueError(
            f'Lid seal: dissolved equilibrium {floor} kg is negative or not finite'
        )
    if floor >= mass:
        return 0.0
    return brentq(
        lambda leak: leak - frac * max(mass - floor_of(leak), 0.0),
        0.0,
        frac * mass,
        xtol=1e-300,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=200,
    )


def _floor_model(config: Config, hf_row: dict, per_kg: dict):
    """``M_eq`` [kg] of a species after leaks ``{species: kg}`` into the row's atmosphere.

    Moles enter at ``per_kg`` [mol kg-1]; T and the IW offset are those of the
    closing solve. With ``outgas.trap_mode = 'front'`` the crystals that froze
    before closure keep their own trapped volatiles, so each reservoir counts only
    the crystals formed after it, ``D_i max(Phi_c - Phi, 0) M_mantle`` with
    ``Phi_c = seal_phi_close``. Raises ``ValueError`` if a closing value it reads
    is unreadable.
    """
    temp = float(hf_row.get('seal_Tmagma_close', np.nan))
    shift = float(hf_row.get('fO2_shift_IW_derived', np.nan))
    if not (np.isfinite(temp) and temp > 0.0 and np.isfinite(shift)):
        raise ValueError(
            f'Lid seal: seal_Tmagma_close = {temp} K or fO2_shift_IW_derived = {shift} '
            'cannot set the dissolved equilibrium'
        )
    reservoir, phi, _ = reservoirs(config, hf_row)
    if config.outgas.trap_mode == 'front':
        phi_c = float(hf_row.get('seal_phi_close', np.nan))
        if not 0.0 < phi_c <= 1.0:
            raise ValueError(
                f'Lid seal: seal_phi_close = {phi_c} is not the melt fraction of a closure'
            )
        frozen = (1.0 - max(phi_c, phi)) * float(
            hf_row['M_mantle']
        )  # crystals older than closure
        d = {s: float(getattr(config.outgas, f'D_const_{s}', 0.0)) for s in reservoir}
        reservoir = {s: r - d[s] * frozen for s, r in reservoir.items()}
    kg0, mol0 = _atmosphere(hf_row)
    gravity, r_int = float(hf_row['gravity']), float(hf_row['R_int'])

    def floor_of(leak: dict, s: str) -> float:
        kg = {g: m + leak.get(g, 0.0) for g, m in kg0.items()}
        mol = {g: n + leak.get(g, 0.0) * per_kg.get(g, 0.0) for g, n in mol0.items()}
        _, bar, p_tot = _pressures(kg, mol, gravity, r_int)
        return equilibrium_fractions(config, bar, p_tot, temp, shift)[s] * reservoir[s]

    return floor_of


def _leaks(config: Config, hf_row: dict, before: dict, frac: dict) -> dict:
    """Leak [kg] over the step of each species in ``frac``, at the floors of the end state.

    A species without a law leaks ``frac M``; the others solve :func:`floor_leak`
    in Gauss-Seidel sweeps, since each floor depends on the whole atmosphere.
    """
    soluble = [s for s in frac if s in soluble_gases and config.outgas.calliope.solubility]
    leak = {s: 0.0 if s in soluble else frac[s] * before[s] for s in frac}
    if not soluble:
        return leak
    per_kg = {s: float(hf_row.get(f'{s}_mol_sealed', 0.0)) / before[s] for s in frac}
    floor_of = _floor_model(config, hf_row, per_kg)
    for _ in range(SWEEP_MAX):
        change = 0.0
        for s in soluble:
            new = floor_leak(before[s], frac[s], lambda x, s=s: floor_of({**leak, s: x}, s))
            change, leak[s] = max(change, abs(new - leak[s]) / before[s]), new
        if change <= SWEEP_TOL:
            return leak
    raise RuntimeError(f'Lid seal: leak sweeps did not settle within {SWEEP_MAX} sweeps')


def leak_step(config: Config, hf_row: dict, dt: float) -> None:
    """Advance the sealed reservoir by ``dt`` [yr] toward its dissolved equilibrium.

    Each species follows ``dM/dt = -(1/tau + k_i) max(M - M_eq, 0)`` in the
    exponential step with ``M_eq`` taken at the end state (:func:`floor_leak`).
    The leaked mass splits into its relaxation and volcanic parts in the ratio
    ``(1/tau) : k_i``. Each element of a species leaves in the same mass fraction
    of its sealed total as the species weights of that element give, so the
    element and species ledgers stay apart from the stoichiometry of the
    chemistry that filled them.
    """
    rates, mdot, capped = leak_rates(config, hf_row, dt)
    hf_row['seal_crust_rate'] = mdot
    hf_row['seal_volc_capped'] = float(capped)
    before = {s: float(hf_row.get(f'{s}_kg_sealed', 0.0)) for s in SEALED_SPECIES}
    lam = {s: sum(rates[s]) for s in SEALED_SPECIES}
    frac = {
        s: -np.expm1(-dt * lam[s]) for s in SEALED_SPECIES if before[s] > 0.0 and lam[s] > 0.0
    }
    leak = _leaks(config, hf_row, before, frac) if frac else {}
    tau_part, volc_part = dict.fromkeys(SEALED_SPECIES, 0.0), dict.fromkeys(SEALED_SPECIES, 0.0)
    for s, out in leak.items():
        if out <= 0.0:
            continue
        tau_part[s], volc_part[s] = out * rates[s][0] / lam[s], out * rates[s][1] / lam[s]
        mol = float(hf_row.get(f'{s}_mol_sealed', 0.0))
        moved = mol * (out / before[s])
        hf_row[f'{s}_kg_sealed'] = before[s] - out
        hf_row[f'{s}_mol_sealed'] = mol - moved
        hf_row[f'{s}_kg_atm'] = float(hf_row.get(f'{s}_kg_atm', 0.0)) + out
        hf_row[f'{s}_mol_atm'] = float(hf_row.get(f'{s}_mol_atm', 0.0)) + moved

    leak_kg = sum(tau_part[g] + volc_part[g] for g in noble_gases)
    for g in noble_gases:
        hf_row[f'{g}_kg_leak_tau'], hf_row[f'{g}_kg_leak_volc'] = tau_part[g], volc_part[g]
    for e, weights in _WEIGHTS.items():
        sealed_e = float(hf_row.get(f'{e}_kg_sealed', 0.0))
        held = sum(w * before[s] for s, w in weights.items())
        share = sealed_e / held if held > 0.0 else 0.0
        by_tau = share * sum(w * tau_part[s] for s, w in weights.items())
        by_volc = share * sum(w * volc_part[s] for s, w in weights.items())
        out = by_tau + by_volc
        hf_row[f'{e}_kg_leak_tau'], hf_row[f'{e}_kg_leak_volc'] = by_tau, by_volc
        hf_row[f'{e}_kg_sealed'] = max(sealed_e - out, 0.0)
        hf_row[f'{e}_kg_atm'] = float(hf_row.get(f'{e}_kg_atm', 0.0)) + out
        leak_kg += out
    hf_row['seal_leak_kg_cum'] = float(hf_row.get('seal_leak_kg_cum', 0.0)) + leak_kg
    if leak_kg > 0.0:
        _refresh_atmosphere(hf_row)
    if frac and config.outgas.calliope.solubility:
        _check_floors(config, hf_row, before, leak, frac)


def _check_floors(config: Config, hf_row: dict, before: dict, leak: dict, frac: dict) -> None:
    """Write ``{s}_kg_eq`` of the end state and check each leaked species against it.

    A species that leaked ends at or above its floor, and its distance to the
    floor is ``exp(-lam dt)`` times the distance it started at; both to round-off
    of its held mass before the step (1e-12 and 1e-9 of it). Its step residual,
    sampled at ``SCAN_POINTS`` leaks over the bracket with the others at their
    final leaks, changes sign at most once at those samples; roots closer than
    one sample spacing, and coupled solutions of several species, pass. A law
    that falls with pressure (CO2, CO and CH4 at high pressure) gives a second
    solution only where the floor falls faster than the leak fills the atmosphere.
    """
    per_kg = {s: hf_row[f'{s}_mol_atm'] / hf_row[f'{s}_kg_atm'] for s in leak if leak[s] > 0.0}
    floor_of = _floor_model(config, hf_row, per_kg)
    for s in soluble_gases:
        m_eq = hf_row[f'{s}_kg_eq'] = floor_of({}, s)
        if leak.get(s, 0.0) <= 0.0:
            continue
        end = float(hf_row[f'{s}_kg_sealed'])
        _check(
            end >= m_eq - 1e-12 * before[s],
            f'{s}_kg_sealed = {end} kg below its floor {m_eq} kg',
        )
        gap = (1.0 - frac[s]) * (before[s] - m_eq)
        _check(
            abs(end - m_eq - gap) <= 1e-9 * before[s],
            f'{s} ends {end - m_eq} kg above its floor, not {gap} kg',
        )
        grid = np.linspace(0.0, frac[s] * before[s], SCAN_POINTS)
        res = [x - frac[s] * max(before[s] - floor_of({s: x - leak[s]}, s), 0.0) for x in grid]
        _check(
            np.count_nonzero(np.diff(np.greater(res, 0.0))) <= 1,
            f'the step of {s} has more than one solution',
        )


def assert_sealed_row(hf_row: dict) -> None:
    """Invariants of a sealed row: no melt inventory, sealed masses finite and >= 0."""
    for name in (*SEALED_SPECIES, *SEALED_ELEMENTS):
        sealed = float(hf_row.get(f'{name}_kg_sealed', 0.0))
        _check(np.isfinite(sealed) and sealed >= 0.0, f'{name}_kg_sealed = {sealed} kg')
        liquid = float(hf_row.get(f'{name}_kg_liquid', 0.0))
        _check(liquid == 0.0, f'{name}_kg_liquid = {liquid} kg while sealed')


def restore_seal_state(config: Config, hf_row: dict) -> bool:
    """Whether a resumed run starts sealed, from ``seal_active`` of its last row.

    With the seal off and nothing held, a stale flag and path in the row are zeroed.

    Raises
    ------
    ValueError
        If the flag, the path or a sealed mass is not readable, if a sealed row
        has no closing temperature ``seal_Tmagma_close`` > 0 K or, under front
        trapping, no closing melt fraction ``seal_phi_close`` in (0, 1], if the row holds
        sealed mass under an open flag, or if it holds sealed mass with the seal
        off, since nothing could open it then.
    """
    on = seal_enabled(config)
    names = (*SEALED_SPECIES, *SEALED_ELEMENTS)
    values = [('seal_active', float(hf_row.get('seal_active', 0.0)))]
    values += [('seal_path', float(hf_row.get('seal_path', PATH_OPEN)))]
    values += [(f'{n}_kg_sealed', float(hf_row.get(f'{n}_kg_sealed', 0.0))) for n in names]
    bad = [k for k, v in values if not (np.isfinite(v) and v >= 0.0)]
    if values[1][1] not in (PATH_OPEN, PATH_MESH, PATH_LID):
        bad.append('seal_path')
    if bad:
        raise ValueError(f'Resume: unreadable seal state in the last row: {", ".join(bad[:4])}')
    held = [k for k, v in values[2:] if v > 0.0]
    if held and not on:
        raise ValueError(
            f'Resume: the last row holds sealed mass ({", ".join(held[:4])}) but '
            'outgas.lid_seal.enabled = false; resume with the seal on.'
        )
    if held and values[0][1] < 0.5:
        raise ValueError(
            f'Resume: seal_active is 0 but the last row holds sealed mass ({held[0]})'
        )
    if not on:
        hf_row['seal_active'] = hf_row['seal_path'] = 0.0
    sealed = on and values[0][1] >= 0.5
    temp = float(hf_row.get('seal_Tmagma_close', np.nan))
    if sealed and not (np.isfinite(temp) and temp > 0.0):
        raise ValueError(f'Resume: a sealed last row needs seal_Tmagma_close > 0 K, not {temp}')
    phi_c = float(hf_row.get('seal_phi_close', np.nan))
    if sealed and config.outgas.trap_mode == 'front' and not 0.0 < phi_c <= 1.0:
        raise ValueError(
            f'Resume: a sealed last row needs seal_phi_close in (0, 1], not {phi_c}'
        )
    return sealed


def sealed_mass(hf_row: dict, element: str) -> float:
    """Mass of one element [kg] held in the sealed reservoir, 0 when unreadable."""
    mass = float(hf_row.get(f'{element}_kg_sealed', 0.0))
    return mass if np.isfinite(mass) and mass > 0.0 else 0.0


def clear_seal(hf_row: dict) -> None:
    """Empty the sealed reservoir and zero every seal column, as a desiccation does."""
    for n in (*SEALED_SPECIES, *SEALED_ELEMENTS):
        hf_row[f'{n}_kg_sealed'] = 0.0
    for n in SEALED_SPECIES:
        hf_row[f'{n}_mol_sealed'] = 0.0
    reset_step_columns(hf_row)
    for k in ('seal_active', 'seal_path', 'seal_delta', 'seal_phi_top'):
        hf_row[k] = 0.0


def reset_step_columns(hf_row: dict) -> None:
    """Zero the per-step seal columns before a step writes them, and after a desiccation."""
    hf_row['seal_open_kg'] = 0.0
    hf_row['seal_crust_rate'] = 0.0
    hf_row['seal_volc_capped'] = 0.0
    for e in LEAK_ELEMENTS:
        hf_row[f'{e}_kg_leak_tau'] = 0.0
        hf_row[f'{e}_kg_leak_volc'] = 0.0
    for s in soluble_gases:
        hf_row[f'{s}_kg_eq'] = 0.0


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
    from proteus.outgas.trapping import remelt_returned_mass
    from proteus.outgas.wrapper import (
        check_ic_oxygen_budget,
        run_crystallized,
        run_outgassing_and_vapourisation,
    )

    seal = config.outgas.lid_seal
    reset_step_columns(hf_row)
    path = int(hf_row.get('seal_path', PATH_OPEN)) if sealed else PATH_OPEN
    if sealed and path == 0:  # a sealed row without its path reads the source it has now
        path = PATH_LID if float(hf_row.get('lid_weight', 0.0)) > 0.0 else PATH_MESH
    delta, phi_top, used = closure_state(config, hf_row, interior_o, path)
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
        sealed, used = False, PATH_OPEN
    if sealed:
        if remelt_returned_mass(hf_row):
            close_seal(hf_row)  # a release this step is under the crust too
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
            hf_row['seal_Tmagma_close'] = max(float(hf_row['T_magma']), config.outgas.T_floor)
            hf_row['seal_phi_close'] = float(hf_row['Phi_global'])
            log.info(
                'Lid seal closes (crust %.3e m, top-cell melt fraction %.3f); %.3e kg sealed',
                delta,
                phi_top,
                moved,
            )
            sealed = True
        else:
            used = PATH_OPEN
    hf_row['seal_active'], hf_row['seal_path'] = float(sealed), float(used)
    return sealed
