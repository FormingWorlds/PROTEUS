"""Atmodeller outgassing module wrapper for PROTEUS.

Wraps the atmodeller package (Bower+2025, ApJ 995:59) to compute
volatile partitioning between atmosphere and magma ocean using
thermodynamically consistent equilibrium chemistry with real gas
EOS and non-ideal solubility laws.

Replaces CALLIOPE as an alternative outgassing module when
``config.outgas.module = 'atmodeller'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import vol_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def _total_volatile_oxygen_kg(hf_row: dict, element_mmw: dict) -> float:
    """Total volatile oxygen [kg]: atmospheric plus melt-dissolved O.

    Sums the oxygen carried by the O-bearing volatile species
    ``{H2O, CO2, CO, SO2, O2}`` across both the atmospheric (``_kg_atm``)
    and dissolved (``_kg_liquid``) reservoirs. This is the whole-planet O
    inventory that ``M_ele`` must include so that ``M_atm <= M_planet``
    holds even when O sits in an oxidising O2 atmosphere.

    Atomic-O mass fractions are derived from ``element_mmw`` (rather than
    hand-rounded values) so a correction to the atomic-mass table
    propagates automatically.

    Parameters
    ----------
    hf_row : dict
        Helpfile row carrying per-species ``{sp}_kg_atm`` / ``{sp}_kg_liquid``.
    element_mmw : dict
        Molar masses [kg/mol] keyed by element symbol (``'O'``, ``'C'``, ...).

    Returns
    -------
    float
        Total volatile oxygen mass [kg]. Zero when no O-bearing species are
        present.
    """
    m_O = element_mmw['O']
    m_C = element_mmw['C']
    m_H = element_mmw['H']
    m_S = element_mmw['S']
    o_mass_frac = {
        'H2O': m_O / (2 * m_H + m_O),
        'CO2': 2 * m_O / (m_C + 2 * m_O),
        'CO': m_O / (m_C + m_O),
        'SO2': 2 * m_O / (m_S + 2 * m_O),
        'O2': 1.0,
    }
    total = 0.0
    for sp, frac in o_mass_frac.items():
        total += float(hf_row.get(f'{sp}_kg_atm', 0.0)) * frac
        total += float(hf_row.get(f'{sp}_kg_liquid', 0.0)) * frac
    return total


# Atom counts per outgassing species, used to split the per-species reservoir
# masses into per-element reservoir masses. Covers the volatile species the
# atmodeller solver partitions (constants.vol_list); the SiO vapour species is
# excluded because atmodeller does not outgas it, and tracking Si only on the
# atmosphere side would make M_ele backend-dependent. Keyed-to-vol_list is
# asserted at import so a new species cannot silently leak its element mass.
_VOLATILE_ELEMENT_STOICH = {
    'H2O': {'H': 2, 'O': 1},
    'CO2': {'C': 1, 'O': 2},
    'O2': {'O': 2},
    'H2': {'H': 2},
    'CH4': {'C': 1, 'H': 4},
    'CO': {'C': 1, 'O': 1},
    'N2': {'N': 2},
    'NH3': {'N': 1, 'H': 3},
    'S2': {'S': 2},
    'SO2': {'S': 1, 'O': 2},
    'H2S': {'H': 2, 'S': 1},
}

assert set(_VOLATILE_ELEMENT_STOICH) == set(vol_list), (
    'atmodeller _VOLATILE_ELEMENT_STOICH must cover exactly constants.vol_list; '
    f'missing={set(vol_list) - set(_VOLATILE_ELEMENT_STOICH)} '
    f'extra={set(_VOLATILE_ELEMENT_STOICH) - set(vol_list)}'
)

# Cache of atmodeller EquilibriumModel instances. atmodeller compiles its JIT
# solver lazily on the first solve and caches it on the model as ``_solver``,
# intending the model to be reused across solves. Constructing a fresh model
# every outgassing step discards that cache, so the solver is recompiled on
# every solve; the per-solve LLVM compilation memory then accumulates until the
# job is killed (out-of-memory) after a few dozen steps. Reusing one model per
# signature compiles the solver once and makes every later solve with the same
# signature a cache hit.
#
# The key is everything the compiled solver's structure depends on: the species
# network (tuple of species names), the fugacity source, and the solver mode.
# Active-vs-inactive elements within a fixed network are handled by atmodeller
# as traced (NaN-masked) values, not as shape, so they do not need to be in the
# key; the species network shrinks in lockstep when an element drops below the
# mass threshold, which changes the key. fugacity source and solver mode are
# fixed per run, so within one run the key reduces to the species network; they
# are included so a future in-process path that varied them could not reuse a
# solver compiled for the other structure. The cached model also carries the
# last solve's ``_output``; the wrapper reads it immediately after each solve in
# the same call, so the shared instance is safe under that discipline. The cache
# holds at most one entry per distinct active species set (a handful over a run
# as minor elements cross the mass threshold).
_MODEL_CACHE: dict = {}


def _cached_model(signature: tuple, build):
    """Return the cached model for ``signature``, building it once on a miss.

    Parameters
    ----------
    signature : tuple
        Hashable identity of the species network (the tuple of species names).
    build : callable
        Zero-argument factory that constructs the model; called only on a cache
        miss so the JIT solver compiles once per distinct species network.

    Returns
    -------
    object
        The cached model instance for this signature.
    """
    model = _MODEL_CACHE.get(signature)
    if model is None:
        model = build()
        _MODEL_CACHE[signature] = model
    return model


def _populate_volatile_element_reservoirs(hf_row: dict, element_mmw: dict) -> None:
    """Derive per-element reservoir masses from the per-species inventory.

    atmodeller writes per-species masses (``{sp}_kg_atm`` / ``_kg_liquid`` /
    ``_kg_solid``) but not the per-element reservoir masses CALLIOPE writes
    natively. The escape step reads ``{e}_kg_atm`` under the default
    ``reservoir = "outgas"`` to size the per-element fractionation; with that
    field absent (zero) it sees an empty reservoir, ``calc_new_elements``
    returns a zeroed inventory, the per-element ``{e}_kg_total`` budget
    collapses, and atmodeller is then skipped for want of a mass constraint so
    the atmosphere freezes.

    This writes only ``{e}_kg_atm`` / ``_kg_liquid`` / ``_kg_solid`` (the
    reservoir split), matching CALLIOPE and ``outgas.common.expected_keys``,
    which reserve the per-element ``{e}_kg_total`` slot for escape (the running
    budget owner) and, for oxygen, the authoritative ``O_kg_total`` write. The
    ``{e}_kg_total`` slot is deliberately NOT written here, so escape's debited
    total is never overwritten by a species reconstruction.

    For each element the mass in a reservoir is the sum over species of the
    species mass in that reservoir times the element's mass fraction in the
    species (atom count times atomic mass, divided by the species molar mass).

    Parameters
    ----------
    hf_row : dict
        Helpfile row carrying per-species ``{sp}_kg_<reservoir>``; updated in
        place with per-element ``{e}_kg_<reservoir>``.
    element_mmw : dict
        Molar masses [kg/mol] keyed by element symbol.
    """
    reservoirs = ('atm', 'liquid', 'solid')
    elements = {el for comp in _VOLATILE_ELEMENT_STOICH.values() for el in comp}
    masses = {el: dict.fromkeys(reservoirs, 0.0) for el in elements}
    for sp, comp in _VOLATILE_ELEMENT_STOICH.items():
        sp_mmw = sum(n * element_mmw[el] for el, n in comp.items())
        if sp_mmw <= 0.0:
            continue
        for r in reservoirs:
            sp_mass = float(hf_row.get(f'{sp}_kg_{r}', 0.0))
            # `not (... > 0)` rejects NaN as well as zero/negative masses.
            if not (sp_mass > 0.0):
                continue
            for el, n in comp.items():
                masses[el][r] += sp_mass * (n * element_mmw[el] / sp_mmw)
    for el, res in masses.items():
        hf_row[f'{el}_kg_atm'] = res['atm']
        hf_row[f'{el}_kg_liquid'] = res['liquid']
        hf_row[f'{el}_kg_solid'] = res['solid']


def _species_output_scalar(output_dict: dict, species_key: str, field: str):
    """Return a finite scalar ``field`` from atmodeller's per-species output.

    atmodeller's ``output.asdict()`` maps each species key (for example
    ``'H2O_g'``) to a dict of JAX/NumPy arrays. This squeezes the requested
    field to a Python float, returning ``None`` when the species, the field, or
    a valid finite numeric conversion is absent, so the caller can fall back
    explicitly rather than propagate a NaN or a stale value.

    Parameters
    ----------
    output_dict : dict
        atmodeller ``output.asdict()`` result.
    species_key : str
        Per-species key including the phase suffix (``'H2O_g'``).
    field : str
        Field name within the species dict (``'gas_mass'``, ``'dissolved_mass'``).

    Returns
    -------
    float or None
        The squeezed finite scalar, or ``None`` when unavailable.
    """
    species_data = output_dict.get(species_key)
    if not isinstance(species_data, dict):
        return None
    val = species_data.get(field)
    if val is None:
        return None
    try:
        scalar = float(np.squeeze(val))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def calc_surface_pressures_atmodeller(dirs: dict, config: Config, hf_row: dict):
    """Compute volatile partitioning using atmodeller.

    Solves for chemical equilibrium between the magma ocean and
    atmosphere, accounting for real gas EOS, solubility laws, and
    optionally condensation. Updates hf_row with partial pressures,
    VMRs, dissolved masses, and total surface pressure.

    Parameters
    ----------
    dirs : dict
        Directory paths.
    config : Config
        PROTEUS configuration object.
    hf_row : dict
        Current helpfile row (modified in place).
    """
    from atmodeller import (
        ChemicalSpecies,
        EquilibriumModel,
        Planet,
        SpeciesNetwork,
    )
    from atmodeller.solubility import get_solubility_models
    from atmodeller.thermodata import IronWustiteBuffer

    from proteus.utils.constants import M_earth, element_mmw, gas_list, noble_gases

    atm_config = config.outgas.atmodeller
    solubility_models = get_solubility_models()

    # Real gas EOS models (None = ideal gas)
    from atmodeller.eos import get_eos_models

    eos_models = get_eos_models()
    _eos_map = {
        'H2O': atm_config.eos_H2O,
        'CO2': atm_config.eos_CO2,
        'H2': atm_config.eos_H2,
        'CH4': atm_config.eos_CH4,
        'CO': atm_config.eos_CO,
    }
    _eos_map = {k: v for k, v in _eos_map.items() if v is not None}

    # Build species network
    species_list = []

    # Map PROTEUS gas species to atmodeller species with solubility
    _sol_map = {
        'H2O': atm_config.solubility_H2O,
        'CO2': atm_config.solubility_CO2,
        'H2': atm_config.solubility_H2,
        'N2': atm_config.solubility_N2,
        'S2': atm_config.solubility_S2,
        'CO': atm_config.solubility_CO,
        'CH4': atm_config.solubility_CH4,
    }
    # Remove None entries (solubility disabled for that species)
    _sol_map = {k: v for k, v in _sol_map.items() if v is not None}

    # Noble gases dissolve by the Jambon, Weill and Braun (1986) Henry's law,
    # the same calibration CALLIOPE uses, so the two backends agree on noble
    # gas solubility. The models ship with atmodeller.
    for gas in noble_gases:
        _sol_map[gas] = f'{gas}_basalt_jambon86'

    # Only include species whose constituent elements ALL have non-zero budgets.
    # Atmodeller's solver fails when the species network introduces elements
    # with no mass constraint (under-determined system -> no convergence).
    # E.g., CH4 has both H and C; if only H has a budget, including CH4
    # leaves C unconstrained.
    _species_elements = {
        'H2O': {'H'},
        'H2': {'H'},
        'CO2': {'C'},
        'CO': {'C'},
        'CH4': {'H', 'C'},
        'N2': {'N'},
        'NH3': {'H', 'N'},
        'S2': {'S'},
        'SO2': {'S'},
        'H2S': {'H', 'S'},
        'O2': set(),  # always included for fO2
    }
    # Each noble gas is its own element and species.
    for gas in noble_gases:
        _species_elements[gas] = {gas}

    # Determine which elements have budgets above threshold. Under
    # planet.fO2_source = "from_O_budget" the user O budget
    # is authoritative, so O joins the active set; under user_constant
    # the IW buffer constrains O and we exclude it from the mass
    # constraint here.
    fO2_source = config.planet.fO2_source
    constrained_elements = (
        ('H', 'C', 'N', 'S', 'O') if fO2_source == 'from_O_budget' else ('H', 'C', 'N', 'S')
    )
    active_elements = set()
    for element in constrained_elements:
        key = f'{element}_kg_total'
        if float(hf_row.get(key, 0.0)) > config.outgas.mass_thresh:
            active_elements.add(element)

    # Noble gases join the active set on any positive inventory. Their budgets
    # are intrinsically trace and sit below the major-volatile mass threshold,
    # so gating them on mass_thresh would drop a realistic noble inventory.
    for gas in noble_gases:
        if float(hf_row.get(f'{gas}_kg_total', 0.0)) > 0.0:
            active_elements.add(gas)

    # A species is included only if ALL its required elements have budgets
    active_species = set()
    for sp, required_elements in _species_elements.items():
        if required_elements.issubset(active_elements):
            active_species.add(sp)

    _atm_gas_species = {
        'H2O': 'H2O',
        'H2': 'H2',
        'CO2': 'CO2',
        'CO': 'CO',
        'CH4': 'CH4',
        'N2': 'N2',
        # atmodeller keys species by Hill-notation formula, so ammonia is
        # 'H3N' and sulfur dioxide is 'O2S' (not 'NH3' / 'SO2'). The wrapper
        # maps the PROTEUS name to atmodeller's name on both the create_gas
        # input and the output-dict / quick_look lookup; a mismatch silently
        # drops that species' atmospheric mass, which under-counts its element
        # (e.g. S in an SO2-dominated oxidising atmosphere).
        'NH3': 'H3N',
        'S2': 'S2',
        'SO2': 'O2S',
        'H2S': 'H2S',
        'O2': 'O2',
    }
    # Noble gases map to their own atmodeller species names.
    for gas in noble_gases:
        _atm_gas_species[gas] = gas

    for proteus_name, atm_name in _atm_gas_species.items():
        if proteus_name not in active_species:
            continue

        # Build kwargs for create_gas
        kwargs = {}

        # Solubility law (if configured)
        sol_key = _sol_map.get(proteus_name)
        if sol_key:
            if sol_key in solubility_models:
                kwargs['solubility'] = solubility_models[sol_key]
            else:
                log.warning(
                    'Solubility model %r not found for %s; using no solubility',
                    sol_key,
                    proteus_name,
                )

        # Real gas EOS (if configured)
        eos_key = _eos_map.get(proteus_name)
        if eos_key:
            if eos_key in eos_models:
                kwargs['activity'] = eos_models[eos_key]
            else:
                log.warning(
                    'EOS model %r not found for %s; using ideal gas', eos_key, proteus_name
                )

        created = ChemicalSpecies.create_gas(atm_name, **kwargs)
        # The wrapper looks up this species' output by f'{atm_name}_g'. atmodeller
        # keys its output by the species' Hill-notation name, so if the mapped
        # atm_name disagrees with the created species' name the output lookup
        # silently misses and this species' atmospheric mass is dropped (for
        # example the SO2 / 'O2S' Hill-name case). Surface a mismatch loudly at
        # the source rather than letting it drop mass downstream.
        expected_key = f'{atm_name}_g'
        if created.name != expected_key:
            log.warning(
                "atmodeller species name %r for %s does not match the wrapper's "
                'output key %r; its atmospheric/dissolved mass will be dropped. '
                'Update _atm_gas_species[%r] to %r.',
                created.name,
                proteus_name,
                expected_key,
                proteus_name,
                created.name.removesuffix('_g'),
            )
        species_list.append(created)

    log.info(
        'Atmodeller species: %s (active elements: %s)',
        [s.name for s in species_list],
        active_elements,
    )

    # Add condensates only if their elements have budgets
    if atm_config.include_condensates and 'C' in active_elements:
        try:
            species_list.append(ChemicalSpecies.create_condensed('C'))
        except Exception:
            pass  # Graphite not available in all versions

    # Build planet state
    M_planet = float(hf_row.get('M_planet', 1.0 * M_earth))
    R_int = float(hf_row.get('R_int', 6.371e6))
    T_magma = float(hf_row.get('T_magma', 3000.0))
    Phi_global = float(hf_row.get('Phi_global', 1.0))

    # Do not allow low temperatures: clamp to the configured floor, the
    # same semantics as the calliope entry point.
    if T_magma < config.outgas.T_floor:
        T_magma = float(config.outgas.T_floor)
        log.warning('Outgassing temperature clipped to %.1f K' % T_magma)

    # Core mass fraction from config
    cmf = config.interior_struct.core_frac

    planet = Planet(
        planet_mass=M_planet,
        core_mass_fraction=cmf,
        mantle_melt_fraction=Phi_global,
        surface_radius=R_int,
        temperature=T_magma,
        pressure=np.nan,  # Thin-atmosphere approximation
    )

    # Fugacity and mass constraints. The two paths split here:
    #
    # - user_constant: fO2 is buffered to the configured IW
    #   offset and atmodeller derives the resulting O distribution.
    #   Mass constraints cover H/C/N/S only.
    # - from_O_budget: the user O budget is authoritative and
    #   atmodeller back-solves for the IW offset. Mass constraints cover
    #   all five elements; no fugacity constraint on O2_g.
    fugacity_constraints: dict = {}
    mass_constraints: dict = {}
    for element in constrained_elements:
        key = f'{element}_kg_total'
        mass_kg = float(hf_row.get(key, 0.0))
        if mass_kg > config.outgas.mass_thresh:
            mass_constraints[element] = mass_kg

    # Noble gas mass constraints. A noble gas is included on any positive
    # inventory, matching the active-species selection above, so a trace noble
    # budget is not dropped by the major-volatile threshold.
    for gas in noble_gases:
        mass_kg = float(hf_row.get(f'{gas}_kg_total', 0.0))
        if mass_kg > 0.0:
            mass_constraints[gas] = mass_kg

    if fO2_source == 'user_constant':
        fugacity_constraints['O2_g'] = IronWustiteBuffer(config.outgas.fO2_shift_IW)

    # Stash the authoritative O target for the 'from_O_budget' source:
    # atmodeller's output
    # element_density / dissolved_mass arithmetic equals target['O'] only
    # up to the solver's element residual, and we must not let that
    # residual drift hf_row['O_kg_total'] across iterations (the escape
    # pipeline reads this column to compute each iteration's debit).
    target_O_kg = mass_constraints.get('O') if fO2_source == 'from_O_budget' else None

    if not mass_constraints:
        log.warning('No volatile element budgets above threshold; skipping atmodeller')
        if fO2_source == 'from_O_budget':
            # No solve ran, so the derived buffer offset is undefined.
            # Leaving the user pre-seed in place would misrecord a skipped
            # solve as "equilibrated at the user offset".
            hf_row['fO2_shift_IW_derived'] = float('nan')
            hf_row['O_res'] = float('nan')
        return

    if fO2_source == 'from_O_budget' and 'O' not in mass_constraints:
        raise ValueError(
            'planet.fO2_source = "from_O_budget" requires '
            'hf_row["O_kg_total"] > outgas.mass_thresh, but the O '
            'budget is below threshold. Increase planet.elements.O_budget '
            'or switch to fO2_source = "user_constant".'
        )

    log.info(
        'Atmodeller solve: T=%.0f K, Phi=%.2f, elements=%s',
        T_magma,
        Phi_global,
        {k: f'{v:.2e}' for k, v in mass_constraints.items()},
    )

    # Build solver parameters from config
    from atmodeller.containers import SolverParameters

    solver_params = SolverParameters(
        atol=config.outgas.solver_atol,
        rtol=config.outgas.solver_rtol,
        max_steps=atm_config.solver_max_steps,
        multistart=atm_config.solver_multistart,
    )

    # Reuse one EquilibriumModel per (species network, fugacity source, solver
    # mode) so atmodeller's JIT solver (cached on the model as ._solver) is
    # compiled once, not rebuilt every solve. The latter two are fixed per run;
    # they are in the key so a reused model can never carry a solver compiled for
    # a different structure. Built here, after the skip/validation guards, so a
    # skipped or rejected call never constructs a model.
    model_signature = (
        tuple(s.name for s in species_list),
        fO2_source,
        atm_config.solver_mode,
    )
    model = _cached_model(
        model_signature,
        lambda: EquilibriumModel(SpeciesNetwork(tuple(species_list))),
    )

    # Solve equilibrium
    try:
        model.solve(
            state=planet,
            fugacity_constraints=fugacity_constraints,
            mass_constraints=mass_constraints,
            solver=atm_config.solver_mode,
            solver_parameters=solver_params,
        )
    except Exception as e:
        log.error('Atmodeller solve failed: %s', e)
        raise

    # Extract results
    output = model.output
    quick_look = output.quick_look()
    total_P = float(np.squeeze(output.total_pressure()))

    log.info('Atmodeller result: P_total=%.2f bar', total_P)

    # atmodeller's per-species output dict carries the thermodynamically
    # consistent atmospheric (``gas_mass``) and dissolved (``dissolved_mass``)
    # masses. Read it once and reuse for both reservoirs below.
    try:
        output_dict = output.asdict()
    except Exception:
        output_dict = {}

    # Map atmodeller output back to hf_row
    _reverse_map = {v: k for k, v in _atm_gas_species.items()}

    gravity = float(hf_row.get('gravity', 9.81))
    area = 4.0 * np.pi * R_int**2

    P_total = 0.0
    for atm_name, p_bar in quick_look.items():
        proteus_name = _reverse_map.get(atm_name.replace('_g', ''))
        if proteus_name is None:
            log.debug('Atmodeller quick_look key %r not in species map; skipping', atm_name)
            continue
        if proteus_name not in gas_list:
            continue
        p_val = float(np.squeeze(p_bar))
        hf_row[f'{proteus_name}_bar'] = p_val
        P_total += p_val

        # Atmospheric mass: use atmodeller's internal per-species atmospheric
        # mass (``gas_mass``), consistent with the ``dissolved_mass`` read below
        # so the per-element atm+liquid split closes against the conserved
        # element budget. The thin-atmosphere relation p*1e5*A/g recovers
        # x_i * M_atm, the species' mole-fraction share of the total column,
        # which mis-weights each species by mu_mean/mu_i and breaks the
        # per-element closure; it is kept only as a fallback for the case where
        # the solver output omits gas_mass.
        gas_mass = _species_output_scalar(output_dict, atm_name, 'gas_mass')
        if gas_mass is not None:
            # A small negative solver residual is clamped to zero rather than
            # routed to the fallback (a near-zero species has a near-zero column
            # mass anyway); the fallback is reserved for a genuinely absent
            # gas_mass, i.e. an atmodeller output-schema change.
            hf_row[f'{proteus_name}_kg_atm'] = max(0.0, gas_mass)
        elif gravity > 0 and area > 0:
            log.warning(
                "atmodeller output missing 'gas_mass' for %s; falling back to the "
                'p*A/g thin-atmosphere reconstruction, which mis-weights the '
                'per-element split by mu_mean/mu_species.',
                atm_name,
            )
            hf_row[f'{proteus_name}_kg_atm'] = p_val * 1e5 * area / gravity
        else:
            hf_row[f'{proteus_name}_kg_atm'] = 0.0

    # Total surface pressure
    hf_row['P_surf'] = P_total

    # VMRs
    if P_total > 0:
        for s in gas_list:
            hf_row[f'{s}_vmr'] = float(hf_row.get(f'{s}_bar', 0.0)) / P_total
    else:
        for s in gas_list:
            hf_row[f'{s}_vmr'] = 0.0

    # Dissolved masses from atmodeller output (thermodynamically consistent).
    # Falls back to kg_total - kg_atm for species not in the solve or without
    # a dissolved_mass output (e.g., gas-only species like H2S, NH3).
    def _as_mass(value):
        """Squeeze an atmodeller mass output to a float, or None on failure."""
        if value is None:
            return None
        try:
            return float(np.squeeze(value))
        except (TypeError, ValueError):
            return None

    # Snapshot the escape-authoritative noble gas element totals before the
    # output overwrites them. For a noble gas the species and the element are
    # the same key, and {gas}_kg_total is owned by the escape pipeline (which
    # debits it each step). The solver's total_mass equals this constraint only
    # to its residual, so the total is restored below rather than overwritten,
    # mirroring how the oxygen element total is preserved.
    noble_total_in = {g: float(hf_row.get(f'{g}_kg_total', 0.0)) for g in noble_gases}

    for proteus_name, atm_name in _atm_gas_species.items():
        if proteus_name not in gas_list:
            continue
        is_noble = proteus_name in noble_gases
        species_key = f'{atm_name}_g'
        species_data = output_dict.get(species_key, {})
        if not isinstance(species_data, dict):
            species_data = {}

        dissolved_kg = _as_mass(species_data.get('dissolved_mass'))
        total_kg = _as_mass(species_data.get('total_mass'))

        hf_row[f'{proteus_name}_kg_solid'] = 0.0

        if total_kg is not None:
            # atmodeller's per-species total_mass is the conserved mass
            # constraint, and its dissolved_mass is the melt share. Deriving
            # the atmospheric mass as total minus dissolved keeps the tracked
            # inventory conserved and correctly mass-weighted per species. The
            # pressure-derived atmospheric mass from the first loop distributes
            # the column by mole fraction, which misweights a species whose
            # molar mass differs from the mean (badly for a light noble gas),
            # so it is overwritten here whenever atmodeller reports the total.
            liquid = min(max(0.0, dissolved_kg if dissolved_kg is not None else 0.0), total_kg)
            hf_row[f'{proteus_name}_kg_liquid'] = liquid
            hf_row[f'{proteus_name}_kg_atm'] = max(0.0, total_kg - liquid)
            # A noble gas element total is escape-owned; restore the pre-solve
            # value so the solver's residual cannot drift the tracked budget
            # across iterations. Its atmosphere and melt split are still taken
            # from the solve.
            hf_row[f'{proteus_name}_kg_total'] = (
                noble_total_in[proteus_name] if is_noble else total_kg
            )
        elif is_noble:
            # An inactive noble gas is not in the solve; clear its stale
            # atmospheric reservoir so it does not leak into P_surf, the mean
            # molar mass, or a later resurrection of its total, and leave its
            # escape-owned element total untouched. The mixing ratio is cleared
            # too: it was already recomputed from the stale partial pressure
            # above, so it must be reset or it would still feed the mean molar
            # mass sum.
            hf_row[f'{proteus_name}_kg_liquid'] = 0.0
            hf_row[f'{proteus_name}_kg_atm'] = 0.0
            hf_row[f'{proteus_name}_bar'] = 0.0
            hf_row[f'{proteus_name}_vmr'] = 0.0
        else:
            # Reactive species not present in the atmodeller output (e.g.
            # excluded from the solve): keep the pressure-derived atmospheric
            # mass from the first loop and treat it as gas-only.
            hf_row[f'{proteus_name}_kg_liquid'] = (
                max(0.0, dissolved_kg) if dissolved_kg is not None else 0.0
            )
            hf_row[f'{proteus_name}_kg_total'] = float(
                hf_row.get(f'{proteus_name}_kg_atm', 0.0)
            ) + float(hf_row[f'{proteus_name}_kg_liquid'])

    # Mean molecular weight (approximate from VMRs). The noble gases are
    # gas_list members with their own VMRs, so they enter this sum directly;
    # their molar mass is the element molar mass since the species and the
    # element are the same monatomic entity.
    _mmw = {
        'H2O': 18.015e-3,
        'CO2': 44.01e-3,
        'H2': 2.016e-3,
        'CO': 28.01e-3,
        'CH4': 16.04e-3,
        'N2': 28.01e-3,
        'S2': 64.13e-3,
        'SO2': 64.07e-3,
        'O2': 32.0e-3,
        'H2S': 34.08e-3,
        'NH3': 17.03e-3,
        **{gas: element_mmw[gas] for gas in noble_gases},
    }
    mmw = sum(float(hf_row.get(f'{s}_vmr', 0.0)) * _mmw.get(s, 28.0e-3) for s in gas_list)
    if mmw > 0:
        hf_row['atm_kg_per_mol'] = mmw

    log.info(
        'Atmodeller: P_surf=%.2f bar, MMW=%.3f g/mol',
        P_total,
        mmw * 1e3,
    )

    # Populate the per-element reservoir split ({e}_kg_atm/liquid/solid) from the
    # per-species inventory so the escape step (which reads {e}_kg_atm under
    # reservoir="outgas") keeps a live element budget under atmodeller, as
    # CALLIOPE writes natively. The per-element {e}_kg_total slot is left to its
    # owner (escape for the running budget; the authoritative O_kg_total write
    # below for oxygen), so escape's debited total is never overwritten here.
    _populate_volatile_element_reservoirs(hf_row, element_mmw)

    # Add condensed-phase mass (e.g. graphite when include_condensates is on) to
    # the per-element solid reservoir. The gas-species inventory above does not
    # carry condensates, so the per-element atm+liquid+solid split must pick the
    # condensed mass up here to close against the element budget for any element
    # that condenses. atmodeller reports the condensed mass per element; it is
    # zero for the elements (and conditions) that do not condense, so this is a
    # no-op except when a condensate forms.
    for el in {e for comp in _VOLATILE_ELEMENT_STOICH.values() for e in comp}:
        condensed = _species_output_scalar(output_dict, f'element_{el}', 'condensed_mass')
        if condensed:
            hf_row[f'{el}_kg_solid'] = float(hf_row.get(f'{el}_kg_solid', 0.0)) + max(
                0.0, condensed
            )

    # Total volatile oxygen (atmospheric + melt-dissolved), summed over the
    # O-bearing volatile species. Computed for every fO2 source so the
    # whole-planet O accounting (M_ele) includes atmospheric O whether or not
    # O was a mass constraint; mirrors CALLIOPE, which always writes O_kg_total.
    volatile_O_kg = _total_volatile_oxygen_kg(hf_row, element_mmw)

    # Plumb the derived IW-buffer offset only for the 'from_O_budget'
    # source. Under
    # user_constant the wrapper's pre-dispatch echo of
    # config.outgas.fO2_shift_IW IS the offset the chemistry equilibrated
    # to (the fugacity constraint set it), so overwriting with
    # atmodeller's back-computed log10dIW_1_bar would lose bit-for-bit
    # echo of the user input and introduce a small Hirschmann-buffer
    # 1-bar-vs-P reconstruction drift. The IW buffer atmodeller uses
    # for the 'from_O_budget' source is Hirschmann combined; CALLIOPE uses O'Neill & Eggins
    # 2002 (~0.95 dex offset at 3000 K). Cross-backend comparison work
    # later in the framework will quantify this; for now the column is
    # backend-faithful (each wrapper reports its own buffer).
    if fO2_source == 'from_O_budget':
        o2_dict = output_dict.get('O2_g', {})
        log10dIW = o2_dict.get('log10dIW_1_bar') if isinstance(o2_dict, dict) else None
        if log10dIW is None:
            log.warning(
                "atmodeller output missing 'O2_g.log10dIW_1_bar' for the "
                "'from_O_budget' source; "
                'fO2_shift_IW_derived left at pre-dispatch default '
                '(%.3f) which is meaningless when O was a mass constraint.',
                hf_row.get('fO2_shift_IW_derived', float('nan')),
            )
        else:
            try:
                hf_row['fO2_shift_IW_derived'] = float(np.squeeze(log10dIW))
            except (TypeError, ValueError) as e:
                log.warning(
                    'atmodeller log10dIW_1_bar JAX conversion failed (%s); '
                    'fO2_shift_IW_derived left at pre-dispatch default '
                    '(%.3f) which is meaningless when O was a mass constraint.',
                    e,
                    hf_row.get('fO2_shift_IW_derived', float('nan')),
                )
        # O mass-balance residual: total volatile O (atmospheric + dissolved,
        # computed above) minus the authoritative user target. Reported in kg
        # to match CALLIOPE's H/C/N/S/O_res convention.
        hf_row['O_res'] = volatile_O_kg - float(target_O_kg)

        # Restore the authoritative user O budget. atmodeller's solver may
        # have converged to atm_O+liq_O slightly off from target_O_kg (within
        # its atol/rtol); the per-iteration drift would otherwise accumulate
        # in the escape pipeline that reads hf_row['O_kg_total'] to compute
        # the next debit.
        hf_row['O_kg_total'] = float(target_O_kg)
    else:
        # Buffered-fO2 sources (e.g. user_constant): O is not a mass
        # constraint, so the equilibrium-derived total volatile O is the
        # tracked element budget. Without this, M_ele / M_planet omit
        # atmospheric O entirely (e.g. the O2 atmosphere at oxidising IW),
        # which breaks the M_atm <= M_planet accounting at high fO2. The
        # escape pipeline's between-step debit of O_kg_total is superseded
        # here each call, exactly as in CALLIOPE under user_constant: the
        # buffered equilibrium re-derives O from the current H/C/S
        # inventory, which itself falls as escape depletes those budgets.
        hf_row['O_kg_total'] = volatile_O_kg
        # No authoritative O target under a buffered fO2 source, so the O
        # mass-balance residual is zero by construction. Written explicitly
        # to match CALLIOPE rather than relying on the pre-dispatch seed.
        hf_row['O_res'] = 0.0
