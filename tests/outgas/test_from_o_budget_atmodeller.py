"""Tests for the PROTEUS-side from_O_budget atmodeller wrapper.

Covers the dispatch logic added to ``proteus.outgas.atmodeller`` when
``planet.fO2_source == 'from_O_budget'``:

* The wrapper builds a 5-key mass_constraints dict (H/C/N/S/O sourced
  from ``hf_row``) and an empty fugacity_constraints dict under from_O_budget.
* The wrapper preserves the legacy contract under user_constant (4-key
  mass_constraints, IronWustiteBuffer fugacity constraint).
* After the solve, the wrapper writes ``fO2_shift_IW_derived`` from
  atmodeller's back-computed ``log10dIW_1_bar`` output and reports the
  O mass residual.
* Under from_O_budget the authoritative ``hf_row['O_kg_total']`` is
  restored to the user target after the solve so per-iteration solver
  drift cannot accumulate into the escape pipeline.
* Under user_constant ``hf_row['O_kg_total']`` is set to the
  chemistry-derived total volatile oxygen (atmospheric + dissolved) so
  the whole-planet O accounting (``M_ele``) includes atmospheric O even
  though O is buffered, not a mass constraint (issue #677).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the whole module if atmodeller isn't installed. The macOS CI
# runner doesn't pull atmodeller (it isn't a hard dependency in
# pyproject.toml); Docker-based runners have it pre-installed in the
# image. Skipping at module scope is cleaner than guarding every test,
# and avoids leaking config-load failures from check_module_dependencies
# when a test builds a Config with outgas.module = "atmodeller".
pytest.importorskip(
    'atmodeller',
    reason='atmodeller package not installed; from_O_budget atmodeller wrapper tests skipped',
)

from proteus.utils.constants import element_list, gas_list, noble_gases  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


logging.getLogger('atmodeller').setLevel(logging.WARNING)


def _make_from_o_budget_config():
    """Build a MagicMock Config consistent with the atmodeller backend
    under from_O_budget dispatch."""
    config = MagicMock()
    config.outgas.module = 'atmodeller'
    config.outgas.fO2_shift_IW = 4.0
    config.outgas.T_floor = 1200.0
    config.outgas.mass_thresh = 1e8
    config.outgas.solver_atol = 1e-8
    config.outgas.solver_rtol = 1e-5
    config.outgas.atmodeller.solver_max_steps = 100
    config.outgas.atmodeller.solver_multistart = 1
    config.outgas.atmodeller.solver_mode = 'robust'
    config.outgas.atmodeller.include_condensates = False
    # Disable every solubility law to keep the species network minimal
    for name in ('H2O', 'CO2', 'H2', 'N2', 'S2', 'CO', 'CH4'):
        setattr(config.outgas.atmodeller, f'solubility_{name}', None)
    for name in ('H2O', 'CO2', 'H2', 'CH4', 'CO'):
        setattr(config.outgas.atmodeller, f'eos_{name}', None)
    config.planet.fO2_source = 'from_O_budget'
    config.interior_struct.core_frac = 0.3
    return config


def _earth_hf_row(O_kg_total: float = 1.0e22) -> dict:
    """Helpfile row populated up to the point ``calc_surface_pressures_atmodeller``
    is called."""
    hf: dict = {
        'M_mantle': 4.03e24,
        'M_int': 5.97e24,
        'M_planet': 5.97e24,
        'gravity': 9.81,
        'R_int': 6.371e6,
        'Phi_global': 1.0,
        'T_magma': 1800.0,
        'Time': 0.0,
    }
    for e in element_list:
        hf[f'{e}_kg_total'] = 0.0
    hf['H_kg_total'] = 1.5e20
    hf['C_kg_total'] = 1.5e19
    hf['N_kg_total'] = 8.0e18
    hf['S_kg_total'] = 8.0e20
    hf['O_kg_total'] = O_kg_total
    for s in gas_list:
        hf[f'{s}_kg_atm'] = 0.0
        hf[f'{s}_kg_liquid'] = 0.0
        hf[f'{s}_kg_solid'] = 0.0
        hf[f'{s}_vmr'] = 0.0
        hf[f'{s}_bar'] = 0.0
    return hf


def _fake_atmodeller_output(log10dIW: float = -0.1):
    """Build a MagicMock that mimics ``EquilibriumModel.solve(...)`` ->
    ``model.output`` chain. The output's ``asdict()`` populates the keys
    the wrapper reads (``O2_g.log10dIW_1_bar`` and per-species ``gas_mass`` /
    ``dissolved_mass``); ``quick_look()`` provides partial pressures.

    Each species carries ``gas_mass`` (the atmospheric mass the wrapper now
    sources directly, rather than reconstructing it from the partial pressure
    via p*A/g) so these dispatch / plumbing tests exercise the primary
    atmospheric-mass path. The values are independent of the partial pressures
    on purpose: a regression that reverted to the p*A/g reconstruction would
    produce different ``{sp}_kg_atm`` here.
    """
    out = MagicMock()
    out.quick_look.return_value = {
        'H2O_g': np.array(50.0),
        'CO2_g': np.array(5.0),
        'N2_g': np.array(1.0),
        'S2_g': np.array(0.5),
        'O2_g': np.array(1e-6),
    }
    out.total_pressure.return_value = np.array(56.5)
    out.asdict.return_value = {
        'O2_g': {'log10dIW_1_bar': np.array(log10dIW), 'gas_mass': np.array(4.0e12)},
        'H2O_g': {'gas_mass': np.array(3.0e20), 'dissolved_mass': np.array(2.0e19)},
        'CO2_g': {'gas_mass': np.array(3.0e19), 'dissolved_mass': np.array(1.0e18)},
        'N2_g': {'gas_mass': np.array(5.0e18)},
        'S2_g': {'gas_mass': np.array(2.5e18)},
        'state': {'temperature': np.array(1800.0), 'pressure': np.array(56.5)},
    }
    return out


# ---------------------------------------------------------------------------
# Dispatch: from_O_budget builds the 5-key mass_constraints
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_builds_5_element_mass_constraints():
    """Under fO2_source = 'from_O_budget' atmodeller must receive O as
    an element-mass constraint. The legacy path constrained only
    H/C/N/S; from_O_budget adds the user O budget.

    Discriminating: the call_args's mass_constraints dict carries the
    'O' key with the value from hf_row['O_kg_total']. A regression that
    forgot to add O would leave only H/C/N/S, and atmodeller would
    silently fall back to the IW fugacity buffer if one were still set.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-0.5)

    with patch(
        'atmodeller.EquilibriumModel',
        return_value=fake_model,
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert set(mass_kwargs.keys()) == {'H', 'C', 'N', 'S', 'O'}
    assert mass_kwargs['O'] == pytest.approx(1.0e22)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_drops_O2_fugacity_constraint():
    """The IronWustiteBuffer fugacity constraint must not be passed
    under from_O_budget. atmodeller would otherwise over-constrain the system
    (5 mass equations + 1 fugacity = 6 constraints on 5 elements).

    Discriminating: the call_args's fugacity_constraints dict is empty.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    fug_kwargs = fake_model.solve.call_args.kwargs['fugacity_constraints']
    assert fug_kwargs == {}
    # Constraint-count invariant: from_O_budget uses 5 element-mass constraints
    # (H/C/N/S/O) and zero fugacity constraints. A regression that
    # added O to mass_constraints but forgot to drop the IW buffer
    # would over-constrain the system (5 mass + 1 fugacity > 5 unknowns).
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert len(mass_kwargs) + len(fug_kwargs) == 5
    # Discrimination: 'O' must be in mass_constraints (the from_O_budget
    # invariant), not silently dropped from both.
    assert 'O' in mass_kwargs


@pytest.mark.unit
def test_legacy_path_keeps_4_element_mass_constraints_and_fO2_buffer():
    """Mirror of the from_O_budget tests: under fO2_source = 'user_constant'
    the wrapper passes H/C/N/S (no O) and a single IW fugacity
    constraint. A regression that swapped which path got the buffer
    would silently invert the legacy chemistry.
    """
    from atmodeller.thermodata import IronWustiteBuffer

    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    fug_kwargs = fake_model.solve.call_args.kwargs['fugacity_constraints']
    assert set(mass_kwargs.keys()) == {'H', 'C', 'N', 'S'}
    assert 'O' not in mass_kwargs
    assert 'O2_g' in fug_kwargs
    assert isinstance(fug_kwargs['O2_g'], IronWustiteBuffer)


@pytest.mark.unit
def test_wrapper_reuses_model_until_species_network_changes():
    """The wrapper builds one EquilibriumModel across solves with the same active
    species set, and a new one only when the network changes.

    This is the regression surface of the JIT-recompile out-of-memory fix:
    atmodeller caches its compiled solver on the model instance, so building a
    fresh model every solve recompiled the solver until the job was killed.
    Two solves with identical budgets must construct the model exactly once
    (the second reuses the cached compiled solver). Dropping sulfur below the
    mass threshold removes the S-bearing species, shrinks the network, changes
    the cache signature, and must construct a second model. A regression that
    rebuilt the model every call would show three constructions; one that never
    rebuilt on a network change would show one.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()

    builds = []

    def _make_model(*args, **kwargs):
        m = MagicMock()
        m.output = _fake_atmodeller_output()
        builds.append(m)
        return m

    with patch('atmodeller.EquilibriumModel', side_effect=_make_model):
        # Two solves, identical active-species network -> built once, then reused.
        calc_surface_pressures_atmodeller(dirs, config, _earth_hf_row(O_kg_total=1.0e22))
        calc_surface_pressures_atmodeller(dirs, config, _earth_hf_row(O_kg_total=1.0e22))
        assert len(builds) == 1  # reused; atmodeller's compiled solver not rebuilt

        # Sulfur drops below the mass threshold -> S2/SO2/H2S leave the network
        # -> different signature -> a new model (and a fresh solver) is built.
        hf_no_s = _earth_hf_row(O_kg_total=1.0e22)
        hf_no_s['S_kg_total'] = 0.0
        calc_surface_pressures_atmodeller(dirs, config, hf_no_s)
        assert len(builds) == 2  # network shrank -> rebuilt exactly once more


# ---------------------------------------------------------------------------
# Helpfile plumbing: fO2_shift_IW_derived comes from atmodeller's output
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_writes_solver_derived_fO2_to_helpfile():
    """Under from_O_budget the wrapper must read atmodeller's back-computed
    log10dIW_1_bar from ``output.asdict()`` and write it to
    ``hf_row['fO2_shift_IW_derived']``. This is the from_O_budget scientific
    output.

    Discriminating: the fake output reports an offset of -2.7. A
    regression that fell back to the wrapper-level default
    (config.outgas.fO2_shift_IW = 4.0) would record 4.0, the wrong
    value.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-2.7)

    # Pre-set the wrapper default (mirrors what run_outgassing does
    # before dispatch). The wrapper must overwrite it.
    hf_row['fO2_shift_IW_derived'] = config.outgas.fO2_shift_IW

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(-2.7)
    # Discrimination: a regression that left the pre-seeded default
    # (config.outgas.fO2_shift_IW = 4.0) in place would land at 4.0,
    # which differs from the solver's -2.7 by 6.7 dex (well above any
    # tolerance). Pin the magnitude of the disagreement.
    assert abs(hf_row['fO2_shift_IW_derived'] - config.outgas.fO2_shift_IW) > 1.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_legacy_path_preserves_pre_dispatch_fO2_echo():
    """Under user_constant the wrapper must NOT overwrite
    fO2_shift_IW_derived with atmodeller's back-computed value, because
    the fugacity constraint already set the offset to exactly
    config.outgas.fO2_shift_IW. Letting atmodeller's Hirschmann-buffer
    reconstruction overwrite the echo would introduce a small drift
    (buffer-formula and 1-bar-vs-P-evaluation differences) that breaks
    bit-for-bit echo of the user input.

    Discriminating: the fake output reports a perturbed log10dIW of
    -2.7, but config.outgas.fO2_shift_IW is 4.0. A regression that
    unconditionally writes the solver value would record -2.7 instead
    of 4.0.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row()
    hf_row['fO2_shift_IW_derived'] = config.outgas.fO2_shift_IW  # pre-seed

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-2.7)

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['fO2_shift_IW_derived'] == pytest.approx(config.outgas.fO2_shift_IW)
    # Discrimination: the solver's -2.7 must NOT have overwritten the
    # echoed 4.0. A regression that unconditionally writes the solver
    # value would land 6.7 dex away from the user input.
    assert abs(hf_row['fO2_shift_IW_derived'] - (-2.7)) > 1.0


# ---------------------------------------------------------------------------
# Per-species kg_total invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_atmodeller_maintains_per_species_kg_total_invariant():
    """The CALLIOPE wrapper provides ``{species}_kg_total = kg_atm +
    kg_liquid + kg_solid`` natively via its solver output. The
    atmodeller wrapper used to leave the column at its previous
    iteration's value, which produced systematically wrong totals when a
    downstream subtraction-fallback read a stale kg_total. The wrapper
    must now write the invariant explicitly per species.

    Discriminating: the fake output sets H2O atmospheric mass via the
    pressure-mass conversion and dissolved mass via output.asdict.
    A regression that skipped the kg_total write would leave the column
    at 0.0 (the fixture seeds zero) while kg_atm + kg_liquid is
    non-zero.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        kg_atm = hf_row[f'{sp}_kg_atm']
        kg_liq = hf_row[f'{sp}_kg_liquid']
        kg_sol = hf_row[f'{sp}_kg_solid']
        kg_total = hf_row[f'{sp}_kg_total']
        assert kg_total == pytest.approx(kg_atm + kg_liq + kg_sol, rel=1e-12)
    # Positivity / non-stale discrimination: at least one species (H2O
    # is the largest in the fake output) must have a strictly positive
    # post-call kg_total. A regression that skipped the write entirely
    # would leave the fixture's seeded 0.0; that would still satisfy
    # the conservation closure above (since kg_atm + kg_liq + kg_sol
    # would also be 0 if the per-species writes were skipped), making
    # the closure alone insufficient.
    assert hf_row['H2O_kg_total'] > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_atmodeller_kg_total_not_stale_after_call():
    """A regression that read stale ``_kg_total`` from a prior
    iteration would silently propagate a wrong value forward. Pre-load
    the fixture with absurd ``_kg_total`` values and verify the wrapper
    overwrites them with the correct ``kg_atm + kg_liquid + kg_solid``.

    Discriminating: H2O pre-loaded at 1e30 kg; if the wrapper left that
    untouched, the kg_total invariant would fail by ~10 orders of
    magnitude.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()
    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        hf_row[f'{sp}_kg_total'] = 1.0e30  # impossible value

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    for sp in ('H2O', 'CO2', 'N2', 'S2'):
        kg_atm = hf_row[f'{sp}_kg_atm']
        kg_liq = hf_row[f'{sp}_kg_liquid']
        kg_sol = hf_row[f'{sp}_kg_solid']
        # The total must be the post-solve sum, not the pre-loaded 1e30.
        assert hf_row[f'{sp}_kg_total'] == pytest.approx(kg_atm + kg_liq + kg_sol)
        assert hf_row[f'{sp}_kg_total'] < 1.0e25  # nowhere near the pre-load


# ---------------------------------------------------------------------------
# Authoritative-O preservation across the call
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_from_o_budget_preserves_authoritative_O_kg_total():
    """Under from_O_budget the user O budget is authoritative. atmodeller's
    converged element_density at iteration N is mass_atm + mass_dissolved
    which differs from the target by the solver's residual. Letting
    that drift contaminate ``hf_row['O_kg_total']`` would cascade into
    the escape pipeline (which reads this column to compute the next
    iteration's debit), so the wrapper restores the authoritative input
    after the solve.

    Discriminating: hf_row['O_kg_total'] equals the input target to
    floating-point precision, regardless of what atmodeller computed
    for atm + dissolved O.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert hf_row['O_kg_total'] == pytest.approx(1.0e22, rel=1e-12)
    # Discrimination: the wrapper must have passed the SAME O value
    # into atmodeller's mass_constraints. A regression where the
    # solver wrote a residual-perturbed value into O_kg_total would
    # also break the mass-constraint input pin.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert mass_kwargs['O'] == pytest.approx(1.0e22, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_user_constant_writes_chemistry_derived_O_kg_total():
    """Under user_constant the wrapper records the chemistry-derived total
    volatile oxygen (atmospheric + dissolved) in ``O_kg_total`` so the
    whole-planet O accounting (``M_ele``) includes atmospheric O even
    though O is buffered, not a mass constraint (issue #677). Without this
    the pre-seeded value (0 under O_mode='ic_chemistry') would persist and
    M_ele would omit the atmospheric O entirely (e.g. an oxidising O2
    atmosphere), breaking the M_atm <= M_planet invariant at high fO2.

    Discriminating: the pre-seeded O_kg_total (5.5e21) must be replaced by
    the equilibrium-derived total volatile O summed over the O-bearing
    species, not left untouched and not driven to zero.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller
    from proteus.utils.constants import element_mmw

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.planet.fO2_source = 'user_constant'
    hf_row = _earth_hf_row(O_kg_total=5.5e21)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    # Independently re-derive the expected total volatile O from the
    # per-species atmospheric+dissolved masses the wrapper just populated
    # (NOT via the helper under test), so a stoichiometry error in the
    # helper would surface here rather than cancel out.
    m_O = element_mmw['O']
    m_C = element_mmw['C']
    m_H = element_mmw['H']
    m_S = element_mmw['S']
    o_frac = {
        'H2O': m_O / (2 * m_H + m_O),
        'CO2': 2 * m_O / (m_C + 2 * m_O),
        'CO': m_O / (m_C + m_O),
        'SO2': 2 * m_O / (m_S + 2 * m_O),
        'O2': 1.0,
    }
    expected = sum(
        (hf_row.get(f'{sp}_kg_atm', 0.0) + hf_row.get(f'{sp}_kg_liquid', 0.0)) * f
        for sp, f in o_frac.items()
    )
    assert hf_row['O_kg_total'] == pytest.approx(expected, rel=1e-12)
    # The fake solve carries real O-bearing volatiles, so the result is a
    # strictly positive reservoir, NOT the bug's 0 and NOT the stale pre-seed.
    assert hf_row['O_kg_total'] > 0.0
    assert abs(hf_row['O_kg_total'] - 5.5e21) > 1e21
    # Discrimination: O must still be solved via the IW fugacity buffer
    # (NOT added as a mass constraint) under user_constant.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert 'O' not in mass_kwargs


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_total_volatile_oxygen_sums_atm_and_dissolved_over_o_bearing_species():
    """``_total_volatile_oxygen_kg`` sums oxygen across both the atmospheric
    and dissolved reservoirs of the O-bearing volatiles {H2O, CO2, CO, SO2,
    O2}, weighted by the stoichiometric O mass fraction. This is the
    whole-planet O reservoir that M_ele must include.

    Discriminating: O-free volatiles (H2, CH4, N2) contribute exactly zero,
    and dropping the dissolved part underestimates the total.
    """
    from proteus.outgas.atmodeller import _total_volatile_oxygen_kg
    from proteus.utils.constants import element_mmw

    m_O = element_mmw['O']
    m_C = element_mmw['C']
    m_H = element_mmw['H']
    m_S = element_mmw['S']
    hf_row = {
        'H2O_kg_atm': 1.8e21,
        'H2O_kg_liquid': 2.0e20,
        'CO2_kg_atm': 4.4e20,
        'SO2_kg_atm': 6.4e19,
        'O2_kg_atm': 3.2e20,
        'CH4_kg_atm': 1.0e20,  # carries no oxygen
        'N2_kg_atm': 5.0e20,  # carries no oxygen
    }
    expected = (
        (1.8e21 + 2.0e20) * (m_O / (2 * m_H + m_O))  # H2O atm + dissolved
        + 4.4e20 * (2 * m_O / (m_C + 2 * m_O))  # CO2
        + 6.4e19 * (2 * m_O / (m_S + 2 * m_O))  # SO2
        + 3.2e20 * 1.0  # O2 is pure oxygen
    )
    got = _total_volatile_oxygen_kg(hf_row, element_mmw)
    assert got == pytest.approx(expected, rel=1e-12)
    # Discrimination 1: CH4 + N2 (6.0e20 kg) carry no O, so a buggy
    # all-volatiles sum would exceed the correct value by a large margin.
    assert abs(got - (expected + 6.0e20)) > 1e20
    # Discrimination 2: the dissolved H2O O is included (not atmosphere-only).
    atm_only = got - 2.0e20 * (m_O / (2 * m_H + m_O))
    assert got - atm_only == pytest.approx(2.0e20 * (m_O / (2 * m_H + m_O)), rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_total_volatile_oxygen_zero_without_o_bearing_species():
    """Edge case: with only O-free volatiles present (or an empty row) the
    total volatile oxygen is exactly zero, not a spurious small float."""
    from proteus.outgas.atmodeller import _total_volatile_oxygen_kg
    from proteus.utils.constants import element_mmw

    assert _total_volatile_oxygen_kg({}, element_mmw) == 0.0
    hf_row = {'H2_kg_atm': 1e20, 'CH4_kg_atm': 2e20, 'N2_kg_liquid': 3e20}
    assert _total_volatile_oxygen_kg(hf_row, element_mmw) == 0.0


# ---------------------------------------------------------------------------
# Edge: O below mass threshold under from_O_budget must fail loudly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_with_O_below_threshold_raises():
    """from_O_budget cannot work if the user O budget falls below the mass
    threshold (no constraint to invert against). The wrapper must fail
    with a clear ValueError rather than silently producing chemistry
    that ignores the O input.

    Discriminating: error names the field the user should change.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    # mass_thresh = 1e8 by config; set O budget well below it.
    hf_row = _earth_hf_row(O_kg_total=1.0)

    with pytest.raises(ValueError, match='from_O_budget'):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # Boundary discrimination: an O budget comfortably above the
    # mass_thresh of 1e8 kg on the same from_O_budget config must NOT raise.
    # The rejection is driven by the threshold, not by from_O_budget itself.
    # A regression that hard-raised on every from_O_budget call would fail
    # this.
    hf_row_ok = _earth_hf_row(O_kg_total=1.0e22)
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row_ok)
    assert hf_row_ok['O_kg_total'] == pytest.approx(1.0e22, rel=1e-12)


# ---------------------------------------------------------------------------
# O residual is computed from the species O distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_from_o_budget_writes_finite_O_residual():
    """Under from_O_budget the wrapper must compute and write ``O_res`` =
    (atmospheric_O + dissolved_O) - target. A finite, bounded, correctly
    signed residual is the strongest assertion available without the real
    solver; the fake output's hand-built numbers give a deterministic check.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    target_O = 1.0e22
    hf_row = _earth_hf_row(O_kg_total=target_O)

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)

    assert 'O_res' in hf_row
    assert np.isfinite(hf_row['O_res'])
    # The fake output produces a finite atmospheric + dissolved O, so
    # the residual must be strictly less than the full target.
    assert abs(hf_row['O_res']) < target_O
    # Sign pin: the equilibrium volatile O (~1e20) sits far below the 1e22
    # target, so O_res = volatile_O - target must be strongly negative. A
    # sign-flipped residual (target - volatile) would pass the bounded
    # check above but land positive here.
    assert hf_row['O_res'] < 0.0


# ---------------------------------------------------------------------------
# Branch coverage: unknown solubility/EOS keys, T_floor early return,
# solver exception, zero P_total VMR fallback. These exercise paths the
# main from_O_budget tests above do not reach because their config uses
# valid keys, a high enough T_magma, and a successful fake solve.
# ---------------------------------------------------------------------------


def test_atmodeller_warns_when_solubility_key_unknown(caplog):
    """A configured solubility model name that does NOT appear in
    atmodeller's solubility registry must log a warning and continue
    with no solubility (the species is still added).

    Discriminating: pin the gas name in the warning message so a
    regression that silently dropped the warning, or that crashed on
    the unknown key instead of warning, would fail this test.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.atmodeller.solubility_H2O = 'totally_not_a_real_law'
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # Warning text must name the missing model.
    assert any('totally_not_a_real_law' in rec.message for rec in caplog.records), (
        f'expected solubility-name warning, got: {[r.message for r in caplog.records]}'
    )
    # Discrimination guard: the dispatch still completed (the solve
    # ran), so hf_row carries a P_surf entry.
    assert 'P_surf' in hf_row


def test_atmodeller_warns_when_eos_key_unknown(caplog):
    """A configured real-gas EOS name not in atmodeller's eos registry
    must log a warning ('using ideal gas') and continue.

    Edge: same shape as the solubility case but the warning text
    differs; pin both the gas name and the 'ideal gas' fallback
    string so future renames of the message do not silently lose
    discrimination.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.atmodeller.eos_H2O = 'phantom_eos_v999'
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output()
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    eos_warnings = [r.message for r in caplog.records if 'phantom_eos_v999' in r.message]
    assert eos_warnings, 'expected EOS-name warning for phantom_eos_v999'
    assert any('ideal gas' in m for m in eos_warnings), (
        'EOS warning should mention the ideal-gas fallback'
    )


def test_atmodeller_clamps_t_magma_to_floor(caplog):
    """When ``T_magma < config.outgas.T_floor`` the solve runs at the floor.

    The outgassing temperature is clamped from below (the same
    semantics as the calliope entry point), not skipped: the Planet
    state must carry T_floor, the solver must be invoked, and the clip
    warning must fire. Discriminating: a regression back to the skip
    behavior would never call solve; a regression that dropped the
    clamp would build the Planet at the raw 1800 K.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    config.outgas.T_floor = 2000.0  # above the hf_row T_magma of 1800 K
    hf_row = _earth_hf_row()

    planet_kwargs = {}

    def fake_planet(**kwargs):
        planet_kwargs.update(kwargs)
        return MagicMock()

    class SolveAttempted(RuntimeError):
        pass

    fake_model = MagicMock()
    fake_model.solve.side_effect = SolveAttempted('solver invoked')
    with (
        caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
        patch('atmodeller.Planet', side_effect=fake_planet),
        pytest.raises(SolveAttempted),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    # The solve was attempted (not skipped) and the planet state was
    # built at the clamped floor temperature, not the raw T_magma.
    assert fake_model.solve.call_count == 1
    assert planet_kwargs['temperature'] == pytest.approx(2000.0, rel=1e-12)
    assert planet_kwargs['temperature'] != pytest.approx(1800.0, rel=1e-3)
    # The clip warning fired.
    assert any('clipped' in rec.message for rec in caplog.records)


def test_atmodeller_propagates_solver_exception(caplog):
    """If atmodeller's ``model.solve(...)`` raises, the wrapper must
    log the failure and re-raise (lines 277-279). The caller is
    responsible for the abort + status-file update.

    Discriminating: pin both that the original exception bubbles up
    (a regression that swallowed it would mask the failure mode) and
    that the log message names the failure.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    dirs = {'output': '/tmp/test'}
    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row()
    fake_model = MagicMock()
    fake_model.solve.side_effect = RuntimeError('contrived solver failure')
    with (
        caplog.at_level(logging.ERROR, logger='fwl.proteus.outgas.atmodeller'),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
        pytest.raises(RuntimeError, match='contrived solver failure'),
    ):
        calc_surface_pressures_atmodeller(dirs, config, hf_row)
    assert any('Atmodeller solve failed' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Noble gas wiring (mocked solver)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_noble_gas_passed_as_mass_constraint_and_uses_total_mass_output():
    """A helium inventory reaches atmodeller as its own element mass
    constraint, and the wrapper reads the per-species reservoir masses from
    atmodeller's conserved total_mass output rather than the pressure formula.

    Discriminating: the mass constraint carries the exact He budget; the
    tracked He total equals atmodeller's total_mass and the atmospheric mass
    falls back to total minus dissolved when the output omits gas_mass, so
    the pressure-derived value (which would misweight the light noble gas)
    is not used.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)
    hf_row['He_kg_total'] = 3.0e16  # a helium inventory switches He on
    for gas in noble_gases:
        hf_row[f'{gas}_bar'] = 0.0

    out = _fake_atmodeller_output(log10dIW=-0.5)
    # He partial pressure plus atmodeller's conserved masses for helium.
    ql = dict(out.quick_look.return_value)
    ql['He_g'] = np.array(0.02)
    out.quick_look.return_value = ql
    ad = dict(out.asdict.return_value)
    ad['He_g'] = {'total_mass': np.array(3.0e16), 'dissolved_mass': np.array(1.0e16)}
    out.asdict.return_value = ad

    fake_model = MagicMock()
    fake_model.output = out

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    # He is a mass constraint with the supplied budget.
    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    assert 'He' in mass_kwargs
    assert mass_kwargs['He'] == pytest.approx(3.0e16)

    # The tracked reservoirs come from atmodeller's total_mass (conserved) and
    # dissolved_mass, not the mole-fraction pressure formula.
    assert hf_row['He_kg_total'] == pytest.approx(3.0e16, rel=1e-9)
    assert hf_row['He_kg_liquid'] == pytest.approx(1.0e16, rel=1e-9)
    assert hf_row['He_kg_atm'] == pytest.approx(2.0e16, rel=1e-9)
    assert hf_row['He_kg_atm'] + hf_row['He_kg_liquid'] == pytest.approx(3.0e16, rel=1e-9)
    # Discrimination: the pressure formula p*1e5*area/g at p=0.02 bar would
    # give ~2.6e16 kg for the atmospheric part alone and a total above the
    # 3e16 budget, so the total-mass path is what keeps He conserved.
    assert hf_row['He_kg_total'] < 3.1e16


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_solver_gas_mass_is_authoritative_over_total_minus_dissolved(caplog):
    """When the output carries gas_mass, it wins over total minus dissolved.

    total_mass - dissolved_mass equals the gas-phase mass only while a
    species holds no third reservoir. A solver output where the two
    derivations disagree (the future solid-partitioning case) must book
    the solver's own gas_mass into the atmosphere, keep the difference
    out, and surface the closure gap in the log so an untracked reservoir
    cannot silently inflate the atmospheric budget.

    Discrimination: gas_mass (1.8e16 kg) and total - dissolved (2.0e16 kg)
    differ by 10 sigma of every tolerance used, so passing on the wrong
    convention is impossible.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)
    hf_row['He_kg_total'] = 3.0e16
    for gas in noble_gases:
        hf_row[f'{gas}_bar'] = 0.0

    out = _fake_atmodeller_output(log10dIW=-0.5)
    ql = dict(out.quick_look.return_value)
    ql['He_g'] = np.array(0.02)
    out.quick_look.return_value = ql
    ad = dict(out.asdict.return_value)
    # 0.2e16 kg sits in a reservoir the mapping does not track, so
    # total - dissolved (2.0e16) overstates the gas phase (1.8e16).
    ad['He_g'] = {
        'total_mass': np.array(3.0e16),
        'dissolved_mass': np.array(1.0e16),
        'gas_mass': np.array(1.8e16),
    }
    out.asdict.return_value = ad

    fake_model = MagicMock()
    fake_model.output = out
    with (
        caplog.at_level(logging.WARNING),
        patch('atmodeller.EquilibriumModel', return_value=fake_model),
    ):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    # The solver's gas-phase mass is booked, not the residual.
    assert hf_row['He_kg_atm'] == pytest.approx(1.8e16, rel=1e-9)
    assert hf_row['He_kg_atm'] != pytest.approx(2.0e16, rel=1e-3)
    assert hf_row['He_kg_liquid'] == pytest.approx(1.0e16, rel=1e-9)
    # The closure gap is surfaced.
    assert any('gas-phase closure gap' in r.message for r in caplog.records)


@pytest.mark.unit
def test_no_noble_budget_leaves_atmodeller_species_unchanged():
    """With no noble inventory, no noble gas is added to the species network
    or the mass constraints, so a run without a noble budget is unaffected by
    the noble gas wiring.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)  # no noble budget

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-0.5)

    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    mass_kwargs = fake_model.solve.call_args.kwargs['mass_constraints']
    for gas in noble_gases:
        assert gas not in mass_kwargs
        assert hf_row.get(f'{gas}_kg_total', 0.0) == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_atmodeller_restores_escape_owned_noble_total():
    """The noble gas element total is escape-owned, so after the solve the
    wrapper restores the pre-solve total rather than the solver's total_mass
    (which drifts by the solver residual). The atmosphere and melt split still
    come from the solve (here via the total minus dissolved fallback, since
    the fixture omits gas_mass), so the reservoirs stay consistent.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)
    hf_row['He_kg_total'] = 3.0e16  # escape-authoritative pre-solve value
    for gas in noble_gases:
        hf_row[f'{gas}_bar'] = 0.0

    out = _fake_atmodeller_output(log10dIW=-0.5)
    ql = dict(out.quick_look.return_value)
    ql['He_g'] = np.array(0.02)
    out.quick_look.return_value = ql
    ad = dict(out.asdict.return_value)
    # Solver total_mass drifts slightly above the constraint (its residual).
    ad['He_g'] = {'total_mass': np.array(3.03e16), 'dissolved_mass': np.array(1.0e16)}
    out.asdict.return_value = ad

    fake_model = MagicMock()
    fake_model.output = out
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    # Restored to the pre-solve value, not the drifted solver total.
    assert hf_row['He_kg_total'] == pytest.approx(3.0e16, rel=1e-12)
    assert hf_row['He_kg_total'] != pytest.approx(3.03e16, rel=1e-3)
    # The atmosphere and melt split still come from the solve.
    assert hf_row['He_kg_liquid'] == pytest.approx(1.0e16, rel=1e-9)
    assert hf_row['He_kg_atm'] == pytest.approx(3.03e16 - 1.0e16, rel=1e-9)


@pytest.mark.unit
def test_atmodeller_clears_stale_reservoir_for_inactive_noble():
    """A noble gas that escape has driven to zero is not in the solve, so its
    stale atmospheric reservoir from a prior iteration is cleared rather than
    resurrected into its element total.
    """
    from proteus.outgas.atmodeller import calc_surface_pressures_atmodeller

    config = _make_from_o_budget_config()
    hf_row = _earth_hf_row(O_kg_total=1.0e22)
    hf_row['He_kg_total'] = 0.0  # fully escaped
    hf_row['He_kg_atm'] = 5.0e15  # stale from a prior iteration
    hf_row['He_kg_liquid'] = 0.0
    hf_row['He_bar'] = 5.0  # stale partial pressure
    hf_row['He_vmr'] = 0.09  # stale mixing ratio from a prior iteration

    fake_model = MagicMock()
    fake_model.output = _fake_atmodeller_output(log10dIW=-0.5)
    with patch('atmodeller.EquilibriumModel', return_value=fake_model):
        calc_surface_pressures_atmodeller({'output': '/tmp/test'}, config, hf_row)

    # He was not in the solve; its stale reservoir is cleared and the
    # escape-owned total stays zero rather than being resurrected.
    assert hf_row['He_kg_total'] == 0.0
    assert hf_row['He_kg_atm'] == 0.0
    assert hf_row['He_bar'] == 0.0
    # The mixing ratio is cleared too, so a fully-escaped noble gas does not
    # leak a stale partial pressure into the mean molar mass.
    assert hf_row['He_vmr'] == 0.0
    # atm_kg_per_mol reflects only the reactive gases (helium contributes none).
    assert hf_row['atm_kg_per_mol'] > 0.0
