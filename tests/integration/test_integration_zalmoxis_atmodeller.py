"""Integration test: zalmoxis (real interior structure) coupled to
atmodeller (real outgassing, Bower+2025 ApJ 995:59).

atmodeller is the JAX-based alternate to CALLIOPE for magma-
atmosphere equilibrium: it supplies real-gas EOS, non-ideal
solubility laws, and condensation. The (zalmoxis, atmodeller) pair
exercises the same dry-mass propagation contract as the (zalmoxis,
calliope) pair (issue #677: whole-planet ``element_list`` includes
O so the Zalmoxis dry-mass subtraction reserves space for the O
atmodeller places in H2O, CO2, SO2, etc.), but through the
atmodeller-specific from_O_budget derivation of ``fO2_shift_IW_derived``.

Integration-tier scope:

- The (interior_struct=zalmoxis, outgas=atmodeller) pair round-
  trips through Config. atmodeller lands on its documented defaults
  (solver_mode='robust', condensates on); Zalmoxis lands on its
  production defaults.
- The atmodeller-side enum (``solver_mode``) and gt(0) boundaries
  (``solver_max_steps``, ``solver_multistart``) are re-pinned under
  this pair. Both gt(0) boundaries get the both-edge selectivity
  pattern (=0 rejects, =1 round-trips) so a regression that
  loosened the floor in either direction surfaces.
- The ``none_if_none`` converter on Atmodeller's ``eos_*`` AND
  ``solubility_*`` field families is case-sensitive at the field
  layer. Under this pair, exercise both families with both lowercase
  ('none' coerces to None) and uppercase variants (pass through).
- Field-count + name-set guards on Atmodeller's
  converter-bearing families: 7 ``solubility_*``, 5 ``eos_*``. A
  silent rename surfaces.
- ``Atmodeller.include_condensates`` defaults True; both states
  round-trip.
- The Zalmoxis ``equilibrate_init`` path also applies to atmodeller:
  the IC reconciliation loop reads ``hf_row[e + '_kg_total']``
  populated by atmodeller and feeds it to the Zalmoxis dry-mass
  subtraction. Pin the equilibrate_max_iter and equilibrate_tol
  validators under this pair so a regression that broke the floors
  surfaces here in addition to the calliope pair.
- The wrapper merge contract: atmodeller's from_O_budget columns
  (``fO2_shift_IW_derived``, ``O_res``) and the per-gas pressures
  (``H2O_bar``, ``CO2_bar``, ``H2_bar``, ``CO_bar``, ``N2_bar``)
  are registered in ``GetHelpfileKeys``; the per-element total
  columns Zalmoxis reads (``H_kg_total``, ``O_kg_total``, etc.)
  are also pinned.

atmodeller is an optional dependency. Module-top
``pytest.importorskip('atmodeller')`` follows the existing pattern
so Docker ``--no-deps`` builds do not fail collection.

The full two-timestep coupled run with real Aragog + real
atmodeller + dummy Zalmoxis lives at the slow tier in
``test_slow_aragog_atmodeller.py``.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytest.importorskip('atmodeller')

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs():
    """Base Config kwargs for the (zalmoxis, atmodeller) combination.

    ``atmos_clim='dummy'`` and ``escape='dummy'`` keep the test
    surface focused on the interior-outgas coupling. ``star='dummy'``
    avoids the spada_zephyrus cross-validator and the MORS
    rotation-constraints branch.
    """
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    return dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        escape=Escape(module='dummy'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


# ---------------------------------------------------------------------------
# Pair round-trip.
# ---------------------------------------------------------------------------


def test_zalmoxis_atmodeller_round_trips_through_config():
    """The (interior_struct=zalmoxis, outgas=atmodeller) pair round-
    trips through Config without raising. atmodeller lands on its
    documented defaults (solver_mode='robust', condensates True);
    Zalmoxis lands on its production defaults (newton outer solver,
    equilibrate_init True).
    """
    from proteus.config import Config
    from proteus.config._outgas import Outgas
    from proteus.config._struct import Struct

    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        outgas=Outgas(module='atmodeller'),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.module == 'zalmoxis'
    assert cfg.outgas.module == 'atmodeller'
    assert cfg.outgas.atmodeller.solver_mode == 'robust'
    assert cfg.outgas.atmodeller.include_condensates is True
    assert cfg.interior_struct.zalmoxis.equilibrate_init is True


# ---------------------------------------------------------------------------
# Atmodeller-side schema (re-pinned for the zalmoxis pair).
# ---------------------------------------------------------------------------


def test_atmodeller_solver_mode_enum_pinned_as_set_under_zalmoxis_pair():
    """Pin ``Atmodeller.solver_mode`` as ``{'robust', 'basic'}``. Set
    equality catches a regression that silently added a third mode
    while keeping the documented two round-tripping.
    """
    import attrs

    from proteus.config._outgas import Atmodeller

    allowed = attrs.fields(Atmodeller).solver_mode.validator.options
    assert set(allowed) == {'robust', 'basic'}, (
        f'Atmodeller.solver_mode enum drifted from documented set: {allowed}'
    )
    for known in ('robust', 'basic'):
        a = Atmodeller(solver_mode=known)
        assert a.solver_mode == known
    with pytest.raises(ValueError, match=r'(?i)solver_mode'):
        Atmodeller(solver_mode='turbo')
    assert Atmodeller().solver_mode == 'robust'


def test_atmodeller_solver_max_steps_and_multistart_gt0_under_zalmoxis():
    """``Atmodeller.solver_max_steps`` and ``solver_multistart`` use
    ``gt(0)``. Pin BOTH boundaries of the accepted range under this
    pair:

    - ``=0`` rejects (catches a ``ge(0)`` regression).
    - ``=1`` round-trips (catches a ``gt(1)`` regression).
    - Documented defaults (256, 10) round-trip.
    """
    from proteus.config._outgas import Atmodeller

    for bad_val in (0, -1):
        with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
            Atmodeller(solver_max_steps=bad_val)
        with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
            Atmodeller(solver_multistart=bad_val)
    a_min = Atmodeller(solver_max_steps=1, solver_multistart=1)
    assert a_min.solver_max_steps == 1
    assert a_min.solver_multistart == 1
    default = Atmodeller()
    assert default.solver_max_steps == 256
    assert default.solver_multistart == 10


def test_atmodeller_include_condensates_default_true_round_trips_off():
    """``Atmodeller.include_condensates`` defaults True; both states
    round-trip. A regression that flipped the default would silently
    drop condensation from the production runs.
    """
    from proteus.config._outgas import Atmodeller

    assert Atmodeller().include_condensates is True
    a_off = Atmodeller(include_condensates=False)
    assert a_off.include_condensates is False


def test_atmodeller_none_sentinel_case_sensitive_on_both_field_families():
    """``none_if_none`` is case-sensitive on Atmodeller's ``eos_*``
    AND ``solubility_*`` field families. Lowercase ``'none'``
    coerces to Python None on BOTH families; uppercase ``'None'``
    and ``'NONE'`` pass through on BOTH families.

    Discrimination: a regression that broadened the converter on
    only one family (e.g. just eos_*) to case-insensitive would
    slip past a test that exercised only the other family. The
    test here re-pins the contract under the zalmoxis pair so the
    family-level coverage holds across multiple pair files.
    """
    from proteus.config._outgas import Atmodeller

    # Lowercase sentinel coerces on each family.
    assert Atmodeller(eos_H2O='none').eos_H2O is None
    assert Atmodeller(eos_CO2='none').eos_CO2 is None
    assert Atmodeller(solubility_CO='none').solubility_CO is None
    assert Atmodeller(solubility_CH4='none').solubility_CH4 is None
    # Uppercase passes through on each family.
    for non_sentinel in ('None', 'NONE'):
        a_eos = Atmodeller(eos_H2='none' if non_sentinel == '' else non_sentinel)
        assert a_eos.eos_H2 == non_sentinel
        a_sol = Atmodeller(solubility_N2=non_sentinel)
        assert a_sol.solubility_N2 == non_sentinel


def test_atmodeller_eos_and_solubility_field_counts_under_zalmoxis_pair():
    """Pin both the COUNT and the documented NAMES of Atmodeller's
    converter-bearing field families: 7 ``solubility_*`` and 5
    ``eos_*``. Re-pinned under the zalmoxis pair so a silent rename
    that slipped past the atmodeller_* pair tests would surface
    here.
    """
    import attrs

    from proteus.config._outgas import Atmodeller

    fields = attrs.fields(Atmodeller)
    solubility_fields = {f.name for f in fields if f.name.startswith('solubility_')}
    eos_fields = {f.name for f in fields if f.name.startswith('eos_')}
    assert solubility_fields == {
        'solubility_H2O',
        'solubility_CO2',
        'solubility_H2',
        'solubility_N2',
        'solubility_S2',
        'solubility_CO',
        'solubility_CH4',
    }
    assert eos_fields == {
        'eos_H2O',
        'eos_CO2',
        'eos_H2',
        'eos_CH4',
        'eos_CO',
    }
    assert len(solubility_fields) == 7
    assert len(eos_fields) == 5


# ---------------------------------------------------------------------------
# Zalmoxis equilibrate path under atmodeller.
# ---------------------------------------------------------------------------


def test_zalmoxis_equilibrate_init_and_iter_validators_under_atmodeller():
    """The CALLIOPE / atmodeller IC reconciliation loop fires under
    ``equilibrate_init=True``. ``equilibrate_max_iter`` ``ge(1)``
    and ``equilibrate_tol`` ``gt(0)`` gate the loop's termination.
    Re-pin both edges under the atmodeller pair so a regression
    that broke a floor surfaces here as well as in the calliope
    pair.
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis().equilibrate_init is True
    assert Zalmoxis(equilibrate_init=False).equilibrate_init is False
    assert Zalmoxis(equilibrate_max_iter=1).equilibrate_max_iter == 1
    with pytest.raises(ValueError, match=r'(?i)equilibrate_max_iter'):
        Zalmoxis(equilibrate_max_iter=0)
    for bad_val in (0.0, -1e-6):
        with pytest.raises(ValueError, match=r'(?i)equilibrate_tol'):
            Zalmoxis(equilibrate_tol=bad_val)
    z = Zalmoxis(equilibrate_tol=1e-12)
    assert z.equilibrate_tol == pytest.approx(1e-12, rel=1e-12)


# ---------------------------------------------------------------------------
# Whole-element propagation under atmodeller.
# ---------------------------------------------------------------------------


def test_atmodeller_whole_element_totals_in_hf_row_schema_under_zalmoxis():
    """Issue #677: the Zalmoxis dry-mass subtraction reads
    ``hf_row[e + '_kg_total']`` for every element in
    ``element_list``. atmodeller writes these columns at every
    outgas call; the schema MUST register them. Re-pinned under the
    atmodeller pair so a regression that dropped a column from
    GetHelpfileKeys is caught regardless of which outgas backend is
    in play.

    Discrimination: per-element check so a regression that dropped
    one element's total fails the specific assertion. The volatile
    species (H, O, C, N, S) and the rock-forming species (Si, Mg,
    Fe, Na) are pinned together because the dry-mass subtraction
    treats them uniformly.
    """
    from proteus.utils.constants import element_list
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    assert set(element_list) == {'H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'}
    keys = GetHelpfileKeys()
    for e in element_list:
        col = f'{e}_kg_total'
        assert col in keys, f'{col} missing from hf_row schema'
    row = ZeroHelpfileRow()
    for e in element_list:
        col = f'{e}_kg_total'
        assert row[col] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[col], float)


# ---------------------------------------------------------------------------
# Wrapper-merge contract: atmodeller from_O_budget columns + per-gas pressures.
# ---------------------------------------------------------------------------


def test_atmodeller_from_o_budget_and_pressure_keys_under_zalmoxis_pair():
    """atmodeller's from_O_budget derives ``fO2_shift_IW_derived`` from the
    O budget and writes it to ``hf_row``. ``O_res`` is the residual
    O the solver could not place into condensed phases. Per-gas
    partial pressures (``<species>_bar``) flow into the surface-
    pressure target the Zalmoxis structure solver reads.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    from_o_budget_keys = ('fO2_shift_IW_derived', 'O_res')
    pressure_keys = ('H2O_bar', 'CO2_bar', 'H2_bar', 'CO_bar', 'N2_bar')
    for key in from_o_budget_keys + pressure_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in from_o_budget_keys + pressure_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
