"""Integration test: atmodeller (real outgas) coupled to MORS (real star).

atmodeller (Bower+2025, ApJ 995:59) supplies the per-iteration
JAX-based volatile partitioning with real-gas EOS, non-ideal
solubility laws, and condensation; MORS supplies the
time-evolving stellar spectrum and bolometric flux. The pair
sits inside any PROTEUS run that selects ``outgas='atmodeller'``
with a real star; this file pins the schema, helper, and
helpfile contracts at the integration tier without booting the
JAX solver or downloading a MORS spectrum.

Integration-tier scope:

- Schema validators round-trip ``outgas.module='atmodeller'``
  with ``star.module='mors'``.
- Atmodeller's ``solver_mode`` enum is pinned as a set
  ``{'robust', 'basic'}`` so a regression that silently added a
  third mode surfaces here.
- ``Atmodeller.solver_max_steps`` and ``solver_multistart``
  enforce ``gt(0)`` at the attrs validator layer.
- The ``none_if_none`` converter on the ``eos_*`` and
  ``solubility_*`` fields is case-sensitive: lowercase 'none'
  coerces to Python None, while 'None' / 'NONE' pass through.
- The number of ``solubility_*`` fields (7) and ``eos_*`` fields
  (5) is pinned via ``attrs.fields(Atmodeller)`` so a regression
  that silently added an eighth solubility law or a sixth EOS
  table fails the count check.
- Mors's ``tracks`` and ``spectrum_source`` enums are pinned as
  sets so a third value added without changing the round-trip
  surface is caught immediately.
- ``valid_mors`` rejects ``age_now <= 0`` and ``age_now is None``
  when the star module is 'mors'.
- ``valid_mors`` rotation-constraints branch (``_star.py:26-38``)
  is pinned end-to-end: both rot fields set raises, neither set
  raises, negative period raises, percentile out of ``[0, 100]``
  raises.
- The wrapper merge guard pins atmodeller from_O_budget columns
  (``fO2_shift_IW_derived``, ``O_res``), per-gas pressures, AND
  MORS stellar columns (``T_star``, ``R_star``, ``M_star``,
  ``F_ins``, ``F_xuv``) in ``GetHelpfileKeys`` so per-iteration
  values flow into the helpfile.

atmodeller is an optional dependency; the module-top
``pytest.importorskip('atmodeller')`` follows the existing
pattern in ``test_integration_agni_atmodeller.py``.

The full two-timestep atmodeller + MORS coupled run is exercised
by the slow-tier ``test_slow_aragog_atmodeller.py`` (with aragog
interior) for the atmodeller leg; the MORS leg is exercised by
``test_smoke_modules.py``.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

pytest.importorskip('atmodeller')

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (atmodeller, mors) production combo.
# ---------------------------------------------------------------------------


def test_outgas_atmodeller_and_star_mors_both_round_trip_through_schema():
    """``outgas.module='atmodeller'`` and ``star.module='mors'``
    both round-trip without raising.

    Discrimination: reject an obviously-wrong outgas module name to
    confirm the validator still fires (rules out a regression that
    disabled validation entirely), and confirm the Mors default
    Star round-trips so a regression in ``valid_mors`` that broke
    the default config would surface here.
    """
    from proteus.config._outgas import Outgas
    from proteus.config._star import Star

    o = Outgas(module='atmodeller')
    assert o.module == 'atmodeller'
    # Atmodeller default solver parameters round-trip.
    assert o.atmodeller.solver_mode == 'robust'
    # An invalid outgas module rejects.
    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='not_a_real_outgas_backend')
    # Mors default Star round-trips with documented defaults.
    s = Star(module='mors')
    assert s.module == 'mors'
    assert s.mors.tracks == 'spada'
    assert s.mors.spectrum_source == 'phoenix'


# ---------------------------------------------------------------------------
# Atmodeller-side schema (re-pinned for this pair).
# ---------------------------------------------------------------------------


def test_atmodeller_solver_mode_enum_pinned_as_set_under_mors_pair():
    """Pin the ``Atmodeller.solver_mode`` enum as the set
    ``{'robust', 'basic'}``. A regression that silently added a
    third mode (e.g. 'experimental') would still let 'robust' and
    'basic' round-trip and would still reject an obvious typo, so
    set equality is the only way to catch enum drift.
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
    # Documented default is 'robust' (slower compile, better convergence).
    assert Atmodeller().solver_mode == 'robust'


def test_atmodeller_solver_step_and_multistart_must_be_positive_under_mors_pair():
    """``Atmodeller.solver_max_steps`` and ``solver_multistart`` use
    ``gt(0)`` at ``_outgas.py:112-113``. Defaults round-trip; zero
    and negative reject. A regression that swapped ``gt(0)`` for
    ``ge(0)`` would accept zero, so the explicit zero-raise
    discriminates.
    """
    from proteus.config._outgas import Atmodeller

    with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
        Atmodeller(solver_max_steps=0)
    with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
        Atmodeller(solver_max_steps=-1)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=0)
    with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
        Atmodeller(solver_multistart=-1)
    # Adjacent-valid round-trip on the smallest accepted integer
    # (1) pins the gt(0) vs ge(0) boundary: ge(0) accepts 0, gt(0)
    # accepts 1 but not 0. The single-step / single-restart
    # configuration is unrealistic for production but well-defined
    # at the schema layer.
    a_min_steps = Atmodeller(solver_max_steps=1)
    assert a_min_steps.solver_max_steps == 1
    a_min_multi = Atmodeller(solver_multistart=1)
    assert a_min_multi.solver_multistart == 1
    default = Atmodeller()
    # Pin the documented defaults so a silent shift surfaces here.
    assert default.solver_max_steps == 1024
    assert default.solver_multistart == 10


def test_atmodeller_none_sentinel_coerced_case_sensitively_under_mors_pair():
    """The ``none_if_none`` converter on the ``eos_*`` and
    ``solubility_*`` fields is case-sensitive: lowercase 'none'
    coerces to Python None, uppercase 'None' / 'NONE' pass through
    as literal strings.

    Discrimination: a regression that dropped the converter would
    leave the lowercase sentinel as 'none' and break downstream
    atmodeller dispatch (which checks against Python None). A
    regression that broadened the converter to case-insensitive
    would coerce 'None' too, changing the documented contract.
    """
    from proteus.config._outgas import Atmodeller

    # Lowercase sentinel coerces on an eos_* field.
    a_lower_eos = Atmodeller(eos_H2O='none')
    assert a_lower_eos.eos_H2O is None
    # And on a solubility_* field.
    a_lower_sol = Atmodeller(solubility_CO='none')
    assert a_lower_sol.solubility_CO is None
    # Uppercase variants pass through on BOTH eos_* and solubility_*
    # fields. A regression that broadened only one of the two
    # converters to case-insensitive would otherwise slip through if
    # the test covered only one field family.
    for non_sentinel in ('None', 'NONE'):
        a_pt_eos = Atmodeller(eos_H2O=non_sentinel)
        assert a_pt_eos.eos_H2O == non_sentinel, (
            f'{non_sentinel!r} should pass through eos_*; got {a_pt_eos.eos_H2O!r}'
        )
        a_pt_sol = Atmodeller(solubility_CO=non_sentinel)
        assert a_pt_sol.solubility_CO == non_sentinel, (
            f'{non_sentinel!r} should pass through solubility_*; got {a_pt_sol.solubility_CO!r}'
        )
    # Non-sentinel string also passes through unchanged on both.
    a_real_eos = Atmodeller(eos_H2O='SHV_CORK')
    assert a_real_eos.eos_H2O == 'SHV_CORK'
    a_real_sol = Atmodeller(solubility_CO='CO_basalt_yoshioka19')
    assert a_real_sol.solubility_CO == 'CO_basalt_yoshioka19'


def test_atmodeller_eos_and_solubility_field_counts_under_mors_pair():
    """Pin the count of converter-bearing fields on Atmodeller so a
    regression that silently adds an eighth solubility law (e.g.
    ``solubility_O2``) or a sixth EOS table fails at the schema
    layer.

    Documented from ``_outgas.py:115-128``: 7 ``solubility_*``
    fields (H2O, CO2, H2, N2, S2, CO, CH4) and 5 ``eos_*`` fields
    (H2O, CO2, H2, CH4, CO).
    """
    import attrs

    from proteus.config._outgas import Atmodeller

    fields = attrs.fields(Atmodeller)
    solubility_fields = [f.name for f in fields if f.name.startswith('solubility_')]
    eos_fields = [f.name for f in fields if f.name.startswith('eos_')]
    assert len(solubility_fields) == 7, (
        f'Expected 7 solubility_* fields on Atmodeller, '
        f'got {len(solubility_fields)}: {solubility_fields}'
    )
    assert len(eos_fields) == 5, (
        f'Expected 5 eos_* fields on Atmodeller, got {len(eos_fields)}: {eos_fields}'
    )
    # Discrimination: pin the documented names so a silent rename
    # (e.g. solubility_CO -> solubility_carbon_monoxide) fails here.
    assert set(solubility_fields) == {
        'solubility_H2O',
        'solubility_CO2',
        'solubility_H2',
        'solubility_N2',
        'solubility_S2',
        'solubility_CO',
        'solubility_CH4',
    }
    assert set(eos_fields) == {
        'eos_H2O',
        'eos_CO2',
        'eos_H2',
        'eos_CH4',
        'eos_CO',
    }


# ---------------------------------------------------------------------------
# MORS-side schema (re-pinned for this pair).
# ---------------------------------------------------------------------------


def test_mors_tracks_and_spectrum_source_enums_pinned_as_sets_under_atmodeller_pair():
    """Pin ``Mors.tracks`` as ``{'spada', 'baraffe'}`` and
    ``Mors.spectrum_source`` as ``{'solar', 'muscles', 'phoenix',
    None}``. Set equality catches regressions that silently added
    a third value while keeping the original two round-tripping.
    """
    import attrs

    from proteus.config._star import Mors

    tracks_allowed = attrs.fields(Mors).tracks.validator.options
    assert set(tracks_allowed) == {'spada', 'baraffe'}, (
        f'Mors.tracks enum drifted from documented set: {tracks_allowed}'
    )
    for known in ('spada', 'baraffe'):
        m = Mors(tracks=known)
        assert m.tracks == known
    with pytest.raises(ValueError, match=r'(?i)tracks'):
        Mors(tracks='not_a_real_track')

    src_allowed = attrs.fields(Mors).spectrum_source.validator.options
    assert set(src_allowed) == {'solar', 'muscles', 'phoenix', None}, (
        f'Mors.spectrum_source enum drifted from documented set: {src_allowed}'
    )
    default = Mors()
    assert default.tracks == 'spada'
    assert default.spectrum_source == 'phoenix'
    assert default.age_now == pytest.approx(4.567, rel=1e-12)


def test_mors_age_now_positivity_at_valid_mors_layer_under_atmodeller_pair():
    """``valid_mors`` at ``_star.py:13-14`` raises when
    ``mors.age_now`` is None or <=0. The check runs at the Star
    cross-validator layer, so the attrs field default (4.567 Gyr)
    must be replaced explicitly to trip it.
    """
    from proteus.config._star import Mors, Star

    for bad in (0.0, -1.0, None):
        with pytest.raises(ValueError, match=r'(?i)age_now'):
            Star(module='mors', mors=Mors(age_now=bad))
    # Positive default round-trips.
    s = Star(module='mors', mors=Mors(age_now=4.567))
    assert s.mors.age_now == pytest.approx(4.567, rel=1e-12)


def test_valid_mors_rotation_constraints_under_atmodeller_pair():
    """``valid_mors`` at ``_star.py:26-38`` enforces "exactly one
    of ``rot_pcntle`` / ``rot_period`` is set", a strictly positive
    period when only the period is set, and a percentile in
    ``[0, 100]`` when only the percentile is set.

    Edge: pin all four rotation-related branches.

    1. Both ``rot_pcntle`` and ``rot_period`` set: collision.
    2. Neither set: missing.
    3. Negative period: invalid value.
    4. Percentile out of ``[0, 100]``: invalid value (both edges).

    The documented defaults (``rot_pcntle=50.0``,
    ``rot_period=None``) satisfy the "exactly one set" rule; the
    positive round-trip catches a regression that flipped them.
    """
    from proteus.config._star import Mors, Star

    # Both set: collision.
    with pytest.raises(ValueError, match=r'(?i)rotation'):
        Star(
            module='mors',
            mors=Mors(spectrum_source='phoenix', rot_pcntle=50.0, rot_period=10.0),
        )
    # Neither set: missing.
    with pytest.raises(ValueError, match=r'(?i)rotation'):
        Star(
            module='mors',
            mors=Mors(spectrum_source='phoenix', rot_pcntle=None, rot_period=None),
        )
    # Negative period: invalid value.
    with pytest.raises(ValueError, match=r'(?i)period'):
        Star(
            module='mors',
            mors=Mors(spectrum_source='phoenix', rot_pcntle=None, rot_period=-1.0),
        )
    # Percentile out of [0, 100]: pin both edges.
    with pytest.raises(ValueError, match=r'(?i)percentile'):
        Star(
            module='mors',
            mors=Mors(spectrum_source='phoenix', rot_pcntle=-1.0),
        )
    with pytest.raises(ValueError, match=r'(?i)percentile'):
        Star(
            module='mors',
            mors=Mors(spectrum_source='phoenix', rot_pcntle=101.0),
        )
    # Documented defaults round-trip.
    s_ok = Star(module='mors', mors=Mors(spectrum_source='phoenix'))
    assert s_ok.mors.rot_pcntle == pytest.approx(50.0, rel=1e-12)
    assert s_ok.mors.rot_period is None


# ---------------------------------------------------------------------------
# Wrapper-merge contract: from_O_budget atmodeller + per-gas + MORS stellar columns.
# ---------------------------------------------------------------------------


def test_atmodeller_mors_helpfile_keys_register_from_o_budget_and_stellar_columns():
    """The wrapper merge propagates per-iteration columns from both
    sides into ``hf_row``. The schema MUST register:

    - from_O_budget atmodeller columns: ``fO2_shift_IW_derived``, ``O_res``.
    - Per-gas pressures: ``H2O_bar``, ``CO2_bar``, ``H2_bar``,
      ``CO_bar``, ``N2_bar``.
    - MORS stellar columns: ``T_star``, ``R_star``, ``M_star``,
      ``F_ins``, ``F_xuv``.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as a float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    atmodeller_from_o_budget_keys = ('fO2_shift_IW_derived', 'O_res')
    pressure_keys = ('H2O_bar', 'CO2_bar', 'H2_bar', 'CO_bar', 'N2_bar')
    stellar_keys = ('T_star', 'R_star', 'M_star', 'F_ins', 'F_xuv')
    for key in atmodeller_from_o_budget_keys + pressure_keys + stellar_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in atmodeller_from_o_budget_keys + pressure_keys + stellar_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); relative form of
        # pytest.approx is undefined at zero, so use absolute.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
