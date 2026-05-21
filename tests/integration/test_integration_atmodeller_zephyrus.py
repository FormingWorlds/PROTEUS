"""Integration test: atmodeller (real outgas) coupled to ZEPHYRUS (real escape).

atmodeller (Bower+2025, ApJ 995:59) supplies the per-iteration
JAX-based volatile partitioning with real-gas EOS, non-ideal
solubility laws, and condensation; ZEPHYRUS supplies the
atmospheric escape leg. ZEPHYRUS is hard-coupled to MORS via the
``spada_zephyrus`` cross-validator at
``src/proteus/config/_config.py:25-31``: a Config with
``escape.module='zephyrus'`` MUST also have ``star.module='mors'``
AND ``star.mors.tracks='spada'``. The atmodeller x ZEPHYRUS pair
therefore requires star=MORS+spada to round-trip.

Integration-tier scope:

- Schema validators round-trip ``outgas.module='atmodeller'``
  with ``escape.module='zephyrus'`` when paired with
  ``star.module='mors'`` and ``mors.tracks='spada'``.
- ``spada_zephyrus`` rejects the same Config with
  ``star.module='dummy'`` or ``mors.tracks='baraffe'``; both
  pin the gate for the atmodeller leg.
- Escape module enum is pinned as ``{None, 'dummy', 'zephyrus',
  'boreas'}`` so a regression that silently added a fifth backend
  surfaces here.
- Atmodeller's ``solver_mode`` enum is pinned as ``{'robust',
  'basic'}``. ``solver_max_steps`` and ``solver_multistart``
  enforce ``gt(0)`` with the boundary round-trip at the minimum
  accepted integer.
- The ``none_if_none`` converter on both the ``eos_*`` and
  ``solubility_*`` field families is case-sensitive: lowercase
  'none' coerces to Python None on each family, uppercase
  variants pass through on each family.
- Field counts on Atmodeller are pinned: 7 ``solubility_*`` and
  5 ``eos_*``, with the documented name sets.
- ``Zephyrus.Pxuv`` validator-layer split: field-level ``ge(0)``
  rejects strictly-negative AND round-trips an adjacent-valid
  small-positive value; cross-validator ``valid_zephyrus``
  rejects ``Pxuv = 0`` (open lower) and ``Pxuv > 10`` (closed
  upper).
- ``Zephyrus.efficiency`` closed-interval endpoints round-trip;
  field-level ``le(1)`` rejects the just-above-unit case AND
  round-trips an adjacent-valid below-unit value.
- The wrapper merge guard pins atmodeller Path-C columns
  (``fO2_shift_IW_derived``, ``O_res``), per-gas pressures, AND
  zephyrus escape columns (``esc_rate_total``,
  ``esc_kg_cumulative``, ``M_vol_initial``) in
  ``GetHelpfileKeys`` so per-iteration values flow into the
  helpfile.

atmodeller is an optional dependency; the module-top
``pytest.importorskip('atmodeller')`` follows the existing
pattern.

The full two-timestep coupled run with both real solvers and a
real atmosphere is exercised by the slow-tier
``test_slow_aragog_atmodeller.py`` (atmodeller leg with aragog
interior) and ``test_integration_mors_zephyrus.py`` (zephyrus
leg with dummy interior).

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
    """Build base Config kwargs for the (atmodeller, zephyrus,
    mors+spada) combination. ``atmos_clim`` is set to 'dummy' so
    the ``janus_escape_atmosphere`` cross-validator at
    ``_config.py:145`` does not fire (that validator gates
    ``escape=zephyrus + atmos_clim=janus + stop.escape.enabled=
    False``); using ``atmos_clim='dummy'`` avoids any latent
    dependency on the ``stop.escape.enabled`` default. The pair
    test does not need AGNI or JANUS in the loop.
    """
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._outgas import Outgas
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star

    return dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        outgas=Outgas(module='atmodeller'),
        star=Star(module='mors'),  # default Mors uses spada + phoenix
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (atmodeller, zephyrus, mors+spada) combo.
# ---------------------------------------------------------------------------


def test_atmodeller_zephyrus_mors_spada_round_trips_through_config():
    """The full hard-coupled triple (outgas=atmodeller,
    escape=zephyrus, star=mors+spada) round-trips through Config
    without raising.

    Discrimination: a regression in either side that broke schema
    construction would surface here. Each module assertion confirms
    the slot landed where expected.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape

    cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert cfg.outgas.module == 'atmodeller'
    assert cfg.escape.module == 'zephyrus'
    assert cfg.star.module == 'mors'
    assert cfg.star.mors.tracks == 'spada'


def test_atmodeller_zephyrus_rejects_dummy_star_at_spada_zephyrus_layer():
    """``spada_zephyrus`` fires when escape=zephyrus is paired with
    a non-MORS star. The atmodeller x zephyrus combination must
    reject the dummy-star configuration even though the atmodeller
    outgas side itself is valid for any star.

    The kwargs use ``atmos_clim='dummy'`` to avoid a latent
    dependency on the ``params.stop.escape.enabled`` default: with
    ``atmos_clim='janus'`` the ``janus_escape_atmosphere``
    cross-validator could fire first when ``stop.escape.enabled``
    is False, masking the ``spada_zephyrus`` rejection we want to
    pin here.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._outgas import Outgas
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    kwargs = dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        outgas=Outgas(module='atmodeller'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)


def test_atmodeller_zephyrus_rejects_baraffe_tracks_at_spada_zephyrus_layer():
    """``spada_zephyrus`` also rejects mors+baraffe even when the
    outgas side is atmodeller. Pinning the baraffe-rejection case
    here so a regression that relaxed the spada-only gate (e.g.
    extended to baraffe as well) surfaces in the atmodeller leg.

    Same ``atmos_clim='dummy'`` choice as the dummy-star rejection
    above: avoids latent dependency on
    ``params.stop.escape.enabled`` defaults.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._outgas import Outgas
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Mors, Star

    kwargs = dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        outgas=Outgas(module='atmodeller'),
        star=Star(module='mors', mors=Mors(tracks='baraffe')),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)


# ---------------------------------------------------------------------------
# Escape module enum pinned as set.
# ---------------------------------------------------------------------------


def test_escape_module_enum_pinned_as_set_under_atmodeller_pair():
    """Pin the Escape.module enum as ``{None, 'dummy', 'zephyrus',
    'boreas'}``. A regression that silently added a fifth escape
    backend would still let the four documented values round-trip
    and would still reject an obvious typo, so set equality is the
    only way to catch enum drift.
    """
    import attrs

    from proteus.config._escape import Escape

    allowed = attrs.fields(Escape).module.validator.options
    assert set(allowed) == {None, 'dummy', 'zephyrus', 'boreas'}, (
        f'Escape.module enum drifted from documented set: {allowed}'
    )
    for known in ('dummy', 'zephyrus', 'boreas'):
        e = Escape(module=known)
        assert e.module == known
    with pytest.raises(ValueError, match=r'(?i)module'):
        Escape(module='not_a_real_escape_backend')


# ---------------------------------------------------------------------------
# Atmodeller-side schema (re-pinned for this pair).
# ---------------------------------------------------------------------------


def test_atmodeller_solver_mode_enum_pinned_as_set_under_zephyrus_pair():
    """Pin the ``Atmodeller.solver_mode`` enum as
    ``{'robust', 'basic'}``. Set equality catches a regression
    that silently added a third mode while keeping the documented
    two round-tripping.
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


def test_atmodeller_solver_step_and_multistart_under_zephyrus_pair():
    """``Atmodeller.solver_max_steps`` and ``solver_multistart``
    use ``gt(0)`` at ``_outgas.py:112-113``.

    Edge: pin BOTH boundaries of the documented accepted range.
    ``=0`` rejects (catches a ``ge(0)`` regression); ``=1`` round-
    trips (catches a ``gt(1)`` regression). Documented defaults
    (256, 10) round-trip.
    """
    from proteus.config._outgas import Atmodeller

    for bad in (0, -1):
        with pytest.raises(ValueError, match=r'(?i)solver_max_steps'):
            Atmodeller(solver_max_steps=bad)
        with pytest.raises(ValueError, match=r'(?i)solver_multistart'):
            Atmodeller(solver_multistart=bad)
    # Adjacent-valid round-trip at the minimum accepted integer
    # pins the gt(0) vs gt(1) regression direction.
    assert Atmodeller(solver_max_steps=1).solver_max_steps == 1
    assert Atmodeller(solver_multistart=1).solver_multistart == 1
    default = Atmodeller()
    assert default.solver_max_steps == 256
    assert default.solver_multistart == 10


def test_atmodeller_none_sentinel_case_sensitive_on_eos_and_solubility_families():
    """The ``none_if_none`` converter on Atmodeller's ``eos_*`` and
    ``solubility_*`` field families is case-sensitive: lowercase
    'none' coerces to Python None on BOTH families; uppercase
    'None' / 'NONE' pass through unchanged on BOTH families.

    Discrimination: cover both family-level passthrough cases. A
    regression that broadened the converter on only one family
    (e.g. solubility's only) to case-insensitive would otherwise
    slip past a test that exercised eos_* alone.
    """
    from proteus.config._outgas import Atmodeller

    # Lowercase sentinel coerces on each family.
    assert Atmodeller(eos_H2O='none').eos_H2O is None
    assert Atmodeller(solubility_CO='none').solubility_CO is None

    # Uppercase passes through on each family.
    for non_sentinel in ('None', 'NONE'):
        a_eos = Atmodeller(eos_H2O=non_sentinel)
        assert a_eos.eos_H2O == non_sentinel
        a_sol = Atmodeller(solubility_CO=non_sentinel)
        assert a_sol.solubility_CO == non_sentinel

    # Real strings pass through on each family.
    assert Atmodeller(eos_H2O='SHV_CORK').eos_H2O == 'SHV_CORK'
    assert (
        Atmodeller(solubility_CO='CO_basalt_yoshioka19').solubility_CO == 'CO_basalt_yoshioka19'
    )


def test_atmodeller_eos_and_solubility_field_counts_under_zephyrus_pair():
    """Pin both the COUNT and the documented NAMES of the converter-
    bearing fields. A silent rename (e.g. ``solubility_CO`` to
    ``solubility_carbon_monoxide``) would fail the set check even
    if the count happened to match.
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
    # Count guards in case the set comparison ever loosened.
    assert len(solubility_fields) == 7
    assert len(eos_fields) == 5


# ---------------------------------------------------------------------------
# Zephyrus-side validator-layer split (Pxuv and efficiency).
# ---------------------------------------------------------------------------


def test_zephyrus_pxuv_cross_validator_rejects_open_lower_and_above_upper_under_atmodeller():
    """``valid_zephyrus`` at ``_escape.py:13-15`` rejects
    ``Pxuv <= 0`` (open lower) and ``Pxuv > 10`` (closed upper).
    The field-level ``ge(0)`` validator does not catch ``Pxuv =
    0``, so the cross-validator owns that case.

    Edge: ``Pxuv = 0.0`` raises at the cross-validator layer,
    ``Pxuv = 10.0`` round-trips (closed upper), ``Pxuv = 10.0001``
    raises (just above upper), the default (5e-5) round-trips.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    # 0 passes field ge(0) but fails the cross check.
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=0.0)),
            **_base_config_kwargs(),
        )
    # Upper boundary 10.0 round-trips.
    cfg = Config(
        escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0)),
        **_base_config_kwargs(),
    )
    assert cfg.escape.zephyrus.Pxuv == pytest.approx(10.0, rel=1e-12)
    # 10.0001 rejects.
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0001)),
            **_base_config_kwargs(),
        )
    # Default 5e-5 round-trips.
    default_cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert default_cfg.escape.zephyrus.Pxuv == pytest.approx(5e-5, rel=1e-12)


def test_zephyrus_pxuv_field_level_rejects_negative_and_round_trips_small_positive():
    """The field-level ``ge(0)`` validator at ``_escape.py:36``
    rejects strictly-negative Pxuv at Zephyrus construction time.
    An adjacent-valid small positive Pxuv round-trips: a
    regression that tightened the field validator to
    ``gt(<floor>)`` would silently reject the documented default
    5e-5; the selectivity check catches that.
    """
    from proteus.config._escape import Zephyrus

    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Zephyrus(Pxuv=-1.0)
    # Adjacent-valid round-trip; well below the documented default 5e-5.
    z = Zephyrus(Pxuv=1e-10)
    assert z.Pxuv == pytest.approx(1e-10, rel=1e-12)


def test_zephyrus_efficiency_closed_interval_endpoints_round_trip_under_atmodeller():
    """``Zephyrus.efficiency`` is constrained to ``[0, 1]``
    inclusive. Both endpoints round-trip under a real Config
    build; the documented default (0.1) round-trips.

    The 0.0 endpoint uses ``pytest.approx(..., abs=)`` because the
    relative form is undefined at zero; the 1.0 endpoint uses the
    relative form so a copy-paste to a smaller value later keeps
    the tolerance scaling correctly.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    cfg_zero = Config(
        escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=0.0)),
        **_base_config_kwargs(),
    )
    assert cfg_zero.escape.zephyrus.efficiency == pytest.approx(0.0, abs=1e-12)
    cfg_one = Config(
        escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=1.0)),
        **_base_config_kwargs(),
    )
    assert cfg_one.escape.zephyrus.efficiency == pytest.approx(1.0, rel=1e-12)
    default_cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert default_cfg.escape.zephyrus.efficiency == pytest.approx(0.1, rel=1e-12)


def test_zephyrus_efficiency_above_unit_rejected_field_layer_with_selectivity():
    """The field-level ``le(1)`` and ``ge(0)`` validators at
    ``_escape.py:37`` reject ``efficiency > 1`` and
    ``efficiency < 0`` at Zephyrus construction time.

    Edge: pin both ends of the closed interval at the field layer.

    - ``efficiency = 1.0001`` rejects (``le(1)``). An adjacent-
      valid ``efficiency = 0.999`` round-trips: a regression that
      tightened ``le(1)`` to ``le(<below_unit>)`` would silently
      reject the documented endpoint 1.0; the selectivity check
      catches that.
    - ``efficiency = 0.0`` round-trips (``ge(0)``): a regression
      that tightened ``ge(0)`` to ``gt(0)`` would silently reject
      the documented endpoint 0.0; pinning the round-trip at the
      field layer catches that independently of the cross-
      validator.
    """
    from proteus.config._escape import Zephyrus

    # Upper-bound rejection and adjacent-valid selectivity.
    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Zephyrus(efficiency=1.0001)
    z_below = Zephyrus(efficiency=0.999)
    assert z_below.efficiency == pytest.approx(0.999, rel=1e-12)
    # Lower-bound selectivity: ge(0) must accept 0.0 (catches a
    # gt(0) regression that would silently reject the endpoint).
    z_zero = Zephyrus(efficiency=0.0)
    assert z_zero.efficiency == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Wrapper-merge contract: Path-C atmodeller + per-gas + zephyrus escape columns.
# ---------------------------------------------------------------------------


def test_atmodeller_zephyrus_helpfile_keys_register_path_c_pressures_and_escape():
    """The wrapper merge propagates per-iteration columns from both
    sides into ``hf_row``. The schema MUST register:

    - Path C atmodeller columns: ``fO2_shift_IW_derived``, ``O_res``.
    - Per-gas pressures: ``H2O_bar``, ``CO2_bar``, ``H2_bar``,
      ``CO_bar``, ``N2_bar``.
    - Zephyrus escape columns: ``esc_rate_total``,
      ``esc_kg_cumulative``, ``M_vol_initial``.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as a float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    atmodeller_path_c_keys = ('fO2_shift_IW_derived', 'O_res')
    pressure_keys = ('H2O_bar', 'CO2_bar', 'H2_bar', 'CO_bar', 'N2_bar')
    escape_keys = ('esc_rate_total', 'esc_kg_cumulative', 'M_vol_initial')
    for key in atmodeller_path_c_keys + pressure_keys + escape_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in atmodeller_path_c_keys + pressure_keys + escape_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); the relative
        # form of pytest.approx is undefined at zero, so use abs.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
