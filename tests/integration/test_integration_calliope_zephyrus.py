"""Integration test: CALLIOPE (real outgas) coupled to ZEPHYRUS (real escape).

ZEPHYRUS is hard-coupled to MORS via the ``spada_zephyrus``
cross-validator at ``src/proteus/config/_config.py:25-31``: a
Config with ``escape.module='zephyrus'`` MUST also have
``star.module='mors'`` AND ``star.mors.tracks='spada'``. The
CALLIOPE x ZEPHYRUS pair therefore requires star=MORS+spada to
round-trip; CALLIOPE provides the per-iteration surface partial
pressures that determine the volatile reservoir from which
ZEPHYRUS draws.

Integration-tier scope:

- Schema validators round-trip ``outgas.module='calliope'`` with
  ``escape.module='zephyrus'`` when paired with
  ``star.module='mors'`` and ``mors.tracks='spada'``.
- ``spada_zephyrus`` rejects the same Config with
  ``star.module='dummy'`` or ``mors.tracks='baraffe'``; both
  pin the gate for the calliope leg.
- Escape module enum is pinned as ``{None, 'dummy', 'zephyrus',
  'boreas'}`` so a regression that silently added a fifth backend
  surfaces here.
- Calliope's ten-species ``include_*`` field set is pinned by
  count via ``attrs.fields(Calliope)``; the documented species
  list and the ``is_included`` helper are pinned with the
  ``AttributeError``-on-unknown-name discrimination guard.
- ``Zephyrus.Pxuv`` validator-layer split: the field-level
  ``ge(0)`` validator at ``_escape.py:36`` and the cross-validator
  ``valid_zephyrus`` at ``_escape.py:13-15`` are tested
  separately. The cross-validator rejects ``Pxuv = 0`` (open
  lower) and ``Pxuv > 10`` (closed upper) that the field
  validator does not cover.
- ``Zephyrus.efficiency`` closed-interval endpoints ``0.0`` and
  ``1.0`` round-trip; ``> 1`` rejects.
- The wrapper merge guard pins calliope per-gas pressure columns
  (``H2O_bar``, ``CO2_bar``, ``N2_bar``, ``H2_bar``, ``CO_bar``)
  AND zephyrus escape columns (``esc_rate_total``,
  ``esc_kg_cumulative``, ``M_vol_initial``) in
  ``GetHelpfileKeys`` so per-iteration values flow into the
  helpfile.

The full two-timestep coupled run with both real solvers and a
real atmosphere is exercised by the slow-tier
``test_integration_mors_zephyrus.py`` (dummy interior + mors +
zephyrus) and the AGNI x CALLIOPE leg covers the calliope
atmosphere boundary.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs():
    """Build base Config kwargs for the (calliope, zephyrus,
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
        outgas=Outgas(module='calliope'),
        star=Star(module='mors'),  # default Mors uses spada + phoenix
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (calliope, zephyrus, mors+spada) combo.
# ---------------------------------------------------------------------------


def test_calliope_zephyrus_mors_spada_round_trips_through_config():
    """The full hard-coupled triple (outgas=calliope, escape=zephyrus,
    star=mors+spada) round-trips through Config without raising.

    Discrimination: a regression in either side that broke schema
    construction would surface here. The asserts confirm each
    module landed where expected.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape

    cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert cfg.outgas.module == 'calliope'
    assert cfg.escape.module == 'zephyrus'
    assert cfg.star.module == 'mors'
    assert cfg.star.mors.tracks == 'spada'


def test_calliope_zephyrus_rejects_dummy_star_at_spada_zephyrus_layer():
    """``spada_zephyrus`` fires when escape=zephyrus is paired with
    a non-MORS star. The calliope x zephyrus combination must
    reject the dummy-star configuration even though the calliope
    outgas side itself is valid for any star.

    The kwargs use ``atmos_clim='dummy'`` to avoid a latent
    dependency on the ``params.stop.escape.enabled`` default: with
    ``atmos_clim='janus'`` the ``janus_escape_atmosphere``
    cross-validator would fire when ``stop.escape.enabled`` is
    False, masking the ``spada_zephyrus`` rejection we want to
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
        outgas=Outgas(module='calliope'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)
    # Selectivity: escape=dummy with same star must construct cleanly.
    cfg_ok = Config(escape=Escape(module='dummy'), **kwargs)
    assert cfg_ok.escape.module == 'dummy'


def test_calliope_zephyrus_rejects_baraffe_tracks_at_spada_zephyrus_layer():
    """``spada_zephyrus`` also rejects mors+baraffe even when the
    outgas side is calliope. Pinning the baraffe-rejection case
    here so a regression that relaxed the spada-only gate (e.g.
    extended to baraffe as well) surfaces in the calliope leg.

    Same ``atmos_clim='dummy'`` choice as the dummy-star rejection
    above: avoids any latent dependency on
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
        outgas=Outgas(module='calliope'),
        star=Star(module='mors', mors=Mors(tracks='baraffe')),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **kwargs)
    # Adjacent-valid: mors+spada with zephyrus must construct cleanly.
    kwargs_spada = {**kwargs, 'star': Star(module='mors', mors=Mors(tracks='spada'))}
    cfg_ok = Config(escape=Escape(module='zephyrus'), **kwargs_spada)
    assert cfg_ok.star.mors.tracks == 'spada'


# ---------------------------------------------------------------------------
# Escape module enum pinned as set.
# ---------------------------------------------------------------------------


def test_escape_module_enum_pinned_as_set():
    """Pin the Escape.module enum as ``{None, 'dummy', 'zephyrus',
    'boreas'}``. A regression that silently added a fifth escape
    backend (e.g. a new HD-escape implementation) would still let
    the four documented values round-trip and would still reject an
    obvious typo, so the set check is the only way to catch enum
    drift.
    """
    import attrs

    from proteus.config._escape import Escape

    allowed = attrs.fields(Escape).module.validator.options
    assert set(allowed) == {None, 'dummy', 'zephyrus', 'boreas'}, (
        f'Escape.module enum drifted from documented set: {allowed}'
    )
    # All four documented values round-trip when paired with valid
    # sub-module defaults (no cross-validator interference because
    # spada_zephyrus only fires inside a Config build).
    for known in ('dummy', 'zephyrus', 'boreas'):
        e = Escape(module=known)
        assert e.module == known
    with pytest.raises(ValueError, match=r'(?i)module'):
        Escape(module='not_a_real_escape_backend')


# ---------------------------------------------------------------------------
# Calliope-side schema (re-pinned for this pair).
# ---------------------------------------------------------------------------


def test_calliope_is_included_preserves_documented_ten_gas_set_under_zephyrus_pair():
    """``Calliope.is_included`` must return True for every gas in
    the documented ten-species set when defaults apply. The
    surface-pressure feedback into ZEPHYRUS's escape-reservoir
    composition reflects on this set; a silently-dropped species
    would push the next escape iteration onto a wrong reservoir
    composition.

    Field-count guard: the number of ``include_*`` fields is pinned
    via ``attrs.fields(Calliope)`` so a regression that silently
    adds or removes a reactive species fails the count check even when the
    documented ten still appear.
    """
    import attrs

    from proteus.config._outgas import Calliope

    c = Calliope()
    documented_species = (
        'H2O',
        'CO2',
        'N2',
        'S2',
        'SO2',
        'H2S',
        'NH3',
        'H2',
        'CH4',
        'CO',
    )
    for gas in documented_species:
        assert c.is_included(gas) is True, f'{gas} missing from Calliope defaults'

    from proteus.utils.constants import noble_gases

    include_fields = [f for f in attrs.fields(Calliope) if f.name.startswith('include_')]
    reactive_fields = [
        f for f in include_fields if f.name.removeprefix('include_') not in noble_gases
    ]
    assert len(reactive_fields) == len(documented_species), (
        f'Expected {len(documented_species)} reactive include_* fields on Calliope, '
        f'got {len(reactive_fields)}: {[f.name for f in reactive_fields]}'
    )
    # The noble gases are opt-in, so each has an include_* flag defaulting off.
    for _noble in noble_gases:
        assert c.is_included(_noble) is False, f'{_noble} should default to off'

    # Discrimination: helper must raise on an undocumented attribute
    # rather than silently return False.
    with pytest.raises(AttributeError):
        c.is_included('Rn')


def test_calliope_solver_parameters_remain_positive_under_zephyrus_pair():
    """Re-pin Calliope's ``nguess > 0`` and ``nsolve > 0`` so a
    regression in the outgas side that the AGNI x CALLIOPE file
    would catch also surfaces here when escape is enabled.
    """
    from proteus.config._outgas import Calliope

    with pytest.raises(ValueError, match=r'(?i)nguess'):
        Calliope(nguess=0)
    with pytest.raises(ValueError, match=r'(?i)nsolve'):
        Calliope(nsolve=0)
    default = Calliope()
    assert default.nguess == 1000
    assert default.nsolve == 3000


# ---------------------------------------------------------------------------
# Zephyrus-side validator-layer split (Pxuv and efficiency).
# ---------------------------------------------------------------------------


def test_zephyrus_pxuv_cross_validator_rejects_open_lower_and_above_upper():
    """``valid_zephyrus`` at ``_escape.py:13-15`` rejects
    ``Pxuv <= 0`` (open lower) and ``Pxuv > 10`` (closed upper).
    The field-level ``ge(0)`` validator at ``_escape.py:36`` does
    NOT catch ``Pxuv = 0`` (it accepts equality), so the
    cross-validator owns that case.

    Edge: ``Pxuv = 0.0`` raises at the cross-validator layer,
    ``Pxuv = 10.0`` round-trips (upper-closed), ``Pxuv = 10.0001``
    raises (just above upper). The default (5e-5) round-trips
    under the full Config build.

    Discrimination: ``Pxuv = 0.0`` discriminates the cross-validator
    from the field validator (the field accepts it, the cross
    rejects). ``Pxuv = 10.0001`` discriminates the upper-bound
    branch that ``ge(0)`` does not gate.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    # Open lower bound: 0 passes ge(0) but valid_zephyrus rejects.
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
    # 10.0001 rejects (just above closed upper).
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0001)),
            **_base_config_kwargs(),
        )
    # Documented default 5e-5 round-trips under the full Config.
    default_cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert default_cfg.escape.zephyrus.Pxuv == pytest.approx(5e-5, rel=1e-12)


def test_zephyrus_pxuv_field_level_validator_rejects_strictly_negative():
    """The field-level ``ge(0)`` validator at ``_escape.py:36``
    rejects strictly-negative Pxuv at Zephyrus construction time,
    BEFORE valid_zephyrus runs at the enclosing Escape construction.

    Discrimination: separating this case from the cross-validator
    test above ensures a regression that removed ``ge(0)`` in
    favour of relying on valid_zephyrus alone would surface here
    (``Zephyrus(Pxuv=-1.0)`` would no longer raise at the field
    layer; it would only raise once wrapped in an Escape).

    Edge: a strictly-positive Pxuv adjacent to the boundary
    (``1e-10`` bar, well below the documented default of 5e-5)
    round-trips through Zephyrus construction. Pinning this
    adjacent-valid case rejects a regression that tightened the
    field validator to ``gt(<positive_floor>)`` and would silently
    reject the documented default.
    """
    from proteus.config._escape import Zephyrus

    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Zephyrus(Pxuv=-1.0)
    # Adjacent-valid round-trip: strictly positive, well below the
    # 5e-5 default; selectivity check on the field validator.
    z = Zephyrus(Pxuv=1e-10)
    assert z.Pxuv == pytest.approx(1e-10, rel=1e-12)


def test_zephyrus_efficiency_closed_interval_endpoints_round_trip():
    """``Zephyrus.efficiency`` is constrained to ``[0, 1]``
    inclusive per the field-level ``(ge(0), le(1))`` at
    ``_escape.py:37`` and the structurally-redundant
    ``valid_zephyrus`` cross-check at ``_escape.py:17-19``.

    Edge: both 0.0 and 1.0 round-trip (closed-closed); the
    documented default (0.1) round-trips. The out-of-range
    rejections are owned by the field validators and tested in
    ``test_integration_agni_zephyrus.py``; this file pins the
    inclusive endpoints and the default under a real Config build
    so the calliope leg has its own coverage.
    """
    from proteus.config import Config
    from proteus.config._escape import Escape, Zephyrus

    for boundary in (0.0, 1.0):
        cfg = Config(
            escape=Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=boundary)),
            **_base_config_kwargs(),
        )
        assert cfg.escape.zephyrus.efficiency == pytest.approx(boundary, abs=1e-12)
    # Default 0.1 round-trips.
    default_cfg = Config(escape=Escape(module='zephyrus'), **_base_config_kwargs())
    assert default_cfg.escape.zephyrus.efficiency == pytest.approx(0.1, rel=1e-12)


def test_zephyrus_efficiency_above_unit_rejected_at_field_layer():
    """The field-level ``le(1)`` validator at ``_escape.py:37``
    rejects ``efficiency > 1`` at Zephyrus construction time,
    BEFORE valid_zephyrus runs. Pin the just-above-unit case to
    discriminate the field validator from the cross-validator
    (both would catch this, but the field validator fires first
    and a regression that removed the field layer would still
    leave the cross-validator catching it inside Escape).

    The strictly-negative efficiency case is owned by the AGNI x
    ZEPHYRUS file; this pair pins the upper-bound edge only to
    avoid duplicating coverage.

    Edge: an efficiency adjacent to the upper bound (``0.999``)
    round-trips through Zephyrus construction. Pinning this
    adjacent-valid case rejects a regression that tightened the
    field validator to ``le(<below_unit>)`` and would silently
    reject the documented endpoint 1.0.
    """
    from proteus.config._escape import Zephyrus

    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Zephyrus(efficiency=1.0001)
    # Adjacent-valid round-trip: just below unit; selectivity check
    # on the field validator.
    z = Zephyrus(efficiency=0.999)
    assert z.efficiency == pytest.approx(0.999, rel=1e-12)


# ---------------------------------------------------------------------------
# Wrapper-merge contract: calliope per-gas pressures + zephyrus escape columns.
# ---------------------------------------------------------------------------


def test_calliope_zephyrus_helpfile_keys_register_outgas_and_escape_columns():
    """The wrapper merge guard at ``atmos_clim/wrapper.py:196-198``
    propagates per-iteration columns from both sides into hf_row.
    The schema MUST register:

    - Calliope per-gas pressures: ``H2O_bar``, ``CO2_bar``,
      ``N2_bar``, ``H2_bar``, ``CO_bar``.
    - Zephyrus escape columns: ``esc_rate_total``,
      ``esc_kg_cumulative``, ``M_vol_initial``.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    calliope_per_gas_keys = (
        'H2O_bar',
        'CO2_bar',
        'N2_bar',
        'H2_bar',
        'CO_bar',
    )
    escape_keys = ('esc_rate_total', 'esc_kg_cumulative', 'M_vol_initial')
    for key in calliope_per_gas_keys + escape_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in calliope_per_gas_keys + escape_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); the relative
        # form of pytest.approx is undefined at zero, so use abs.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
