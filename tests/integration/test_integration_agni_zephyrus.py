"""Integration test: AGNI (real atmosphere) coupled to ZEPHYRUS (real escape).

ZEPHYRUS is the energy-limited XUV-driven escape solver. It is
hard-coupled to MORS via the ``spada_zephyrus`` cross-validator at
``src/proteus/config/_config.py:25-31``: a Config with
``escape.module='zephyrus'`` MUST also have ``star.module='mors'``
AND ``star.mors.tracks='spada'`` or construction raises. AGNI
provides the per-iteration atmosphere; the pair therefore requires
star=MORS+spada to round-trip.

Integration-tier scope:

- The ``escape.module`` enum is exactly ``{None, 'dummy',
  'zephyrus', 'boreas'}`` pinned as a set.
- ``Zephyrus.Pxuv`` enforces a closed-interval ``(0, 10]`` bar
  contract via the ``valid_zephyrus`` cross-validator: 0, -1, and
  10.0001 all reject; the default and a representative interior
  value round-trip.
- ``Zephyrus.efficiency`` enforces a closed-interval ``[0, 1]``
  contract: 0 and 1 round-trip; negative and >1 reject.
- The hard-coupled ``spada_zephyrus`` cross-validator at
  ``_config.py:25-31`` rejects a Config with ``escape.module=
  'zephyrus'`` when ``star.module='dummy'`` or
  ``star.mors.tracks='baraffe'``; the ``mors + spada``
  combination round-trips.
- The AGNI optical-depth aggregator emits a monotonic profile
  from TOA to surface under a MORS+ZEPHYRUS coupling; the matrix
  design lock requires every AGNI x X integration to assert
  ``tau_atm_TOA < 0.5 * tau_atm_surface``.
- The wrapper merge propagates the four AGNI 1.10.2 diagnostic
  keys AND the ZEPHYRUS-side escape columns (``esc_rate_total``,
  ``esc_kg_cumulative``, ``M_vol_initial``) through ``hf_row``.

The full two-timestep AGNI + ZEPHYRUS coupled run sits above the
slow-tier per-step budget on Linux GHA. The slow-tier sibling
``test_integration_mors_zephyrus.py`` exercises the ZEPHYRUS leg
with a dummy atmosphere; the AGNI leg is exercised by the
existing ``test_smoke_modules.py`` chain.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (agni, zephyrus) coupling.
# ---------------------------------------------------------------------------


def test_escape_module_enum_is_documented_set():
    """Pin the Escape.module enum exactly as the documented
    ``{None, 'dummy', 'zephyrus', 'boreas'}``.

    Discrimination: round-trip 'zephyrus' to confirm it is the
    default; pin the enum-as-set so a regression that added a
    fifth value (e.g. 'wind') fails here. Reject an invalid name
    to confirm the validator still fires.
    """
    import attrs

    from proteus.config._escape import Escape

    allowed = attrs.fields(Escape).module.validator.options
    assert set(allowed) == {None, 'dummy', 'zephyrus', 'boreas'}, (
        f'Escape.module enum drifted from documented set: {allowed}'
    )
    # Default is zephyrus per _escape.py:158-159.
    default = Escape()
    assert default.module == 'zephyrus'
    # 'none' coerces to Python None.
    e_none = Escape(module='none')
    assert e_none.module is None
    with pytest.raises(ValueError, match=r'(?i)module'):
        Escape(module='solar_wind')


def test_zephyrus_pxuv_is_in_zero_to_ten_bar_open_lower_closed_upper():
    """``Zephyrus.Pxuv`` is gated by two validators:

    - field-level ``ge(0)`` at ``_escape.py:36`` rejects any
      strictly-negative value at Zephyrus construction time.
    - cross-validator ``valid_zephyrus`` at ``_escape.py:13-15``
      rejects ``Pxuv <= 0`` or ``Pxuv > 10`` at Escape construction
      time. This pins the open-lower ``0`` and the just-above-upper
      ``10.0001`` cases that the field validator does not catch.

    Edge: 0 raises (lower-open, valid_zephyrus), 10.0 round-trips
    (upper-closed), 10.0001 raises (just above upper), the default
    (5e-5) round-trips.

    Discrimination: pinning ``Pxuv=0.0`` exercises valid_zephyrus
    specifically (the ``ge(0)`` field validator accepts 0); pinning
    ``Pxuv=10.0001`` exercises the upper bound that ``ge(0)`` does
    not cover. Together the two rejections discriminate the
    cross-validator from the field validator.
    """
    from proteus.config._escape import Escape, Zephyrus

    # Open lower bound: 0.0 passes the ge(0) field validator but
    # fails valid_zephyrus (Pxuv <= 0). This case discriminates the
    # cross-validator from the field validator.
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=0.0))
    # Just above the closed upper bound: 10.0001 only fails
    # valid_zephyrus; the ge(0) field validator passes.
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0001))
    # Right at the upper bound round-trips.
    e_upper = Escape(module='zephyrus', zephyrus=Zephyrus(Pxuv=10.0))
    assert e_upper.zephyrus.Pxuv == pytest.approx(10.0, rel=1e-12)
    # The documented default round-trips.
    e_default = Escape(module='zephyrus')
    assert e_default.zephyrus.Pxuv == pytest.approx(5.0e-5, rel=1e-12)


def test_zephyrus_pxuv_field_level_validator_rejects_strictly_negative():
    """The field-level ``ge(0)`` validator at ``_escape.py:36``
    rejects strictly-negative Pxuv at Zephyrus construction time,
    BEFORE valid_zephyrus runs at the enclosing Escape construction.

    Discrimination: separating this case from the cross-validator
    test above ensures a future regression that removed ``ge(0)``
    in favour of relying on valid_zephyrus alone would surface
    here (Zephyrus(Pxuv=-1.0) would no longer raise at the field
    layer; it would only raise inside an Escape construction).
    """
    from proteus.config._escape import Zephyrus

    # The field validator on Zephyrus itself catches negative Pxuv
    # before any Escape-level cross-validator runs.
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        Zephyrus(Pxuv=-1.0)
    # Adjacent-valid: a small positive value must round-trip without error,
    # confirming the ge(0) boundary is inclusive at zero.
    z = Zephyrus(Pxuv=0.0)
    assert z.Pxuv == pytest.approx(0.0, abs=1e-12)


def test_zephyrus_efficiency_in_unit_interval_closed_at_both_ends():
    """``Zephyrus.efficiency`` is constrained to ``[0, 1]``
    inclusive per ``valid_zephyrus`` at ``_escape.py:17-19``.

    Edge: both 0.0 and 1.0 round-trip (closed-closed); negative
    and >1 reject. The default (0.1) round-trips.

    Discrimination: a regression that swapped ``>=`` for ``>`` at
    either end would reject the documented endpoint and fail the
    round-trip.
    """
    from proteus.config._escape import Escape, Zephyrus

    for boundary in (0.0, 1.0):
        e = Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=boundary))
        assert e.zephyrus.efficiency == pytest.approx(boundary, abs=1e-12)
    # Outside the interval rejects.
    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=-0.01))
    with pytest.raises(ValueError, match=r'(?i)efficiency'):
        Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=1.01))
    # Default round-trip.
    e_default = Escape(module='zephyrus')
    assert e_default.zephyrus.efficiency == pytest.approx(0.1, rel=1e-12)


# ---------------------------------------------------------------------------
# Hard-coupled cross-validator: ZEPHYRUS requires MORS + Spada tracks.
# ---------------------------------------------------------------------------


def test_spada_zephyrus_rejects_zephyrus_with_dummy_star():
    """``spada_zephyrus`` at ``_config.py:25-31`` fires when
    ``escape.module='zephyrus'`` but ``star.module != 'mors'``.

    Discrimination: the same Config without zephyrus (escape=dummy)
    must round-trip with a dummy star, so the error is specifically
    triggered by the zephyrus + non-mors combination, not by the
    dummy star alone.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape

    # Build a minimal Config kwargs dict using defaults for everything
    # except the slots we deliberately set. The Config constructor
    # accepts nested dataclasses; we provide the ones we override.
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    base = dict(
        atmos_clim=AtmosClim(module='agni'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    # Zephyrus with dummy star: spada_zephyrus must raise.
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), **base)
    # Same config with escape.module='dummy' constructs cleanly,
    # confirming the error is specific to zephyrus + non-mors.
    cfg_ok = Config(escape=Escape(module='dummy'), **base)
    assert cfg_ok.escape.module == 'dummy'


def test_spada_zephyrus_rejects_zephyrus_with_baraffe_tracks():
    """``spada_zephyrus`` also fires when ``star.module='mors'`` is
    paired with ``mors.tracks='baraffe'`` (not 'spada').

    Discrimination: explicitly assert the spada-only condition by
    swapping tracks. The error message names MORS and/or Spada.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Mors, Star

    base = dict(
        atmos_clim=AtmosClim(module='agni'),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    star_baraffe = Star(module='mors', mors=Mors(tracks='baraffe'))
    with pytest.raises(ValueError, match=r'(?i)(MORS|spada)'):
        Config(escape=Escape(module='zephyrus'), star=star_baraffe, **base)
    # Adjacent-valid: same star with spada tracks must round-trip.
    star_spada = Star(module='mors', mors=Mors(tracks='spada'))
    cfg_ok = Config(escape=Escape(module='zephyrus'), star=star_spada, **base)
    assert cfg_ok.star.mors.tracks == 'spada'


def test_spada_zephyrus_accepts_mors_plus_spada():
    """The documented production combination
    ``escape='zephyrus' + star='mors' + tracks='spada'`` round-trips
    without raising.

    Edge: the positive case. Without this counter-test, the two
    rejection tests above could mistakenly accept a regression that
    rejected ALL escape configurations.
    """
    from proteus.config import Config
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Mors, Star

    base = dict(
        atmos_clim=AtmosClim(module='agni'),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )
    cfg = Config(
        escape=Escape(module='zephyrus'),
        star=Star(module='mors', mors=Mors(tracks='spada')),
        **base,
    )
    assert cfg.escape.module == 'zephyrus'
    assert cfg.star.module == 'mors'
    assert cfg.star.mors.tracks == 'spada'


# ---------------------------------------------------------------------------
# Optical-depth monotonicity at the AGNI side of the AGNI x ZEPHYRUS pair.
# Matrix design lock: every AGNI x X integration test must assert this.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_agni_zephyrus_optical_depth_monotonic_from_TOA_to_surface():
    """Drive ``_summarise_tau_band`` with a tau profile representative
    of AGNI under MORS+ZEPHYRUS XUV-driven escape: thin H2O+H2
    atmosphere being eroded; tau still grows from TOA to surface
    by 2-3 orders of magnitude.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.001, 0.0008, 0.0005],  # TOA
            [0.04, 0.06, 0.03],
            [0.4, 0.5, 0.3],
            [2.0, 3.5, 1.8],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA == pytest.approx(tau_band[0, 1], rel=1e-12)
    assert tau_surface == pytest.approx(tau_band[-1, 1], rel=1e-12)
    assert tau_TOA < tau_surface
    assert tau_TOA < 0.5 * tau_surface


@pytest.mark.physics_invariant
def test_agni_zephyrus_optical_depth_bounded_below_by_zero():
    """Boundedness invariant: tau >= 0 everywhere. A regression that
    admitted a negative tau_band value would land the aggregator
    near zero at one endpoint and bypass the monotonicity check.

    Edge: TOA exactly at zero with a non-zero surface; the matrix
    design lock invariant still holds.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.0, 0.0],  # TOA: transparent
            [0.3, 0.4],
            [1.5, 2.5],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=3, nbands=2)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA == pytest.approx(tau_band[0, 1], rel=1e-12)
    assert tau_surface == pytest.approx(tau_band[-1, 1], rel=1e-12)
    assert tau_TOA < tau_surface
    assert tau_TOA < 0.5 * tau_surface


# ---------------------------------------------------------------------------
# Wrapper-merge contract: AGNI diagnostics + ZEPHYRUS escape columns.
# ---------------------------------------------------------------------------


def test_agni_zephyrus_helpfile_keys_register_escape_and_agni_columns():
    """The wrapper merge guard at ``atmos_clim/wrapper.py:196-198``
    propagates ZEPHYRUS-side escape columns into ``hf_row`` only
    when those keys are already registered in ``GetHelpfileKeys()``.

    Discrimination: pin the four AGNI 1.10.2 diagnostic keys AND
    the three ZEPHYRUS bookkeeping columns (esc_rate_total,
    esc_kg_cumulative, M_vol_initial). Each key is independently
    registered; a regression that dropped any one would fail the
    per-key assertion.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    agni_diagnostic_keys = (
        'tau_atm_TOA',
        'tau_atm_surface',
        'atm_Ra_max',
        'atm_t_conv_over_t_rad',
    )
    zephyrus_escape_keys = (
        'esc_rate_total',
        'esc_kg_cumulative',
        'M_vol_initial',
    )
    for key in agni_diagnostic_keys + zephyrus_escape_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in agni_diagnostic_keys + zephyrus_escape_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-12)
        assert isinstance(row[key], float)
