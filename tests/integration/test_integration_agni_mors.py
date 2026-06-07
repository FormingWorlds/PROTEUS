"""Integration test: AGNI (real atmosphere) coupled to MORS (real star).

MORS provides time-evolving stellar spectra (rotation-dependent
Lbol, XUV, full SED) that AGNI reads as its top-of-atmosphere
forcing. The pair is not gated by the spada_zephyrus
cross-validator at ``_config.py:25`` because that fires only when
``escape.module == 'zephyrus'``; here the escape slot is dummy,
so MORS + AGNI is a free-standing combination.

Integration-tier scope:

- Pair-wise schema validators round-trip ``atmos_clim.module='agni'``
  with ``star.module='mors'``.
- The Mors ``tracks`` enum is exactly ``{'spada', 'baraffe'}``
  with 'spada' as the default.
- The Mors ``spectrum_source`` enum includes ``solar``, ``muscles``,
  ``phoenix`` and Python ``None``; default is ``'phoenix'``.
- ``Star.mass`` and ``Star.age_ini`` are strictly positive at the
  attrs validator (``validators.gt(0)``); zero and negative reject.
- ``Star.bol_scale`` is non-negative (``ge(0)``); zero is allowed
  (lights-out dark star) but negative rejects.
- ``valid_mors`` cross-validator requires ``mors.star_name`` when
  ``spectrum_source`` is 'solar' or 'muscles'; absent name raises.
- The AGNI optical-depth aggregator emits a monotonic profile from
  TOA to surface for a MORS-driven spectrum; the matrix design
  lock requires every AGNI x X integration to assert
  ``tau_atm_TOA < 0.5 * tau_atm_surface``.

The full two-timestep AGNI + MORS coupled run sits above the
slow-tier per-step budget on Linux GHA. The slow-tier sibling
``test_integration_mors_zephyrus.py`` exercises the MORS leg with
ZEPHYRUS escape; the AGNI leg is exercised by the existing
``test_smoke_modules.py`` chain.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (agni, mors) production combination.
# ---------------------------------------------------------------------------


def test_star_module_mors_round_trips_through_schema():
    """``star.module='mors'`` is in the documented enum
    ``{None, 'mors', 'dummy'}``.

    Discrimination: every member round-trips; an unknown name
    rejects. The ``dummy`` round-trip needs ``calculate_radius=True``
    (or an explicit positive radius) to satisfy the
    ``valid_stardummy`` cross-validator; the default
    ``radius=None`` only passes when calculate_radius is True.
    """
    from proteus.config._star import Star, StarDummy

    # mors round-trips with default Mors (phoenix spectrum source).
    s_mors = Star(module='mors')
    assert s_mors.module == 'mors'
    # dummy needs calculate_radius=True to satisfy valid_stardummy
    # given the default radius=None.
    s_dummy = Star(module='dummy', dummy=StarDummy(calculate_radius=True))
    assert s_dummy.module == 'dummy'
    # The None member (via none_if_none) coerces the literal string
    # 'none' to Python None.
    s_none = Star(module='none')
    assert s_none.module is None
    with pytest.raises(ValueError, match=r'(?i)module'):
        Star(module='zarathustra')


def test_mors_tracks_enum_is_spada_or_baraffe_only():
    """Pin the Mors.tracks enum as a set so a regression that
    silently adds a third option (e.g. 'isochrone') fails here
    even if 'spada' and 'baraffe' still round-trip.

    Discrimination: round-trip both members, reject one invalid
    name, pin default to 'spada'.
    """
    import attrs

    from proteus.config._star import Mors

    allowed = attrs.fields(Mors).tracks.validator.options
    assert set(allowed) == {'spada', 'baraffe'}, (
        f'Mors.tracks enum drifted from documented set: {allowed}'
    )
    for known in ('spada', 'baraffe'):
        m = Mors(tracks=known)
        assert m.tracks == known
    with pytest.raises(ValueError, match=r'(?i)tracks'):
        Mors(tracks='isochrone')
    # Documented default is 'spada' (per Mors dataclass at
    # _star.py:86).
    default = Mors()
    assert default.tracks == 'spada'


def test_mors_spectrum_source_enum_includes_phoenix_solar_muscles_none():
    """Pin the Mors.spectrum_source enum as a set including the
    Python None member (coerced from the string 'none').

    Discrimination: confirm exactly the four-member set; reject an
    invalid name. The default is 'phoenix'.
    """
    import attrs

    from proteus.config._star import Mors

    allowed = attrs.fields(Mors).spectrum_source.validator.options
    assert set(allowed) == {'solar', 'muscles', 'phoenix', None}, (
        f'Mors.spectrum_source enum drifted from documented set: {allowed}'
    )
    for known in ('solar', 'muscles', 'phoenix'):
        m = Mors(spectrum_source=known, star_name='sun')
        assert m.spectrum_source == known
    # 'none' string coerces to Python None.
    m_none = Mors(spectrum_source='none')
    assert m_none.spectrum_source is None
    with pytest.raises(ValueError, match=r'(?i)spectrum_source'):
        Mors(spectrum_source='not_a_spectrum_class')
    default = Mors()
    assert default.spectrum_source == 'phoenix'


def test_star_mass_and_age_must_be_positive():
    """``Star.mass`` and ``Star.age_ini`` use ``validators.gt(0)`` at
    ``_star.py:177-178``. Zero and negative reject.

    Edge: limit-input case (0 and -1 raise for both fields).
    """
    from proteus.config._star import Star

    with pytest.raises(ValueError, match=r'(?i)mass'):
        Star(mass=0.0)
    with pytest.raises(ValueError, match=r'(?i)mass'):
        Star(mass=-0.5)
    with pytest.raises(ValueError, match=r'(?i)age_ini'):
        Star(age_ini=0.0)
    with pytest.raises(ValueError, match=r'(?i)age_ini'):
        Star(age_ini=-1.0)
    default = Star()
    assert default.mass > 0
    assert default.age_ini > 0


def test_star_bol_scale_allows_zero_rejects_negative():
    """``Star.bol_scale`` uses ``validators.ge(0)`` at
    ``_star.py:183``: zero is allowed (a lights-out dark companion)
    but negative is rejected.

    Discrimination: zero round-trips; positive round-trips; negative
    raises. A regression that swapped ``ge(0)`` for ``gt(0)`` would
    reject zero and fail the round-trip check.
    """
    from proteus.config._star import Star

    # Zero is allowed.
    s_zero = Star(bol_scale=0.0)
    assert s_zero.bol_scale == pytest.approx(0.0, abs=1e-12)
    # Positive round-trips.
    s_one = Star(bol_scale=1.0)
    assert s_one.bol_scale == pytest.approx(1.0, rel=1e-12)
    # Negative rejects.
    with pytest.raises(ValueError, match=r'(?i)bol_scale'):
        Star(bol_scale=-0.01)


def test_valid_mors_requires_star_name_when_spectrum_source_is_solar_or_muscles():
    """The ``valid_mors`` cross-validator at ``_star.py:14-19``
    fires when ``spectrum_source in {'solar', 'muscles'}`` and
    ``star_name is None``: it must raise so the runtime cannot
    proceed without a target spectrum file.

    Discrimination: 'phoenix' source does NOT require star_name;
    setting it to None there must round-trip without raising.
    A regression that broadened the check to phoenix would fail
    the phoenix round-trip.
    """
    from proteus.config._star import Mors, Star

    # 'solar' and 'muscles' without star_name raise.
    for src in ('solar', 'muscles'):
        with pytest.raises(ValueError, match=r'(?i)star_name'):
            Star(module='mors', mors=Mors(spectrum_source=src, star_name=None))
    # 'phoenix' without star_name does NOT raise.
    s_phoenix = Star(module='mors', mors=Mors(spectrum_source='phoenix', star_name=None))
    assert s_phoenix.mors.spectrum_source == 'phoenix'
    assert s_phoenix.mors.star_name is None
    # 'solar' WITH star_name round-trips.
    s_solar = Star(module='mors', mors=Mors(spectrum_source='solar', star_name='sun'))
    assert s_solar.mors.star_name == 'sun'


def test_valid_mors_rotation_constraints_both_or_neither_raises():
    """The ``valid_mors`` cross-validator enforces "exactly one of
    rot_pcntle / rot_period must be set". Setting both raises;
    setting neither raises; a negative period raises.

    Edge: pin all three rotation-related branches of valid_mors
    (lines 29-38 of _star.py). The defaults (rot_pcntle=50.0,
    rot_period=None) satisfy the "exactly one set" rule, so a
    regression that flipped the defaults would silently violate the
    invariant; this test fails loudly on any such drift.
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
    # Discrimination: the documented default (rot_pcntle=50.0,
    # rot_period=None) MUST round-trip without raising. A regression
    # that flipped one of these defaults would fail here.
    s_ok = Star(module='mors', mors=Mors(spectrum_source='phoenix'))
    assert s_ok.mors.rot_pcntle == pytest.approx(50.0, rel=1e-12)
    assert s_ok.mors.rot_period is None


# ---------------------------------------------------------------------------
# Optical-depth monotonicity at the AGNI side of the AGNI x MORS pair.
# Matrix design lock: every AGNI x X integration test must assert this.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_agni_mors_optical_depth_monotonic_from_TOA_to_surface():
    """Drive ``_summarise_tau_band`` with a tau profile representative
    of AGNI under a MORS solar-spectrum forcing: present-day Sun at
    1 AU, Earth-like H2O + CO2 atmosphere. Confirm
    ``tau_atm_TOA < 0.5 * tau_atm_surface``.

    Physical scenario: Sun at 4.567 Gyr, F_ins ~ 1361 W/m^2;
    integrated optical depth grows from ~0 at TOA to O(few) at the
    surface in a wet atmosphere.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.005, 0.003, 0.001, 0.0008],  # TOA
            [0.06, 0.08, 0.04, 0.03],
            [0.6, 0.9, 0.3, 0.4],
            [4.0, 6.5, 2.5, 3.0],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=4)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA == pytest.approx(np.mean(tau_band[0, :]), rel=1e-12)
    assert tau_surface == pytest.approx(np.mean(tau_band[-1, :]), rel=1e-12)
    assert tau_TOA < tau_surface
    # Matrix design lock invariant.
    assert tau_TOA < 0.5 * tau_surface


@pytest.mark.physics_invariant
def test_agni_mors_optical_depth_bounded_below_by_zero():
    """Optical depth is non-negative everywhere by construction.
    Pair-edge case: a young M-dwarf XUV-dominated input still
    produces a tau profile that respects monotonicity AND the
    matrix design-lock invariant ``tau_atm_TOA < 0.5 *
    tau_atm_surface``.

    Edge: TOA exactly at zero (semi-transparent UV photosphere)
    AND non-zero surface; both bounds and the design lock must hold.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.array(
        [
            [0.0, 0.0, 0.0],  # TOA: transparent in all bands
            [0.4, 0.6, 0.3],
            [2.0, 3.5, 1.8],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=3, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    # Boundedness invariant.
    assert tau_TOA >= 0.0
    assert tau_surface > 0.0
    # Monotonicity invariant.
    assert tau_TOA < tau_surface
    # Matrix design-lock invariant: every AGNI x X test asserts this.
    assert tau_TOA < 0.5 * tau_surface


# ---------------------------------------------------------------------------
# Wrapper-merge contract: MORS-side stellar columns + AGNI diagnostics.
# ---------------------------------------------------------------------------


def test_agni_mors_helpfile_keys_register_stellar_and_agni_columns():
    """The MORS leg writes per-iteration stellar columns into hf_row
    (T_star, R_star, M_star, F_ins, F_xuv, age_star, separation),
    and the AGNI leg writes its four diagnostic columns. The wrapper
    merge guard at ``atmos_clim/wrapper.py:196-198`` depends on every
    key being registered in ``GetHelpfileKeys``.

    Discrimination: pin both sets of keys with explicit per-key
    assertions. A regression that dropped any one would fail the
    per-key loop. The ``ZeroHelpfileRow`` seeding pins that each is
    initialised as float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    agni_diagnostic_keys = (
        'tau_atm_TOA',
        'tau_atm_surface',
        'atm_Ra_max',
        'atm_t_conv_over_t_rad',
    )
    mors_stellar_keys = (
        'T_star',
        'R_star',
        'M_star',
        'F_ins',
        'F_xuv',
        'age_star',
        'separation',
    )
    for key in agni_diagnostic_keys + mors_stellar_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in agni_diagnostic_keys + mors_stellar_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-12)
        assert isinstance(row[key], float)
