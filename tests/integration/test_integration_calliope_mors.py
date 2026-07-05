"""Integration test: CALLIOPE (real outgas) coupled to MORS (real star).

CALLIOPE supplies the per-iteration surface partial pressures from
its fO2-buffered equilibrium solver; MORS supplies the
time-evolving stellar spectrum and bolometric flux. The pair sits
inside any PROTEUS run that drops the dummy outgas + dummy star
slots together; this file pins the schema, helper, and helpfile
contracts at the integration tier without booting Julia or
calling the real solvers.

Integration-tier scope:

- Schema validators round-trip ``outgas.module='calliope'`` with
  ``star.module='mors'``.
- Calliope's ten reactive ``include_*`` fields are pinned by count
  via ``attrs.fields(Calliope)`` so a silently added or removed
  reactive species fails the count guard even when the documented
  ten still appear; the five opt-in noble gas flags are checked
  separately as defaults-off.
- Calliope's ``is_included`` helper preserves the documented
  ten-gas set when defaults apply and raises ``AttributeError`` on
  an undocumented species.
- Calliope's ``nguess`` and ``nsolve`` solver-loop parameters
  enforce ``gt(0)`` at the attrs validator layer.
- Mors's ``tracks`` and ``spectrum_source`` enums are pinned as
  sets so a third value added without changing the round-trip
  surface is caught immediately.
- ``valid_mors`` rejects ``age_now`` <= 0 and ``age_now is None``
  when the star module is 'mors'.
- ``Mors.phoenix_radius / phoenix_log_g / phoenix_Teff`` use
  ``optional(gt(0))`` so the documented default ``None`` rounds
  trips, a positive override rounds trips, and zero or negative
  raise.
- The wrapper merge guard pins both Calliope per-gas pressure
  columns (``H2O_bar``, ``CO2_bar``, ``N2_bar``, ``H2_bar``,
  ``CO_bar``) and Mors stellar columns (``T_star``, ``R_star``,
  ``M_star``, ``F_ins``, ``F_xuv``) in ``GetHelpfileKeys`` so
  per-iteration values flow into the helpfile.

The full two-timestep coupled run with both real solvers and a
real atmosphere is exercised by the slow-tier
``test_slow_aragog_calliope.py`` (with aragog interior) and the
MORS leg is exercised by ``test_smoke_modules.py``.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (calliope, mors) production combo.
# ---------------------------------------------------------------------------


def test_outgas_calliope_and_star_mors_both_round_trip_through_schema():
    """``outgas.module='calliope'`` and ``star.module='mors'`` both
    round-trip without raising.

    Discrimination: reject an obviously-wrong outgas module name to
    confirm the validator still fires (rules out a regression that
    disabled validation entirely), and confirm the Mors default
    Star round-trips (so a regression in ``valid_mors`` that broke
    the default config would surface here).
    """
    from proteus.config._outgas import Outgas
    from proteus.config._star import Star

    o = Outgas(module='calliope')
    assert o.module == 'calliope'
    # Calliope default solver loop parameters round-trip.
    assert o.calliope.nguess > 0
    assert o.calliope.nsolve > 0
    # An invalid outgas module rejects.
    with pytest.raises(ValueError, match=r'(?i)module'):
        Outgas(module='not_a_real_outgas_backend')
    # Mors default Star round-trips with documented defaults.
    s = Star(module='mors')
    assert s.module == 'mors'
    assert s.mors.tracks == 'spada'
    assert s.mors.spectrum_source == 'phoenix'


def test_calliope_solver_parameters_must_be_positive():
    """``Calliope.nguess`` and ``Calliope.nsolve`` use ``gt(0)`` at
    ``_outgas.py:55-56``. Defaults round-trip; zero and negative
    raise. A regression that swapped ``gt(0)`` for ``ge(0)`` would
    accept zero, so the explicit zero-raise discriminates.
    """
    from proteus.config._outgas import Calliope

    with pytest.raises(ValueError, match=r'(?i)nguess'):
        Calliope(nguess=0)
    with pytest.raises(ValueError, match=r'(?i)nguess'):
        Calliope(nguess=-100)
    with pytest.raises(ValueError, match=r'(?i)nsolve'):
        Calliope(nsolve=0)
    with pytest.raises(ValueError, match=r'(?i)nsolve'):
        Calliope(nsolve=-50)

    default = Calliope()
    # Pin the documented defaults so a silent shift surfaces here.
    assert default.nguess == 1000
    assert default.nsolve == 3000


def test_calliope_is_included_preserves_documented_ten_gas_set():
    """``Calliope.is_included`` must return True for every gas in
    the documented ten-species set when defaults apply. The
    surface-pressure feedback into AGNI / MORS-radiative-equilibrium
    code paths reflects on this set; a silently-dropped species
    would push the next iteration down the wrong solubility branch.

    Discrimination: every species pinned separately; the trailing
    ``is_included('Rn')`` raise pins that the helper does not
    silently return False for an absent attribute.
    """
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

    # Field-count guard: pins the number of include_* fields so a
    # regression that silently adds or removes a reactive species fails the
    # count check even if the documented ten still appear.
    import attrs

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
    # rather than silently return False; the attrs class does not
    # carry an include_Rn field.
    with pytest.raises(AttributeError):
        c.is_included('Rn')


# ---------------------------------------------------------------------------
# MORS-side: enums pinned as sets + age_now positivity + phoenix overrides.
# ---------------------------------------------------------------------------


def test_mors_tracks_and_spectrum_source_enums_pinned_as_sets():
    """Pin ``Mors.tracks`` as ``{'spada', 'baraffe'}`` and
    ``Mors.spectrum_source`` as ``{'solar', 'muscles', 'phoenix',
    None}``. Both checks use set equality so a regression that
    silently added a third value would fail even though the
    documented values still round-trip.
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
    # Defaults match the documented production combination.
    default = Mors()
    assert default.tracks == 'spada'
    assert default.spectrum_source == 'phoenix'
    assert default.age_now == pytest.approx(4.567, rel=1e-12)


def test_mors_age_now_positivity_at_valid_mors_layer():
    """``valid_mors`` at ``_star.py:13-14`` raises when
    ``mors.age_now`` is None or <=0. The check runs at the Star
    cross-validator layer, so the attrs field default (4.567 Gyr)
    must be replaced explicitly to trip it.

    Edge: zero, negative, and None all reject; the positive
    documented default rounds trips.
    """
    from proteus.config._star import Mors, Star

    for bad in (0.0, -1.0, None):
        with pytest.raises(ValueError, match=r'(?i)age_now'):
            Star(module='mors', mors=Mors(age_now=bad))
    # Positive default round-trips through the cross-validator.
    s = Star(module='mors', mors=Mors(age_now=4.567))
    assert s.mors.age_now == pytest.approx(4.567, rel=1e-12)


def test_valid_mors_rotation_constraints_both_or_neither_or_out_of_range_raise():
    """``valid_mors`` at ``_star.py:26-38`` enforces "exactly one of
    ``rot_pcntle`` / ``rot_period`` is set", a strictly positive
    period when only the period is set, and a percentile in
    ``[0, 100]`` when only the percentile is set.

    Edge: pin all four rotation-related branches.

    1. Both ``rot_pcntle`` and ``rot_period`` set: collision.
    2. Neither set: missing.
    3. Negative period: invalid value.
    4. Percentile out of ``[0, 100]``: invalid value (the
       percentile-range branch is otherwise uncovered by the
       matrix; pin it here so a regression to ``< 0`` or ``>
       100`` clamping surfaces).

    The documented defaults (``rot_pcntle=50.0``,
    ``rot_period=None``) satisfy the "exactly one set" rule, so a
    regression that flipped one of these defaults would silently
    violate the invariant; the positive round-trip catches that.
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
    # Percentile out of [0, 100]: invalid value. Pin both edges.
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
    # Documented defaults (rot_pcntle=50.0, rot_period=None) round-trip.
    s_ok = Star(module='mors', mors=Mors(spectrum_source='phoenix'))
    assert s_ok.mors.rot_pcntle == pytest.approx(50.0, rel=1e-12)
    assert s_ok.mors.rot_period is None


def test_mors_phoenix_overrides_accept_none_default_and_positive_overrides():
    """``Mors.phoenix_radius``, ``phoenix_log_g`` and
    ``phoenix_Teff`` use ``optional(gt(0))`` with a
    ``none_if_none`` converter so the documented ``None`` default
    rounds trips, a positive override rounds trips, and zero or
    negative raise. A regression that swapped ``optional(gt(0))``
    for plain ``gt(0)`` would reject the documented default.

    Edge: covers all three fields and discriminates the ``None``
    handling that the ``none_if_none`` converter installs.
    """
    from proteus.config._star import Mors

    # None default round-trips.
    default = Mors()
    assert default.phoenix_radius is None
    assert default.phoenix_log_g is None
    assert default.phoenix_Teff is None

    # Positive overrides round-trip.
    m = Mors(phoenix_radius=1.0, phoenix_log_g=4.44, phoenix_Teff=5778.0)
    assert m.phoenix_radius == pytest.approx(1.0, rel=1e-12)
    assert m.phoenix_log_g == pytest.approx(4.44, rel=1e-12)
    assert m.phoenix_Teff == pytest.approx(5778.0, rel=1e-12)

    # Zero rejects on each (gt(0), not ge(0)).
    with pytest.raises(ValueError, match=r'(?i)phoenix_radius'):
        Mors(phoenix_radius=0.0)
    with pytest.raises(ValueError, match=r'(?i)phoenix_log_g'):
        Mors(phoenix_log_g=0.0)
    with pytest.raises(ValueError, match=r'(?i)phoenix_Teff'):
        Mors(phoenix_Teff=0.0)


# ---------------------------------------------------------------------------
# Wrapper-merge contract: calliope per-gas pressures + MORS stellar columns.
# ---------------------------------------------------------------------------


def test_calliope_mors_helpfile_keys_register_outgas_and_stellar_columns():
    """The wrapper merge propagates per-iteration columns from both
    sides into ``hf_row``. The schema MUST register:

    - Calliope per-gas pressures: ``H2O_bar``, ``CO2_bar``,
      ``N2_bar``, ``H2_bar``, ``CO_bar``.
    - MORS stellar columns: ``T_star``, ``R_star``, ``M_star``,
      ``F_ins``, ``F_xuv``.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as a float zero; the type check pins that an
    int-vs-float drift would surface here too.
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
    stellar_keys = ('T_star', 'R_star', 'M_star', 'F_ins', 'F_xuv')
    for key in calliope_per_gas_keys + stellar_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in calliope_per_gas_keys + stellar_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); the relative
        # form of pytest.approx is undefined at zero, so use the
        # absolute form.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
