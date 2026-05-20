"""Integration test: aragog (real interior) coupled to MORS (real star).

MORS provides time-evolving stellar spectra; aragog provides the
production interior-energetics solver. The pair sits inside any
real-physics PROTEUS run where the dummy atmosphere is replaced by
JANUS or AGNI; this file pins the per-iteration boundary state at
the integration tier without booting Julia or running the real
Aragog solver.

Integration-tier scope:

- Schema validators round-trip ``interior_energetics.module=
  'aragog'`` with ``star.module='mors'``.
- Aragog backend enum is exactly ``{'jax', 'numpy'}`` pinned as a
  set; the production default 'jax' round-trips.
- Aragog core_bc enum is ``{'quasi_steady', 'energy_balance',
  'gradient', 'bower2018'}`` pinned as a set; default is
  'energy_balance'.
- Aragog phase_smoothing enum is ``{'tanh', 'cubic_hermite'}``
  pinned as a set; default is 'tanh'.
- Mors tracks enum and Mors spectrum_source enum (the MORS-side
  contracts; ensures the pair-test catches MORS drift even without
  AGNI in the loop).
- ``mors.age_now`` enforced strictly positive via the valid_mors
  cross-validator.
- The wrapper merge guard pins both the interior columns
  (T_magma, Phi_global, F_int, F_cmb) and the stellar columns
  (T_star, R_star, M_star, F_ins, F_xuv) in GetHelpfileKeys so
  per-iteration values flow into the helpfile.

The full two-timestep aragog + MORS coupled run is exercised by
the slow-tier ``test_slow_aragog_calliope.py`` (with calliope
outgas) at the production aragog backend.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


# ---------------------------------------------------------------------------
# Schema-validator round-trips for the (aragog, mors) production combo.
# ---------------------------------------------------------------------------


def test_interior_aragog_and_star_mors_both_round_trip_through_schema():
    """``interior_energetics.module='aragog'`` and ``star.module='mors'``
    both round-trip without raising.

    Discrimination: reject an obviously-wrong interior_energetics
    module to confirm the validator still fires, and confirm the
    Mors default Star round-trips (so a regression in valid_mors
    that broke the default config would surface here).
    """
    from proteus.config._interior import Interior
    from proteus.config._star import Star

    i = Interior(module='aragog')
    assert i.module == 'aragog'
    # Aragog backend default round-trips.
    assert i.aragog.backend == 'jax'
    # An invalid interior_energetics module rejects.
    with pytest.raises(ValueError, match=r'(?i)module'):
        Interior(module='not_a_real_interior_backend')
    # Mors default Star round-trips.
    s = Star(module='mors')
    assert s.module == 'mors'
    assert s.mors.tracks == 'spada'
    assert s.mors.spectrum_source == 'phoenix'


def test_aragog_backend_enum_pinned_as_set():
    """Pin the Aragog.backend enum as ``{'jax', 'numpy'}``. A
    regression that added a third backend silently (e.g. 'diffrax')
    would still let 'jax' and 'numpy' round-trip and would still
    reject an obvious typo, so the set check is the only way to
    catch enum drift.
    """
    import attrs

    from proteus.config._interior import Aragog

    allowed = attrs.fields(Aragog).backend.validator.options
    assert set(allowed) == {'jax', 'numpy'}, (
        f'Aragog.backend enum drifted from documented set: {allowed}'
    )
    for known in ('jax', 'numpy'):
        a = Aragog(backend=known)
        assert a.backend == known
    with pytest.raises(ValueError, match=r'(?i)backend'):
        Aragog(backend='diffrax_research_only')


def test_aragog_core_bc_enum_pinned_as_set():
    """Pin the Aragog.core_bc enum as ``{'quasi_steady',
    'energy_balance', 'gradient', 'bower2018'}``. The default is
    'energy_balance' (SPIDER-parity BC); legacy 'quasi_steady'
    still rounds-trips for back-compatibility runs.
    """
    import attrs

    from proteus.config._interior import Aragog

    allowed = attrs.fields(Aragog).core_bc.validator.options
    documented = {'quasi_steady', 'energy_balance', 'gradient', 'bower2018'}
    assert set(allowed) == documented, (
        f'Aragog.core_bc enum drifted from documented set: {allowed}'
    )
    for known in documented:
        a = Aragog(core_bc=known)
        assert a.core_bc == known
    with pytest.raises(ValueError, match=r'(?i)core_bc'):
        Aragog(core_bc='free_boundary')
    # Default is SPIDER-parity 'energy_balance'.
    assert Aragog().core_bc == 'energy_balance'


def test_aragog_phase_smoothing_enum_pinned_as_set():
    """Pin the Aragog.phase_smoothing enum as ``{'tanh',
    'cubic_hermite'}``. Default is 'tanh' (SPIDER parity).
    """
    import attrs

    from proteus.config._interior import Aragog

    allowed = attrs.fields(Aragog).phase_smoothing.validator.options
    assert set(allowed) == {'tanh', 'cubic_hermite'}, (
        f'Aragog.phase_smoothing enum drifted from documented set: {allowed}'
    )
    for known in ('tanh', 'cubic_hermite'):
        a = Aragog(phase_smoothing=known)
        assert a.phase_smoothing == known
    with pytest.raises(ValueError, match=r'(?i)phase_smoothing'):
        Aragog(phase_smoothing='polynomial')
    assert Aragog().phase_smoothing == 'tanh'


def test_aragog_solver_method_enum_pinned_as_set():
    """Pin the Aragog.solver_method enum as ``{'cvode', 'radau',
    'bdf'}``. Default is 'cvode' (SUNDIALS, SPIDER parity).

    Discrimination: a regression that silently added a fourth
    solver (e.g. 'diffrax_research_only' graduating from the
    backend flag to a first-class solver_method) would still let
    cvode/radau/bdf round-trip; the set check is the only way to
    catch the drift.
    """
    import attrs

    from proteus.config._interior import Aragog

    allowed = attrs.fields(Aragog).solver_method.validator.options
    assert set(allowed) == {'cvode', 'radau', 'bdf'}, (
        f'Aragog.solver_method enum drifted from documented set: {allowed}'
    )
    for known in ('cvode', 'radau', 'bdf'):
        a = Aragog(solver_method=known)
        assert a.solver_method == known
    with pytest.raises(ValueError, match=r'(?i)solver_method'):
        Aragog(solver_method='diffrax_research_only')
    # SPIDER-parity default.
    assert Aragog().solver_method == 'cvode'


def test_aragog_atol_temperature_equivalent_must_be_positive():
    """``Aragog.atol_temperature_equivalent`` uses ``gt(0)`` at
    ``_interior.py:156``. The default is 1e-8 K (SPIDER parity);
    zero and negative reject.

    Discrimination: a regression that swapped ``gt(0)`` for
    ``ge(0)`` would accept zero; the explicit zero-raise rejects
    that.
    """
    from proteus.config._interior import Aragog

    with pytest.raises(ValueError, match=r'(?i)atol_temperature_equivalent'):
        Aragog(atol_temperature_equivalent=0.0)
    with pytest.raises(ValueError, match=r'(?i)atol_temperature_equivalent'):
        Aragog(atol_temperature_equivalent=-1e-12)
    default = Aragog()
    assert default.atol_temperature_equivalent == pytest.approx(1.0e-8, rel=1e-12)


# ---------------------------------------------------------------------------
# MORS-side: tracks and spectrum_source enums + age_now positivity.
# ---------------------------------------------------------------------------


def test_mors_tracks_and_spectrum_source_enums_pinned_for_pair():
    """Even without AGNI in this pair, the MORS-side schema must
    round-trip cleanly. Re-pin both enums as sets so a regression
    in MORS drift surfaces here too (the matrix would otherwise
    only catch this in the AGNI x MORS file).
    """
    import attrs

    from proteus.config._star import Mors

    tracks_allowed = attrs.fields(Mors).tracks.validator.options
    assert set(tracks_allowed) == {'spada', 'baraffe'}
    src_allowed = attrs.fields(Mors).spectrum_source.validator.options
    assert set(src_allowed) == {'solar', 'muscles', 'phoenix', None}
    # Defaults match the documented production combination.
    default = Mors()
    assert default.tracks == 'spada'
    assert default.spectrum_source == 'phoenix'
    assert default.age_now == pytest.approx(4.567, rel=1e-12)


def test_mors_age_now_positivity_at_valid_mors_layer():
    """``valid_mors`` at ``_star.py:13-14`` raises when
    ``mors.age_now`` is None or <=0. Zero, negative, and None all
    reject when the star module is 'mors'.
    """
    from proteus.config._star import Mors, Star

    for bad in (0.0, -1.0, None):
        with pytest.raises(ValueError, match=r'(?i)age_now'):
            Star(module='mors', mors=Mors(age_now=bad))


# ---------------------------------------------------------------------------
# Wrapper-merge contract: aragog interior columns + MORS stellar columns.
# ---------------------------------------------------------------------------


def test_aragog_mors_helpfile_keys_register_interior_and_stellar_columns():
    """The wrapper merge propagates per-iteration columns from both
    sides into ``hf_row``. The schema MUST register:

    - Interior: ``T_magma``, ``Phi_global``, ``F_int``, ``F_cmb``.
    - Stellar: ``T_star``, ``R_star``, ``M_star``, ``F_ins``,
      ``F_xuv``.

    Discrimination: every key tested separately so a regression
    that dropped any one fails the per-key loop. ZeroHelpfileRow
    seeds each as float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    interior_keys = ('T_magma', 'Phi_global', 'F_int', 'F_cmb')
    stellar_keys = ('T_star', 'R_star', 'M_star', 'F_ins', 'F_xuv')
    for key in interior_keys + stellar_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in interior_keys + stellar_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); use pytest.approx
        # with an absolute tolerance because the relative form is
        # meaningless when the expected value is exactly zero.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
