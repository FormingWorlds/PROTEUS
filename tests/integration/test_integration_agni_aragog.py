"""Integration test: AGNI (real atmosphere) coupled to aragog (real interior).

Per-iteration coupling: AGNI computes F_atm and feeds it as the
boundary condition to the aragog entropy solver; aragog computes
T_magma and feeds it back to AGNI as the surface temperature. This
file exercises the integration-tier portions of that boundary:

- Pair-wise schema validators round-trip `atmos_clim.module='agni'`
  with `interior_energetics.module='aragog'` (the production
  combination at the Paper-1 paper-3 fiducial).
- The AGNI optical-depth aggregator `_summarise_tau_band` returns
  scalars suitable for direct insertion into `hf_row` via the wrapper
  merge. The matrix design lock (commit b8021c33) requires every
  AGNI-side integration to assert the optical-depth monotonicity
  invariant `tau_atm_TOA < 0.5 * tau_atm_surface`.
- The wrapper merge propagates the four AGNI 1.10.2 diagnostic keys
  (`tau_atm_TOA`, `tau_atm_surface`, `agni_Ra_max`,
  `agni_t_conv_over_t_rad`) from the atmosphere output dict into
  `hf_row` via the registered `GetHelpfileKeys` columns.
- The aragog backend pin (`backend='jax'`, the production default in
  `config/_interior.py`) round-trips against the schema.

The full two-timestep AGNI + aragog coupled run with real Julia +
real CVode + real JAX would land at ~30 min on Linux GHA, which sits
above the slow-tier per-step budget. The slow-tier sibling lives in
``test_slow_aragog_atmodeller.py`` for the aragog interior leg; the
AGNI-specific real-binary coupling is exercised by the existing
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
# Schema-validator round-trip for the (agni, aragog) production combination.
# ---------------------------------------------------------------------------


def test_atmos_clim_agni_module_round_trips_through_schema():
    """``atmos_clim.module='agni'`` is in the documented enum and must
    round-trip through the attrs validator without raising.

    Discrimination: also confirm 'janus' and 'dummy' round-trip (rules
    out a regression that broke the validator into raising on every
    input) and 'totally_invalid_backend' is rejected (rules out a
    regression that disabled the validator entirely).
    """
    from proteus.config._atmos_clim import AtmosClim

    for known in ('agni', 'janus'):
        ac = AtmosClim(module=known)
        assert ac.module == known
    # 'dummy' carries an additional cross-validator: dummy atmos_clim
    # is incompatible with the default rayleigh=True, so we must
    # disable Rayleigh scattering in the dummy round-trip.
    dummy_ac = AtmosClim(module='dummy', rayleigh=False)
    assert dummy_ac.module == 'dummy'
    with pytest.raises(ValueError, match=r'(?i)module'):
        AtmosClim(module='totally_invalid_backend')


def test_interior_energetics_aragog_module_round_trips_through_schema():
    """``interior_energetics.module='aragog'`` is in the enum and the
    documented aragog backend ``'jax'`` round-trips through the
    Interior schema.

    Discrimination: confirm 'numpy' (the non-JAX fallback) also
    round-trips, and an invalid backend is rejected.
    """
    from proteus.config._interior import Interior

    i = Interior(module='aragog')
    assert i.module == 'aragog'
    # backend selector lives on the aragog sub-config; confirm
    # production 'jax' default plus the 'numpy' fallback both
    # round-trip.
    assert i.aragog.backend in ('jax', 'numpy'), (
        f'aragog default backend unexpectedly outside enum: {i.aragog.backend!r}'
    )
    from proteus.config._interior import Aragog

    for known in ('jax', 'numpy'):
        a = Aragog(backend=known)
        assert a.backend == known
    with pytest.raises(ValueError, match=r'(?i)backend'):
        Aragog(backend='no_such_backend')


# ---------------------------------------------------------------------------
# Optical-depth monotonicity from the AGNI-side aggregator.
# Matrix design lock: every AGNI x X integration test must assert this.
# ---------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_agni_aragog_optical_depth_monotonic_from_TOA_to_surface():
    """Drive ``_summarise_tau_band`` with a tau profile that increases
    monotonically with depth (per band) and confirm the aggregated
    hf_row scalars satisfy `tau_atm_TOA < 0.5 * tau_atm_surface`.

    Physical scenario: a thick CO2-N2 atmosphere over a partially-
    molten mantle with non-trivial gas opacity. ``tau_band`` from
    AGNI is integrated from TOA downwards, so the value at TOA is
    near zero and grows with optical depth toward the surface.

    Discrimination guard: the wrong-direction-of-integration
    regression would put the largest tau at TOA and the smallest at
    the surface, failing the strict inequality. The `< 0.5 *
    tau_atm_surface` guard rejects a regression that shrunk the gap
    rather than inverting it.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    # tau_band shape: (nlev_c, nbands). Pick four levels (TOA -> surface)
    # and three bands. The values grow with depth in every band.
    tau_band = np.array(
        [
            [0.01, 0.02, 0.005],  # TOA
            [0.3, 0.5, 0.2],
            [1.5, 2.0, 1.0],
            [5.0, 8.0, 4.0],  # surface
        ]
    )
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)

    # Closed-form: per-row means are 0.01166..., 0.333..., 1.5, 5.666....
    assert tau_TOA == pytest.approx(0.011666666666666667, rel=1e-12)
    assert tau_surface == pytest.approx(5.666666666666667, rel=1e-12)
    # Monotonicity invariant (the matrix design lock).
    assert tau_TOA < tau_surface
    # Scale guard against a regression that shrank rather than
    # inverted the gap: tau_TOA must be well below half the surface
    # value.
    assert tau_TOA < 0.5 * tau_surface


@pytest.mark.physics_invariant
def test_agni_aragog_optical_depth_floor_at_zero():
    """A perfectly-transparent atmosphere (tau_band = 0 everywhere)
    must aggregate to tau_atm_TOA == tau_atm_surface == 0.

    Edge: limit-input case. Discrimination: a regression that
    contaminated the aggregator with an additive offset would land
    at a non-zero value at one or both endpoints; the strict-zero
    pin catches both.
    """
    from proteus.atmos_clim.agni import _summarise_tau_band

    tau_band = np.zeros((4, 3))
    atmos = SimpleNamespace(tau_band=tau_band, nlev_c=4, nbands=3)
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    assert tau_TOA == pytest.approx(0.0, abs=1e-30)
    assert tau_surface == pytest.approx(0.0, abs=1e-30)
    # Boundedness: optical depth is non-negative.
    assert tau_TOA >= 0.0
    assert tau_surface >= 0.0


# ---------------------------------------------------------------------------
# AGNI -> hf_row -> aragog: the wrapper-merge contract.
# ---------------------------------------------------------------------------


def test_agni_aragog_wrapper_merge_propagates_optical_depth_keys_to_hf_row():
    """The atmos_clim wrapper at ``proteus/atmos_clim/wrapper.py:196-198``
    copies AGNI output keys into hf_row only when those keys are
    already present in hf_row. The four AGNI 1.10.2 diagnostics
    (``tau_atm_TOA``, ``tau_atm_surface``, ``agni_Ra_max``,
    ``agni_t_conv_over_t_rad``) must therefore be registered in
    ``GetHelpfileKeys()``.

    Discrimination: drive a fresh hf_row through ``ZeroHelpfileRow``
    (which builds from ``GetHelpfileKeys``) and assert all four
    diagnostic keys are present with float zero defaults. A
    regression that dropped any of them from ``GetHelpfileKeys``
    would cause the wrapper merge guard to silently discard the
    corresponding diagnostic on every iteration.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    for key in (
        'tau_atm_TOA',
        'tau_atm_surface',
        'agni_Ra_max',
        'agni_t_conv_over_t_rad',
    ):
        assert key in keys, (
            f'{key} must be registered in GetHelpfileKeys() so the wrapper '
            f'merge guard at atmos_clim/wrapper.py:197 propagates it.'
        )
    # ZeroHelpfileRow round-trip: the keys are initialised to 0.0
    # (float) so a fresh helpfile row can accept an AGNI assignment.
    row = ZeroHelpfileRow()
    for key in (
        'tau_atm_TOA',
        'tau_atm_surface',
        'agni_Ra_max',
        'agni_t_conv_over_t_rad',
    ):
        assert key in row
        assert row[key] == pytest.approx(0.0, abs=1e-12)
        assert isinstance(row[key], float)


@pytest.mark.physics_invariant
def test_agni_aragog_diagnostic_summarisers_emit_finite_or_nan_only():
    """The two AGNI diagnostic summarisers must emit finite floats
    on a well-formed input and NaN-only on a degenerate input. This
    is the contract the wrapper layer depends on to keep hf_row
    well-formed regardless of AGNI's solver state.

    Discrimination: a regression that returned None or raised on a
    degenerate input would corrupt the helpfile CSV with sentinels
    that downstream readers cannot parse. The math.isnan guard pins
    that the fallback is specifically NaN, not zero or None.
    """
    import math

    from proteus.atmos_clim.agni import _summarise_diagnostics, _summarise_tau_band

    # Well-formed AGNI return: every required field present.
    good = SimpleNamespace(
        tau_band=np.array([[0.01, 0.02], [1.0, 1.5], [5.0, 6.0]]),
        nlev_c=3,
        nbands=2,
        diagnostic_Ra=np.array([0.5, 4.0, 2.0]),
        timescale_conv=np.array([0.0, 1.0e3, 5.0e3]),
        timescale_rad=np.array([1.0e6, 1.0e6, 1.0e6]),
        mask_c=np.array([False, True, True]),
    )
    tau_TOA, tau_surface = _summarise_tau_band(good)
    Ra_max, ratio = _summarise_diagnostics(good)
    for v in (tau_TOA, tau_surface, Ra_max, ratio):
        assert math.isfinite(v), f'expected finite, got {v}'
    # Sign + bounds: every diagnostic must be non-negative on a
    # physical input.
    assert tau_TOA >= 0
    assert tau_surface >= 0
    assert Ra_max >= 0
    assert ratio > 0

    # Degenerate input for tau aggregator: zero-size tau_band -> NaN.
    empty = SimpleNamespace(tau_band=np.zeros((0, 0)), nlev_c=0, nbands=0)
    tau_TOA_empty, tau_surface_empty = _summarise_tau_band(empty)
    assert math.isnan(tau_TOA_empty)
    assert math.isnan(tau_surface_empty)

    # Degenerate input for diagnostics aggregator: purely radiative
    # atmosphere (mask_c all False). Ra_max stays finite from the
    # diagnostic array; the convective/radiative timescale ratio is
    # NaN because there is no convective level to anchor at.
    no_conv = SimpleNamespace(
        tau_band=np.ones((3, 2)),
        nlev_c=3,
        nbands=2,
        diagnostic_Ra=np.array([1.0, 2.0, 3.0]),
        timescale_conv=np.zeros(3),
        timescale_rad=np.ones(3) * 1e6,
        mask_c=np.zeros(3, dtype=bool),
    )
    Ra_no_conv, ratio_no_conv = _summarise_diagnostics(no_conv)
    assert math.isfinite(Ra_no_conv)
    assert Ra_no_conv == pytest.approx(3.0, rel=1e-12)
    assert math.isnan(ratio_no_conv)
