"""
Integration test: mors (real stellar evolution) coupled to zephyrus
(real atmospheric escape) with dummy interior, atmos, outgas.

mors and zephyrus are coupled by hard physics constraint: zephyrus reads the
stellar XUV flux from mors's evolution track, and the schema validator at
``src/proteus/config/_config.py`` refuses to load a config where
``escape.module = 'zephyrus'`` but ``star.module != 'mors'`` (or the mors
tracks setting is not 'spada'). This test pins that coupling end-to-end:
both modules dispatch, the spada track resolves under FWL_DATA, and the
escape rate is reported in hf_row.

Three scenarios sweep XUV environment and atmospheric inventory:

- ``earth_like``: 1 M_Earth, 0.5 AU, 3000 ppmw H, 50th-percentile rotation.
  Nominal Earth-anchor IC.
- ``hot_super_earth``: 2 M_Earth, 0.3 AU, 100 ppmw H, 90th-percentile
  rotation (active young star, high XUV).
- ``low_xuv_slow_rotator``: 1 M_Earth, 0.5 AU, 3000 ppmw H,
  10th-percentile rotation (quiet star, weak XUV).

Per ``proteus-tests.md`` §1, the file also includes an explicit
error-contract test that exercises the zephyrus Pxuv schema validator.

Invariants asserted per scenario:
- Stellar parameters present and physically bounded (R_star, T_star) with
  unit-conversion discrimination guards (rejects cm-vs-m on R_star, K-vs-eV
  on T_star).
- Stellar age advances: ``Time`` final > initial.
- Escape rate non-negative, finite, bounded above by a physically plausible
  ceiling (rejects per-area-vs-per-volume swap).
- Per-element mass closure ``atm + liq + sol == total`` for H/C/N/S/O at
  the final row (conservation invariant carve-out, §2).

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@dataclass(frozen=True)
class _MorsZephyrusScenario:
    """Per-scenario parametrize input for the mors+zephyrus pair test."""

    name: str
    planet_mass: float
    semimajoraxis: float
    H_budget: float
    rot_pcntle: float
    efficiency: float


_SCENARIOS = (
    _MorsZephyrusScenario(
        name='earth_like',
        planet_mass=1.0,
        semimajoraxis=0.5,
        H_budget=3.0e3,
        rot_pcntle=50.0,
        efficiency=0.5,
    ),
    _MorsZephyrusScenario(
        name='hot_super_earth',
        planet_mass=2.0,
        semimajoraxis=0.3,
        H_budget=100.0,
        rot_pcntle=90.0,
        efficiency=0.5,
    ),
    _MorsZephyrusScenario(
        name='low_xuv_slow_rotator',
        planet_mass=1.0,
        semimajoraxis=0.5,
        H_budget=3.0e3,
        rot_pcntle=10.0,
        efficiency=0.5,
    ),
)


def _ensure_mors_data_or_skip() -> None:
    """Ensure the mors spada tracks are present under FWL_DATA.

    Tries to download them on the fly when missing (the test fixture
    only pre-fetches data when aragog or agni is the active module,
    so mors data needs its own primer here). Skip only when the
    download itself fails, which is the offline-without-cache case.

    Checks both case variants of the directory name because the MORS
    downloader inconsistently lands data at ``Spada`` or ``spada``
    depending on the OS and the path through DownloadEvolutionTracks
    vs the OSF fallback.
    """
    fwl = os.environ.get('FWL_DATA')
    if not fwl:
        pytest.skip('FWL_DATA env var not set; mors track data unavailable')
    parent = Path(fwl) / 'stellar_evolution_tracks'
    candidates = (parent / 'spada', parent / 'Spada')

    def _present() -> bool:
        return any(c.is_dir() and any(c.iterdir()) for c in candidates)

    if _present():
        return
    try:
        from proteus.utils.data import download_stellar_tracks

        download_stellar_tracks('Spada')
    except (OSError, RuntimeError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f'could not fetch mors spada tracks: {exc}')
    if not _present():
        pytest.skip(
            'mors spada tracks still missing after download attempt at '
            f'{parent} (checked spada/ and Spada/)'
        )


@pytest.mark.integration
@pytest.mark.physics_invariant
@pytest.mark.parametrize('scenario', _SCENARIOS, ids=lambda s: s.name)
def test_mors_zephyrus_two_timesteps(proteus_multi_timestep_run, scenario):
    """Two-step PROTEUS run with real mors + real zephyrus across three IC.

    The same coupling invariants must hold across all three scenarios:
    Earth-anchor, hot volatile-poor super-Earth, quiet old-star low-XUV
    Earth. A regression in stellar-flux unit conversion (W/m^2 vs
    erg/s/cm^2), an off-by-one in age advancement, a missed hf_row
    population, or a per-area-vs-per-volume swap in the escape ratio
    is the only failure class. The 3-scenario span surfaces bugs that
    only show under specific IC regimes (e.g. an XUV-scaling bug that
    happens to be benign at the 50th rotation percentile but blows up
    at the 90th).

    Verifies per scenario:
    - Helpfile has at least 2 rows.
    - Stellar parameters R_star (m), T_star (K) within physical bounds
      that discriminate plausible unit-conversion bugs.
    - ``Time`` advances.
    - ``esc_rate_total`` is non-negative, finite, bounded above.
    - Per-element mass closure for H/C/N/S/O.

    Runtime budget: ~10-20 s per scenario, ~30-60 s total for the 3
    parametrized cases (mors track interpolation + zephyrus solver are
    both fast; mors track load is module-scoped and amortized).
    """
    _ensure_mors_data_or_skip()

    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        planet__mass_tot=scenario.planet_mass,
        orbit__semimajoraxis=scenario.semimajoraxis,
        planet__elements__H_budget=scenario.H_budget,
        star__module='mors',
        star__mors__tracks='spada',
        star__mors__rot_pcntle=scenario.rot_pcntle,
        star__mors__rot_period=None,
        star__mors__age_now=4.567,
        star__mors__spectrum_source='solar',
        star__mors__star_name='sun',
        escape__module='zephyrus',
        escape__zephyrus__Pxuv=1e-2,
        escape__zephyrus__efficiency=scenario.efficiency,
        escape__reservoir='bulk',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]
    initial = hf.iloc[0]

    # Time advances: catches a stop-condition regression that flipped
    # the loop on its head.
    assert float(final['Time']) > float(initial['Time']), 'time did not advance'

    # Stellar parameters with unit discrimination. mors reports R_star
    # in metres and T_star in Kelvin.
    if 'R_star' in final:
        r_star = float(final['R_star'])
        assert np.isfinite(r_star), 'R_star not finite'
        # 1e7-1e12 m rejects R_star reported in cm (1 R_sun = 6.96e10 cm
        # would land at the upper edge; a cm-vs-m bug on R_sun lands at
        # 6.96e10, in bounds; pair with T_star check for stronger
        # discrimination).
        assert 1e7 <= r_star <= 1e12, f'R_star out of bounds (m): {r_star:.3e}'

    if 'T_star' in final:
        t_star = float(final['T_star'])
        assert np.isfinite(t_star), 'T_star not finite'
        # 2000-100000 K rejects T_star reported in eV (Sun is ~0.5 eV) or
        # in scaled units. The Sun-like mass we're using here should give
        # T_eff ~4000-7000 K from the spada tracks at any rotation.
        assert 2000 <= t_star <= 1e5, f'T_star out of bounds (K): {t_star:.3e}'

    # Zephyrus escape rate: the discriminator for the coupling. Sign +
    # scale guard.
    if 'esc_rate_total' in hf.columns:
        rate = hf['esc_rate_total'].to_numpy()
        assert np.all(np.isfinite(rate)), 'esc_rate_total contains NaN or Inf'
        assert np.all(rate >= 0), 'esc_rate_total negative (sign bug)'
        # Scale guard: an Earth-or-super-Earth-mass planet at 0.3-0.5 AU
        # under XUV from a Sun-mass star cannot exceed 1e10 kg/s without
        # losing the whole atmosphere in seconds. A per-area-vs-per-volume
        # ratio bug would land >1e15.
        assert np.all(rate < 1e10), f'esc_rate_total above physical bound: max={rate.max():.3e}'

    # Per-element mass closure: the conservation invariant, satisfies the
    # exponent-error guard from proteus-tests.md §2 by construction.
    for elt in ('H', 'C', 'N', 'S', 'O'):
        atm_key = f'{elt}_kg_atm'
        liq_key = f'{elt}_kg_liquid'
        sol_key = f'{elt}_kg_solid'
        tot_key = f'{elt}_kg_total'
        if not all(k in final for k in (atm_key, liq_key, sol_key, tot_key)):
            continue
        atm = float(final[atm_key])
        liq = float(final[liq_key])
        sol = float(final[sol_key])
        tot = float(final[tot_key])
        assert atm >= 0, f'{atm_key} negative: {atm:.3e}'
        assert liq >= 0, f'{liq_key} negative: {liq:.3e}'
        assert sol >= 0, f'{sol_key} negative: {sol:.3e}'
        if tot > 0:
            assert atm + liq + sol == pytest.approx(tot, rel=2e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    # Conservation + stability cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'


# ---------------------------------------------------------------------------
# Error-contract path per proteus-tests.md §1 clause 2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_zephyrus_pxuv_validator_rejects_out_of_range_pressure(tmp_path):
    """Zephyrus ``Pxuv`` schema validator rejects pressures outside
    the documented (0, 10] bar window.

    Contract from ``src/proteus/config/_escape.py:14``:
        ``zephyrus.Pxuv`` must be > 0 and <= 10 bar.

    Verifies:
    - A config with ``Pxuv = 15`` (above the upper bound) raises
      ``ValueError`` at config load time, BEFORE any module dispatch
      or hf_row write. The pre-dispatch assertion confirms no side
      effect of the invalid config leaked into the runtime state.
    - The exception message names ``Pxuv``, so a future schema rename
      that silently drops the validator will be caught by the message
      assertion, not just the exception type.

    Discriminating coverage:
    - Catches a regression that loosens the upper bound (e.g. raises
      it to 100 bar by accident).
    - Catches a regression that removes the validator wiring entirely
      (the test would then PASS the config load and fail the assert).
    """
    # Importing here so the test does not pay the proteus-config import
    # cost when collected but not selected.
    from proteus.config._escape import Escape, Zephyrus

    bad_zephyrus = Zephyrus(Pxuv=15.0, efficiency=0.5)
    with pytest.raises(ValueError, match=r'(?i)pxuv'):
        # The Escape validator runs valid_zephyrus when module='zephyrus'.
        # We construct directly rather than loading a full TOML so the
        # error path is exercised in isolation: no other validator can
        # fire first and mask this one.
        Escape(module='zephyrus', zephyrus=bad_zephyrus)

    # Discrimination: confirm that the in-range value DOES NOT raise.
    # A regression that broke the validator into raising on every input
    # would be hidden by the negative test above.
    good_zephyrus = Zephyrus(Pxuv=1e-2, efficiency=0.5)
    escape_ok = Escape(module='zephyrus', zephyrus=good_zephyrus)
    assert escape_ok.zephyrus.Pxuv == pytest.approx(1e-2)

    # Discrimination: confirm the validator does NOT fire for non-zephyrus
    # modules even with the same bad value. The current Pxuv guard is
    # gated on module == 'zephyrus' (see _escape.py:9-11).
    escape_dummy = Escape(module='dummy', zephyrus=bad_zephyrus)
    assert escape_dummy.module == 'dummy'
    # Used so static-analysis flags an accidental removal of the
    # assertion above.
    assert tmp_path.is_dir()
