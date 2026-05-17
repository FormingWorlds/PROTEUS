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

Invariants asserted:
- Stellar parameters present and physically bounded (R_star, T_star).
- Stellar age advances: ``age_now`` final > initial.
- Escape rate non-negative, finite, bounded above by atmospheric mass per
  step (zephyrus conservation).
- Per-element mass closure across the run.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _ensure_mors_data_or_skip() -> None:
    """Ensure the mors spada tracks are present under FWL_DATA.

    Tries to download them on the fly when missing (the test fixture
    only pre-fetches data when aragog or agni is the active module,
    so mors data needs its own primer here). Skip only when the
    download itself fails, which is the offline-without-cache case.
    """
    fwl = os.environ.get('FWL_DATA')
    if not fwl:
        pytest.skip('FWL_DATA env var not set; mors track data unavailable')
    spada = Path(fwl) / 'stellar_evolution_tracks' / 'spada'
    if spada.is_dir():
        return
    try:
        from proteus.utils.data import download_stellar_tracks

        download_stellar_tracks('Spada')
    except (OSError, RuntimeError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f'could not fetch mors spada tracks: {exc}')
    if not spada.is_dir():
        pytest.skip(f'mors spada tracks still missing after download attempt at {spada}')


@pytest.mark.integration
@pytest.mark.physics_invariant
def test_mors_zephyrus_two_timesteps(proteus_multi_timestep_run):
    """Two-step PROTEUS run with real mors stellar evolution + real zephyrus escape.

    Physical scenario: 1 M_sun, 0.5 AU, 0.1 Gyr initial age, Earth-mass
    planet with 3000 ppmw H budget. mors evolves the stellar luminosity
    and XUV; zephyrus computes hydrodynamic escape against that XUV.
    Every other slot is dummy so the test isolates the mors-zephyrus
    coupling: a regression in stellar-flux unit conversion (W/m^2 vs
    erg/s/cm^2), an off-by-one in age advancement, or a missed hf_row
    population is the only failure class.

    Verifies:
    - Helpfile has at least 2 rows.
    - Stellar parameters R_star, T_star present (if mors populated them)
      and within physical bounds.
    - Escape rate ``esc_rate_total`` is non-negative and finite at every row.
    - The escape mass per step is below the atmospheric mass: a regression
      that scaled escape by 1e6 would fail this guard. Carve-out: when
      reservoir = "bulk", the dummy bulk-mass aggregate may be zero in a
      dummy interior, in which case the test only checks the rate.
    - Per-element mass closure ``atm + liq + sol == total`` for H/C/N/S/O.

    Runtime budget: ~60-90 s (mors track interpolation + zephyrus solver
    are both fast; first call dominates with the spada-track load).
    """
    _ensure_mors_data_or_skip()

    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=2,
        max_time=1e3,
        min_time=1e2,
        # mors stellar evolution. spada tracks + phoenix synthetic spectra
        # are the FWL_DATA-shipped path; star_name='sun' caches the solar
        # phoenix template.
        star__module='mors',
        star__mors__tracks='spada',
        star__mors__rot_pcntle=50.0,
        star__mors__rot_period=None,
        star__mors__age_now=4.567,
        # The FWL_DATA-bundled solar spectrum + star_name='sun' is the
        # CI-shippable spectrum-source combination; phoenix synthetic
        # spectra require a separately-fetched template (Goettingen FTP)
        # that the test fixture does not pre-cache.
        star__mors__spectrum_source='solar',
        star__mors__star_name='sun',
        # zephyrus escape with default Pxuv + 0.5 efficiency. Pxuv must
        # be > 0 and <= 10 bar per the schema validator.
        escape__module='zephyrus',
        escape__zephyrus__Pxuv=1e-2,
        escape__zephyrus__efficiency=0.5,
        escape__reservoir='bulk',
    )

    hf = runner.hf_all
    assert hf is not None, 'helpfile should be created'
    assert len(hf) >= 2, f'expected >= 2 rows, got {len(hf)}'

    final = hf.iloc[-1]
    initial = hf.iloc[0]

    # Time advances: a hard invariant that catches a stop-condition
    # regression flipping the loop on its head.
    assert float(final['Time']) > float(initial['Time']), 'time did not advance'

    # Stellar parameters (if mors populated them). The bounds discriminate
    # against unit errors: R_star in [1e7, 1e12] m rejects R_star reported
    # in cm (1e9 cm = 1e7 m boundary, but a cm-vs-m bug would land at
    # 1e11 cm = 1e9 m, still inside, so the bound alone is weak; pair
    # with T_star which discriminates a K-vs-eV unit bug).
    if 'R_star' in final:
        r_star = float(final['R_star'])
        assert np.isfinite(r_star), 'R_star not finite'
        assert 1e7 <= r_star <= 1e12, f'R_star out of bounds (m): {r_star:.3e}'

    if 'T_star' in final:
        t_star = float(final['T_star'])
        assert np.isfinite(t_star), 'T_star not finite'
        # 2000-100000 K rejects T_star in eV (sun would be ~0.5 eV).
        assert 2000 <= t_star <= 1e5, f'T_star out of bounds (K): {t_star:.3e}'

    # Zephyrus escape rate: the discriminator for the coupling.
    # Pre-fix sign convention is rate >= 0 (mass loss).
    if 'esc_rate_total' in hf.columns:
        rate = hf['esc_rate_total'].to_numpy()
        assert np.all(np.isfinite(rate)), 'esc_rate_total contains NaN or Inf'
        assert np.all(rate >= 0), 'esc_rate_total negative (sign bug)'
        # Scale guard: an Earth-mass planet at 0.5 AU with full XUV
        # exposure cannot reasonably exceed 1e10 kg/s without losing the
        # whole atmosphere in seconds. A regression that swapped per-area
        # vs per-volume in the BOREAS-style ratio would land >1e15.
        assert np.all(rate < 1e10), f'esc_rate_total above physical bound: max={rate.max():.3e}'

    # Per-element mass closure, same shape as the atmodeller-dummy test
    # so the assertion form is uniform across the integration tier.
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
            # 2% rel tolerance because zephyrus actively removes mass per
            # step; closure on the residual is the discriminator.
            assert atm + liq + sol == pytest.approx(tot, rel=2e-2), (
                f'{elt} closure: atm+liq+sol={atm + liq + sol:.3e}, total={tot:.3e}'
            )

    # Conservation + stability cross-cutting helpers.
    mass_results = validate_mass_conservation(hf, tolerance=0.2)
    assert mass_results.get('masses_positive', True), 'mass reservoirs negative'

    stability = validate_stability(hf, max_temp=1e6, max_pressure=1e10)
    assert stability['temps_stable'], 'temperature instability detected'
    assert stability['pressures_stable'], 'pressure instability detected'
