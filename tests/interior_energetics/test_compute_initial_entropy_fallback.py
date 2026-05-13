"""Unit tests for the energetics-side P_cmb fallback in
``proteus.interior_energetics.common.compute_initial_entropy``.

Verifies the 2026-04-29 fix that replaces the hardcoded 135 GPa
Earth-only fallback with a Noack & Lasbleis (2020) mass-aware estimate.
The previous implementation raised ``ValueError`` for any
``mass_tot`` outside [0.5, 2.0] M_Earth; the new implementation
accepts the full 0.5-10 M_Earth band and logs the NL20 result.

We test by capturing the warning log emitted before the downstream
EntropyEOS / Zalmoxis-adiabat path runs. The downstream path needs
real EOS data files we don't want to mock, so we let it raise its
own (non-ValueError) failure mode and confirm only the *guard*
behaviour at the top of the function.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_minimal_config(
    mass_tot=1.0,
    core_frac=0.325,
    core_frac_mode='mass',
    delta_T_super=500.0,
):
    """Stub config exposing the fields compute_initial_entropy reads."""
    cfg = MagicMock()
    cfg.planet.mass_tot = mass_tot
    cfg.planet.delta_T_super = delta_T_super
    cfg.planet.tcmb_init = 6000.0
    cfg.planet.temperature_mode = 'liquidus_super'
    cfg.planet.ini_entropy = 3000.0
    cfg.interior_struct.core_frac = core_frac
    cfg.interior_struct.core_frac_mode = core_frac_mode
    cfg.interior_struct.module = 'dummy'
    return cfg


def _call_and_swallow_downstream(config):
    """Call compute_initial_entropy and absorb downstream failures.

    The fallback log line we want to assert on fires before any EOS
    table is touched, so the only failure mode that matters here is
    the deleted Earth-window ``ValueError``. Anything else from deeper
    in the call chain (FileNotFoundError, RuntimeError on missing
    EOS, ImportError on missing Zalmoxis melting_curves, ...) is
    expected in this stripped test environment and gets swallowed.
    """
    from proteus.interior_energetics import common

    try:
        common.compute_initial_entropy(
            config,
            hf_row=None,
            spider_eos_dir=None,
        )
    except ValueError as e:
        msg = str(e)
        # Regression guard: the deleted Earth-window guard must NOT
        # come back. Any ValueError with the old wording fails the test.
        assert '135 GPa P_cmb fallback' not in msg, (
            f'Earth-window ValueError guard still active: {msg}'
        )
        assert 'cannot use the Earth-like' not in msg, (
            f'Earth-window ValueError guard still active: {msg}'
        )
    except Exception:
        # Downstream EOS / Zalmoxis-adiabat failures are expected here
        # and not what this test is about.
        pass


@pytest.mark.parametrize('mass_tot', [0.5, 1.0, 3.0, 5.0, 10.0])
def test_super_earth_no_longer_raises_earth_window_error(mass_tot, caplog):
    """The pre-fix implementation raised a ValueError that explicitly
    mentioned ``"cannot use the Earth-like 135 GPa P_cmb fallback"``
    for mass_tot outside [0.5, 2.0] M_Earth. Verify that error is
    gone across the full 0.5-10 M_Earth band.
    """
    cfg = _make_minimal_config(mass_tot=mass_tot)
    caplog.set_level(
        logging.WARNING,
        logger='fwl.proteus.interior_energetics.common',
    )
    _call_and_swallow_downstream(cfg)
    # The Noack & Lasbleis (2020) fallback log must appear regardless
    # of downstream success.
    text = '\n'.join(rec.message for rec in caplog.records)
    assert 'Noack & Lasbleis (2020) mass-aware fallback' in text, (
        f'Noack & Lasbleis (2020) log line missing for mass_tot={mass_tot} M_Earth: {text}'
    )
    assert f'mass_tot={mass_tot:.2f} M_Earth' in text, (
        f'Mass not echoed in Noack & Lasbleis (2020) log line: {text}'
    )


def test_NL20_log_lines_pcmb_scales_with_mass(caplog):
    """The NL20 log line must echo a P_cmb that scales with mass.
    Discriminating: a regression to a constant fallback would emit
    the same P_cmb GPa value regardless of mass.
    """
    caplog.set_level(
        logging.WARNING,
        logger='fwl.proteus.interior_energetics.common',
    )

    p_cmb_logged = {}
    for mass_tot in (1.0, 5.0):
        caplog.clear()
        cfg = _make_minimal_config(mass_tot=mass_tot)
        _call_and_swallow_downstream(cfg)
        text = '\n'.join(rec.message for rec in caplog.records)
        # Extract "P_cmb=NNN.N GPa" from the log
        import re

        m = re.search(r'P_cmb=(\d+\.\d+)\s+GPa', text)
        assert m is not None, f'P_cmb not present in log: {text}'
        p_cmb_logged[mass_tot] = float(m.group(1))

    assert p_cmb_logged[5.0] > p_cmb_logged[1.0] + 100.0, (
        f'NL20 P_cmb did not scale with mass: '
        f'1 M_E -> {p_cmb_logged[1.0]:.1f} GPa, '
        f'5 M_E -> {p_cmb_logged[5.0]:.1f} GPa. '
        f'Expected at least 100 GPa difference.'
    )
