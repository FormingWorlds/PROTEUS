"""Contract and boundary tests for Zalmoxis low-temperature extension coupling."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import zalmoxis.eos_export

from proteus.interior_struct.zalmoxis import (
    _ps_cache_key,
    _ps_resume_key,
    generate_spider_tables,
)
from tests.interior_struct.test_zalmoxis import _lowt_stubbed_run

pytestmark = pytest.mark.unit


# ── F9: Signature and keyword parameter contract ────────────────────────────


def test_zalmoxis_generate_spider_eos_tables_signature_contract(tmp_path, monkeypatch):
    """Verify keyword parameter contract between PROTEUS and generate_spider_eos_tables.

    Tests both branches: when solid_lowt_file is present in the installed Zalmoxis,
    all required keywords are verified. When absent, PROTEUS raises RuntimeError.
    """
    real_sig = inspect.signature(zalmoxis.eos_export.generate_spider_eos_tables)
    expected_keywords = {
        'eos_file',
        'solidus_func',
        'liquidus_func',
        'P_range',
        'n_P',
        'n_S',
        'output_dir',
        'solid_eos_file',
        'liquid_eos_file',
        'solid_lowt_file',
    }

    # Branch 1: Installed Zalmoxis has solid_lowt_file parameter.
    if 'solid_lowt_file' in real_sig.parameters:
        for kw in expected_keywords:
            assert kw in real_sig.parameters, (
                f"Keyword '{kw}' missing from generate_spider_eos_tables"
            )

    # Branch 2: When solid_lowt_file is missing from signature, PROTEUS raises RuntimeError.
    def _stub_without_lowt(
        eos_file,
        solidus_func,
        liquidus_func,
        P_range=(1e5, 200e9),
        n_P=150,
        n_S=150,
        output_dir=None,
        solid_eos_file=None,
        liquid_eos_file=None,
    ):
        pass

    try:
        _lowt_stubbed_run(tmp_path, monkeypatch, lowt=True, tables=_stub_without_lowt)
    except RuntimeError as exc:
        assert 'takes solid_lowt_file' in str(exc)
    except Exception as exc:
        pytest.fail(
            f"Expected RuntimeError('takes solid_lowt_file') but raised {type(exc).__name__}: {exc}"
        )
    else:
        pytest.fail("DID NOT RAISE RuntimeError('takes solid_lowt_file')")


# ── F10: Golden cache key digest string ─────────────────────────────────────


def test_ps_cache_key_golden_digest_without_extension():
    """Verify deterministic golden cache key and digest string when low-T extension is off."""
    common = dict(
        P_max=3.5e11,
        nP=8,
        nS=8,
        mzf=0.8,
        layout='unified',
        mantle_eos='PALEOS:MgSiO3',
        eos_file='/d/u.dat',
        solid_eos='/d/s.dat',
        liquid_eos='/d/l.dat',
    )
    expected_key = 'P_max=3.500000e+11_nP=8_nS=8_mzf=0.8_layout=unified_eos=PALEOS-MgSiO3-077d71466450_gen=g'
    computed_key = _ps_cache_key(**common, generator='g', lowt_file=None)

    assert computed_key == expected_key
    assert '077d71466450' in computed_key


# ── F11: Resumed run behaviour and resume key with extension ────────────────


def test_resumed_run_with_switch_changed_keeps_old_tables_and_resume_key_has_extension(
    tmp_path, monkeypatch
):
    """A resumed run with lowt_extension toggled keeps existing tables while resume key tracks extension."""
    # First, run with switch off to generate tables and cache marker.
    out, tables, zmod = _lowt_stubbed_run(
        tmp_path,
        monkeypatch,
        lowt=False,
    )
    assert out is not None
    run_dir = tmp_path / 'run'
    eos_dir = run_dir / 'data' / 'spider_eos'
    assert (eos_dir / 'solidus_P-S.dat').exists()
    old_marker = (eos_dir / '.cache_info.txt').read_text().strip()

    # Now configure a resumed run with the switch turned on.
    config = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.lookup_nP = 8
    config.interior_struct.zalmoxis.lookup_nS = 8
    config.interior_struct.zalmoxis.lowt_extension = True
    config.planet.mass_tot = 1.0
    config.params.resume = True

    # Current behaviour 1: resumed run keeps existing tables and does not call generator.
    tables.reset_mock()
    resumed_out = generate_spider_tables(config, str(run_dir))
    assert resumed_out is not None
    assert resumed_out['eos_dir'] == str(eos_dir)
    assert tables.call_count == 0

    # Current behaviour 2: resume key calculation includes the extension and differs from old marker.
    entry = {'format': 'paleos_unified', 'eos_file': str(tmp_path / 'solid.dat')}
    mat_dicts = {'PALEOS:MgSiO3': entry}
    resume_k = _ps_resume_key(config, 'PALEOS:MgSiO3', entry, mat_dicts)
    assert resume_k != old_marker
    assert resume_k.startswith('P_max=3.500000e+11_nP=8_nS=8_mzf=0.8_layout=unified_eos=')
    # Because lowt_extension=True, the cache key digest reflects the extension file.
    assert old_marker.split('_eos=')[1] != resume_k.split('_eos=')[1]
