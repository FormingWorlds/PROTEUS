"""Contract and boundary tests for Zalmoxis low-temperature extension coupling."""

from __future__ import annotations

import inspect
import logging
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


def test_the_installed_zalmoxis_takes_every_keyword_proteus_passes(tmp_path, monkeypatch):
    """Contract with the installed zalmoxis.eos_export.generate_spider_eos_tables: when it has
    solid_lowt_file, every keyword PROTEUS passes with the switch on is one of its parameters;
    when it lacks it, PROTEUS stops the build with the RuntimeError that names solid_lowt_file."""
    real = zalmoxis.eos_export.generate_spider_eos_tables
    params = inspect.signature(real).parameters
    if 'solid_lowt_file' in params:
        _, tables, _ = _lowt_stubbed_run(tmp_path, monkeypatch, lowt=True)
        passed = set(tables.call_args.kwargs)
        assert 'solid_lowt_file' in passed
        assert passed <= set(params), passed - set(params)
        assert len(tables.call_args.args) <= len(params)
    else:
        with pytest.raises(Exception) as exc:
            _lowt_stubbed_run(tmp_path, monkeypatch, lowt=True, tables=real)
        assert exc.type is RuntimeError, repr(exc.value)
        assert 'whose generate_spider_eos_tables takes solid_lowt_file' in str(exc.value)


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


def test_resumed_run_with_switch_changed_keeps_old_tables_and_resume_key_has_extension(
    tmp_path, monkeypatch, caplog
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

    # Current behaviour 1: resumed run keeps existing tables, does not call the generator, warns.
    tables.reset_mock()
    with caplog.at_level(logging.WARNING):
        resumed_out = generate_spider_tables(config, str(run_dir))
    assert any('ignores the changed settings' in r.getMessage() for r in caplog.records)
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
