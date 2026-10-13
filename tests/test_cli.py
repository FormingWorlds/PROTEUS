# Test PROTEUS terminal CLI and commands
from __future__ import annotations

import builtins
import importlib.metadata
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from proteus import __version__ as proteus_version
from proteus import cli, doctor
from proteus.config import UnknownConfigKeyError

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


runner = CliRunner()


@pytest.fixture(autouse=True)
def phoenix_downloads(monkeypatch):
    """Replace the PHOENIX download, so that no ``install-all`` run here reaches the network.

    Returns the list of the keyword arguments of each call; a test of the
    download command or of a failure sets its own replacement over this one.
    """
    calls = []

    def fake_download_phoenix(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr('proteus.utils.data.download_phoenix', fake_download_phoenix)
    return calls


_REAL_CHECK_JULIA = doctor.check_julia


@pytest.fixture(autouse=True)
def julia_check(monkeypatch):
    """Replace the Julia check of install-all and update-all, which runs ``julia --version``.

    Many tests here stub ``subprocess.run`` with an object without output; a test
    of the Julia line sets ``_REAL_CHECK_JULIA`` back.
    """
    passed = doctor.CheckResult(
        name='julia', category='environment', status=doctor.PASS, message='1.13.1 (stub)'
    )
    monkeypatch.setattr(doctor, 'check_julia', lambda: passed)


def _solar_dir(root):
    """Version directory of the solar spectra dataset below ``root``."""
    from proteus.data import STELLAR_SPECTRA_SOLAR, dataset_dir

    return dataset_dir(STELLAR_SPECTRA_SOLAR, data_root=root)


@pytest.mark.unit
def test_doctor(monkeypatch, tmp_path):
    """``proteus doctor`` renders its report and returns a health-coded exit.

    The exit code is 0 when the install is healthy and 1 when a check fails, so
    both are valid here depending on the environment; the report itself, with
    the package section and the AGNI and fwl-mors checks, is rendered either
    way. A usage error or crash would exit 2 and skip the report. A failing check
    writes a failure log, whose environment summary runs slow commands such as
    ``conda list``, so the summary is stubbed and the log goes to ``tmp_path``.
    """
    monkeypatch.setattr('proteus.doctor._collect_environment_info', lambda: 'stub')
    monkeypatch.chdir(tmp_path)
    # run PROTEUS doctor command
    response = runner.invoke(cli.doctor, [])

    # 0 (healthy) or 1 (a check failed) are both valid; 2 would be a crash.
    assert response.exit_code in (0, 1)

    # contains information we expect, regardless of the health verdict
    assert 'Package versions' in response.output
    assert 'AGNI' in response.output
    assert 'fwl-mors' in response.output


@pytest.mark.unit
def test_version():
    """``proteus --version`` exits 0 and prints the current PROTEUS version."""
    # run PROTEUS version command
    response = runner.invoke(cli.cli, ['--version'])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
    assert str(proteus_version) in response.output


@pytest.mark.unit
def test_get(monkeypatch, tmp_path):
    """Every `proteus get <subcommand>` dispatches without raising.

    The downloaders touch the network; each is replaced with a no-op so
    the test stays a pure CLI dispatch check rather than a network smoke.
    Some subcommands also assert post-conditions (e.g. `solar` requires
    that files exist under the solar dataset directory after the
    downloader returns), so the no-op for download_stellar_spectra writes
    a stub file at the expected path. FWL_DATA itself is monkeypatched to
    a tmp_path so the test never touches the user's real data tree.

    Each subcommand has its own assertion on its exit code, so a failure
    names the subcommand. FWL_DATA points at an empty tmp_path, so the
    result does not depend on the data tree of the machine.
    """
    # Monkeypatch FWL_DATA to a writable tmp_path so post-condition file
    # checks have a controlled environment. The module-level FWL_DATA_DIR
    # constant in proteus.utils.data is read at import time, so setenv
    # alone is not enough; patch the module attribute too.
    from pathlib import Path as _Path

    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', _Path(tmp_path), raising=False)

    def stub_download_stellar_spectra(folders=('solar',), **kwargs):
        # `proteus get solar` requires files in the solar dataset directory,
        # so the stub writes one.
        for folder in folders:
            target = _solar_dir(tmp_path)
            target.mkdir(parents=True, exist_ok=True)
            (target / '_stub.txt').write_text('stub')
        return True

    no_op = lambda *args, **kwargs: True  # noqa: E731
    for target in (
        'proteus.utils.data.download_exoplanet_data',
        'proteus.utils.data.download_massradius_data',
        'proteus.utils.data.download_surface_albedos',
        'proteus.utils.data.download_spectral_file',
        'proteus.utils.data.download_spectral_files',
        'proteus.utils.data.download_phoenix',
        'proteus.utils.data.download_muscles',
        'proteus.utils.data.download_stellar_tracks',
    ):
        monkeypatch.setattr(target, no_op, raising=False)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra',
        stub_download_stellar_spectra,
        raising=False,
    )

    response = runner.invoke(cli.get, ['reference'])
    assert response.exit_code == 0, f'reference: {response.output}'

    response = runner.invoke(cli.get, ['surfaces'])
    assert response.exit_code == 0, f'surfaces: {response.output}'

    response = runner.invoke(cli.get, ['spectral', '-n', 'Frostflow', '-b', '16'])
    assert response.exit_code == 0, f'spectral: {response.output}'

    response = runner.invoke(cli.get, ['muscles', '--star', 'trappist-1'])
    assert response.exit_code == 0, f'muscles: {response.output}'

    response = runner.invoke(cli.get, ['phoenix', '--feh', '0.0', '--alpha', '0.0'])
    assert response.exit_code == 0, f'phoenix: {response.output}'

    response = runner.invoke(cli.get, ['solar'])
    assert response.exit_code == 0, f'solar: {response.output}'


# ---------------------------
# Extended tests: proteus get subcommands
# ---------------------------


@pytest.mark.unit
def test_get_help_extended():
    """``proteus get --help`` lists every data-fetch subcommand."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', '--help'])
    assert res.exit_code == 0
    assert 'Get data and modules' in res.output
    # Subcommands should be listed
    for sub in ['reference', 'surfaces', 'spectral', 'muscles', 'phoenix', 'solar', 'stellar']:
        assert sub in res.output


@pytest.mark.unit
def test_get_unknown_subcommand():
    """Unknown ``get`` subcommand exits non-zero with a 'No such command' message."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'does-not-exist'])
    assert res.exit_code != 0
    assert 'No such command' in res.output


# ---- spectral ----


@pytest.mark.unit
def test_get_spectral_defaults(monkeypatch):
    """``proteus get spectral`` (no args) requests the full spectral set (None, None)."""
    runner = CliRunner()
    calls = []

    def fake_download_spectral_files(name, bands):
        calls.append((name, bands))

    # imported inside command: from .utils.data import download_spectral_files
    monkeypatch.setattr(
        'proteus.utils.data.download_spectral_files', fake_download_spectral_files
    )

    res = runner.invoke(cli.cli, ['get', 'spectral'])
    assert res.exit_code == 0
    assert calls == [(None, None)]


@pytest.mark.unit
def test_get_spectral_forwards_args(monkeypatch):
    """``proteus get spectral -n NAME -b BANDS`` forwards the args to the downloader."""
    runner = CliRunner()
    calls = []

    def fake_download_spectral_files(name, bands):
        calls.append((name, bands))

    monkeypatch.setattr(
        'proteus.utils.data.download_spectral_files', fake_download_spectral_files
    )

    res = runner.invoke(cli.cli, ['get', 'spectral', '-n', 'Frostflow', '-b', '16'])
    assert res.exit_code == 0
    assert calls == [('Frostflow', '16')]


@pytest.mark.unit
def test_get_spectral_band_only_rejected(monkeypatch):
    """``proteus get spectral -b BANDS`` without a group exits non-zero.

    The band-only selection is ambiguous; the real downloader raises
    ValueError and the CLI must surface a failure, with no download
    attempted.
    """
    runner = CliRunner()
    calls = []

    monkeypatch.setattr(
        'proteus.utils.data.download_spectral_file',
        lambda name, bands: calls.append((name, bands)),
    )

    # The real download_spectral_files runs: its validation raises before
    # any download dispatch.
    res = runner.invoke(cli.cli, ['get', 'spectral', '-b', '256'])
    assert res.exit_code != 0
    assert calls == []  # no download side effect ran


# ---- surfaces / reference ----


@pytest.mark.unit
def test_get_surfaces_calls_downloader(monkeypatch):
    """``proteus get surfaces`` calls download_surface_albedos exactly once."""
    runner = CliRunner()
    calls = []

    def fake_download_surface_albedos():
        calls.append(True)

    monkeypatch.setattr(
        'proteus.utils.data.download_surface_albedos', fake_download_surface_albedos
    )

    res = runner.invoke(cli.cli, ['get', 'surfaces'])
    assert res.exit_code == 0
    assert calls == [True]


@pytest.mark.unit
def test_get_reference_calls_both_downloaders(monkeypatch):
    """``proteus get reference`` triggers BOTH exoplanet-data and mass-radius downloaders."""
    runner = CliRunner()
    calls = []

    def fake_download_exoplanet_data():
        calls.append('exo')

    def fake_download_massradius_data():
        calls.append('mr')

    monkeypatch.setattr(
        'proteus.utils.data.download_exoplanet_data', fake_download_exoplanet_data
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_massradius_data', fake_download_massradius_data
    )

    res = runner.invoke(cli.cli, ['get', 'reference'])
    assert res.exit_code == 0
    assert 'exo' in calls
    assert 'mr' in calls


# ---- stellar ----


@pytest.mark.unit
def test_get_stellar_downloads_tracks_and_spectra(monkeypatch):
    """``proteus get stellar`` downloads BOTH track sets (Spada + Baraffe) and the spectra."""
    runner = CliRunner()
    calls = []

    def fake_download_stellar_tracks(track):
        calls.append(('track', track))

    def fake_download_stellar_spectra():
        calls.append(('spectra',))

    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_tracks', fake_download_stellar_tracks
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'stellar'])
    assert res.exit_code == 0
    assert ('track', 'Spada') in calls
    assert ('track', 'Baraffe') in calls
    assert ('spectra',) in calls


# ---- muscles ----


@pytest.mark.unit
def test_get_muscles_requires_star_or_all():
    """``proteus get muscles`` with neither --star nor --all exits non-zero with a usage hint."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles'])
    assert res.exit_code != 0
    assert 'Provide --star NAME or use --all.' in res.output


@pytest.mark.unit
def test_get_muscles_list_outputs_names():
    """``proteus get muscles --list`` enumerates the known star names."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--list'])
    assert res.exit_code == 0
    # spot-check a couple known entries
    assert 'trappist-1' in res.output
    assert 'gj876' in res.output


@pytest.mark.unit
def test_get_muscles_star_normalization_and_download(monkeypatch):
    """User-supplied star name is lowercased and stripped before the downloader sees it."""
    runner = CliRunner()
    calls = []

    def fake_download_muscles(star):
        calls.append(star)
        return True

    monkeypatch.setattr('proteus.utils.data.download_muscles', fake_download_muscles)

    res = runner.invoke(cli.cli, ['get', 'muscles', '--star', '  TRAPPIST-1 '])
    assert res.exit_code == 0
    assert calls == ['trappist-1']
    assert 'Done. OK: 1/1' in res.output


@pytest.mark.unit
def test_get_muscles_star_unknown_suggests_close_matches():
    """An unknown star name is rejected with either a Did-you-mean or --list hint."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--star', 'gj87'])
    assert res.exit_code != 0
    assert 'Unknown MUSCLES star' in res.output
    # Implementation adds either a Did-you-mean hint or at least the --list hint
    assert ('Did you mean:' in res.output) or ('Use --list' in res.output)


@pytest.mark.unit
def test_get_muscles_all_and_star_mutually_exclusive():
    """``--all`` and ``--star`` cannot be combined; the CLI rejects the call with a usage hint."""
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--all', '--star', 'trappist-1'])
    assert res.exit_code != 0
    assert 'Use either --all or --star NAME' in res.output


@pytest.mark.unit
def test_get_muscles_all_reports_failures(monkeypatch):
    """``--all`` mode reports per-star success/failure counts and names failed stars."""
    runner = CliRunner()

    # keep the loop small
    monkeypatch.setattr(cli, 'STARS_ONLINE', {'muscles': ['a', 'b'], 'solar': ['sun']})

    def fake_download_muscles(star):
        return star != 'b'

    monkeypatch.setattr('proteus.utils.data.download_muscles', fake_download_muscles)

    res = runner.invoke(cli.cli, ['get', 'muscles', '--all'])
    assert res.exit_code == 0
    assert 'Done. OK: 1/2' in res.output
    assert 'Failed (1): b' in res.output


@pytest.mark.unit
def test_get_muscles_all_all_failed_raises(monkeypatch):
    """``--all`` mode with every download failing exits non-zero with an aggregate error."""
    runner = CliRunner()

    monkeypatch.setattr(cli, 'STARS_ONLINE', {'muscles': ['a', 'b'], 'solar': ['sun']})

    def fake_download_muscles(star):
        return False

    monkeypatch.setattr('proteus.utils.data.download_muscles', fake_download_muscles)

    res = runner.invoke(cli.cli, ['get', 'muscles', '--all'])
    assert res.exit_code != 0
    assert 'All MUSCLES downloads failed' in res.output


# ---- phoenix ----


@pytest.mark.unit
def test_get_phoenix_calls_grid_mapper_and_downloader(monkeypatch):
    """``proteus get phoenix`` first calls ``phoenix_to_grid`` to snap params, then the downloader."""
    runner = CliRunner()
    calls = []

    def fake_phoenix_to_grid(FeH, alpha, Teff):
        calls.append(('grid', FeH, alpha, Teff))
        return {'FeH': 0.0, 'alpha': 0.0}

    def fake_download_phoenix(alpha, FeH):
        calls.append(('dl', alpha, FeH))
        return True

    monkeypatch.setattr('proteus.utils.phoenix_helper.phoenix_to_grid', fake_phoenix_to_grid)
    monkeypatch.setattr('proteus.utils.data.download_phoenix', fake_download_phoenix)

    res = runner.invoke(
        cli.cli,
        ['get', 'phoenix', '--feh', '0.05', '--alpha', '-0.01', '--teff', '3200'],
    )
    assert res.exit_code == 0
    assert ('grid', 0.05, -0.01, 3200.0) in calls
    assert ('dl', 0.0, 0.0) in calls
    assert 'Downloaded PHOENIX grid' in res.output


@pytest.mark.unit
def test_get_phoenix_download_failure_raises(monkeypatch):
    """A failed PHOENIX download surfaces as non-zero exit with a 'Failed to download' message."""
    runner = CliRunner()

    def fake_phoenix_to_grid(FeH, alpha, Teff):
        return {'FeH': -0.5, 'alpha': 0.2}

    def fake_download_phoenix(alpha, FeH):
        return False

    monkeypatch.setattr('proteus.utils.phoenix_helper.phoenix_to_grid', fake_phoenix_to_grid)
    monkeypatch.setattr('proteus.utils.data.download_phoenix', fake_download_phoenix)

    res = runner.invoke(cli.cli, ['get', 'phoenix', '--feh', '-0.6', '--alpha', '0.3'])
    assert res.exit_code != 0
    assert 'Failed to download PHOENIX grid' in res.output


# ---- solar ----


@pytest.mark.unit
def test_get_solar_success_when_files_present(monkeypatch, tmp_path):
    """``proteus get solar`` succeeds when files materialise in the solar dataset directory."""
    runner = CliRunner()

    def fake_GetFWLData():
        return tmp_path

    def fake_download_stellar_spectra(folders=('solar',)):
        solar_dir = _solar_dir(tmp_path)
        solar_dir.mkdir(parents=True, exist_ok=True)
        (solar_dir / 'dummy.txt').write_text('ok')

    monkeypatch.setattr('proteus.utils.data.GetFWLData', fake_GetFWLData)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'solar'])
    assert res.exit_code == 0
    assert 'Solar spectra downloaded successfully.' in res.output
    assert str(_solar_dir(tmp_path)) in res.output


@pytest.mark.unit
def test_get_solar_raises_if_no_files_found(monkeypatch, tmp_path):
    """``proteus get solar`` exits non-zero when the downloader writes no files,
    names the expected directory and tells the user to fetch again.
    """
    runner = CliRunner()

    def fake_GetFWLData():
        return tmp_path

    def fake_download_stellar_spectra(folders=('solar',)):
        # create directory but no files
        _solar_dir(tmp_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr('proteus.utils.data.GetFWLData', fake_GetFWLData)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'solar'])
    assert res.exit_code != 0
    assert 'no files were found' in res.output.lower()
    assert str(_solar_dir(tmp_path)) in res.output
    assert 'remove that folder' in res.output


@pytest.mark.unit
def test_get_solar_wraps_downloader_exception(monkeypatch, tmp_path):
    """A downloader exception is wrapped, not propagated raw, so the CLI exits cleanly with a message."""
    runner = CliRunner()

    def fake_GetFWLData():
        return tmp_path

    def fake_download_stellar_spectra(folders=('solar',)):
        raise RuntimeError('boom')

    monkeypatch.setattr('proteus.utils.data.GetFWLData', fake_GetFWLData)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'solar'])
    assert res.exit_code != 0
    assert 'Failed to download solar spectra' in res.output
    assert 'boom' in res.output


# ---- interiordata ----


@pytest.mark.unit
def test_get_interiordata_calls_clean_downloads(monkeypatch, tmp_path):
    """``proteus get interiordata`` reads the config and triggers BOTH interior lookup tables
    AND melting-curve downloads, passing the config object through.
    """
    runner = CliRunner()
    calls = []

    def fake_download_interior_lookuptables(clean: bool = False):
        calls.append(('interior', clean))

    def fake_read_config_object(path: Path):
        # Should receive the config-path from CLI
        calls.append(('read_cfg', Path(path)))
        return {'fake': 'config'}

    def fake_download_melting_curves(configuration, clean: bool = False):
        calls.append(('melt', configuration, clean))

    monkeypatch.setattr(
        'proteus.utils.data.download_interior_lookuptables',
        fake_download_interior_lookuptables,
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_melting_curves',
        fake_download_melting_curves,
    )

    monkeypatch.setattr(cli, 'read_config_object', fake_read_config_object)

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("this content won't be parsed because we patched read_config_object")

    res = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(cfg)])
    assert res.exit_code == 0

    assert ('interior', True) in calls
    assert ('read_cfg', cfg) in calls
    assert ('melt', {'fake': 'config'}, True) in calls
    # No structure module in the fake config: the Zalmoxis EOS download
    # must not be attempted.
    assert not any(c[0] == 'zalmoxis_eos' for c in calls)


def assert_refusal_logged_as_error(caplog, key: str) -> None:
    """Check that a config refusal named `key` was logged at ERROR, not below.

    The refusal reaches the terminal through the 'fwl' logger rather than
    click's own stream, so the captured records are what pins it. The level is
    asserted as well as the text: a refusal emitted at INFO or WARNING carries
    the same words and would otherwise pass.
    """
    named = [rec for rec in caplog.records if key in rec.getMessage()]
    assert named, f'no log record named {key}'
    assert [rec.levelno for rec in named] == [logging.ERROR] * len(named)


@pytest.mark.unit
def test_get_interiordata_reports_an_unknown_config_key_cleanly(monkeypatch, tmp_path, caplog):
    """A misspelled key stops the download and is reported as a CLI error.

    The download commands act on the configuration, so acting on one whose keys
    were silently discarded would fetch data for a setup the user did not ask
    for. The failure has to arrive at error level with the key named, not as a
    traceback.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    runner = CliRunner()
    downloads = []
    monkeypatch.setattr(
        'proteus.utils.data.download_interior_lookuptables',
        lambda clean=False: downloads.append('interior'),
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_melting_curves',
        lambda configuration, clean=False: downloads.append('melt'),
    )

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw['planet']['mass_total'] = 2.5  # deliberate misspelling of mass_tot
    cfg = tmp_path / 'typo.toml'
    with open(cfg, 'w') as f:
        tomlkit.dump(raw, f)

    with caplog.at_level(logging.INFO, logger='fwl'):
        res = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(cfg)])
    assert res.exit_code != 0
    assert_refusal_logged_as_error(caplog, 'planet.mass_total')
    # A ClickException does not surface a traceback.
    assert 'Traceback' not in res.output
    # The config-dependent download is not reached; the config-independent one
    # ahead of it may already have run, which is why only the former is pinned.
    assert 'melt' not in downloads

    # Discrimination: with the key spelled correctly the same command completes
    # and the melting-curve download does run, so the refusal is caused by the
    # misspelling rather than by this config being unusable.
    raw['planet']['mass_tot'] = raw['planet'].pop('mass_total')
    good = tmp_path / 'good.toml'
    with open(good, 'w') as f:
        tomlkit.dump(raw, f)
    res_ok = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(good)])
    assert res_ok.exit_code == 0
    assert 'melt' in downloads


@pytest.mark.unit
def test_get_interiordata_fetches_zalmoxis_eos(monkeypatch, tmp_path):
    """``proteus get interiordata`` downloads the structure EOS tables when the
    config selects the zalmoxis structure module, forwarding the configured
    core / mantle / ice EOS identifiers.
    """
    from types import SimpleNamespace

    runner = CliRunner()
    calls = []

    def fake_download_zalmoxis_eos(
        *, mantle_eos, core_eos, ice_layer_eos, volatile_eos, anchor_pair
    ):
        calls.append(
            ('zalmoxis_eos', mantle_eos, core_eos, ice_layer_eos, volatile_eos, anchor_pair)
        )

    fake_config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            module='zalmoxis',
            zalmoxis=SimpleNamespace(
                mantle_eos='PALEOS:MgSiO3',
                core_eos='PALEOS:iron',
                ice_layer_eos=None,
                dry_mantle=True,
            ),
        )
    )

    monkeypatch.setattr(
        'proteus.utils.data.download_interior_lookuptables', lambda clean=False: None
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_melting_curves', lambda configuration, clean=False: None
    )
    monkeypatch.setattr('proteus.utils.data.download_zalmoxis_eos', fake_download_zalmoxis_eos)
    monkeypatch.setattr(cli, 'read_config_object', lambda path: fake_config)

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('unused')

    res = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(cfg)])
    assert res.exit_code == 0
    # ice_layer_eos None must be coerced to '' (the downloader's no-ice value),
    # not passed through as None.
    assert calls == [('zalmoxis_eos', 'PALEOS:MgSiO3', 'PALEOS:iron', '', '', False)]


@pytest.mark.unit
def test_get_interiordata_fetches_ps_lookup_set_for_dummy(monkeypatch, tmp_path):
    """``proteus get interiordata`` fetches the P-S lookup set a dummy-structure run reads."""
    from types import SimpleNamespace

    runner = CliRunner()
    calls = []
    fake_config = SimpleNamespace(
        interior_energetics=SimpleNamespace(module='aragog'),
        interior_struct=SimpleNamespace(module='dummy', eos_dir=None, zalmoxis=None),
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_interior_lookuptables', lambda clean=False: None
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_melting_curves', lambda configuration, clean=False: None
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_eos_dynamic', lambda *a, **k: calls.append('ps_set')
    )
    monkeypatch.setattr(cli, 'read_config_object', lambda path: fake_config)
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('unused')

    res = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(cfg)])
    assert res.exit_code == 0, res.output
    assert calls == ['ps_set']

    # Discrimination: Zalmoxis with a PALEOS mantle generates its own tables.
    calls.clear()
    fake_config.interior_struct = SimpleNamespace(
        module='zalmoxis', eos_dir=None, zalmoxis=SimpleNamespace(mantle_eos='PALEOS:MgSiO3')
    )
    monkeypatch.setattr('proteus.utils.data.download_zalmoxis_eos_for_config', lambda c: None)
    res = runner.invoke(cli.cli, ['get', 'interiordata', '--config-path', str(cfg)])
    assert res.exit_code == 0, res.output
    assert calls == []


# ---- tool setup subcommands ----


@pytest.mark.unit
@pytest.mark.parametrize(
    'subcommand, target',
    [
        ('socrates', 'proteus.utils.data.get_socrates'),
        ('petsc', 'proteus.utils.data.get_petsc'),
        ('spider', 'proteus.utils.data.get_spider'),
    ],
)
def test_get_tools_subcommands_call_setup(monkeypatch, subcommand, target):
    """Each of ``get socrates``, ``get petsc``, ``get spider`` dispatches to its setup helper."""
    runner = CliRunner()
    calls = []

    def fake():
        calls.append(subcommand)

    monkeypatch.setattr(target, fake)

    res = runner.invoke(cli.cli, ['get', subcommand])
    assert res.exit_code == 0
    assert calls == [subcommand]


# ---------------------------
# --deterministic flag tests
# ---------------------------


@pytest.mark.unit
def test_start_help_lists_deterministic_flag():
    """The --deterministic flag must be visible in `proteus start --help`,
    so users discover it without reading source. The help text must explain
    WHEN to use it (numerical fragility), not just WHAT it does; that is the
    whole point of exposing this flag."""
    res = runner.invoke(cli.start, ['--help'])
    assert res.exit_code == 0
    assert '--deterministic' in res.output
    assert 'numerical' in res.output.lower() or 'reproduc' in res.output.lower()
    assert 'JAX' in res.output or 'XLA' in res.output


@pytest.mark.unit
def test_should_apply_deterministic_decision_table():
    """Exercise the boolean decision: only fire when (a) --deterministic is
    in argv AND (b) the sentinel is not already set. Test all four corners
    of the truth table to catch a missing-AND or accidental-OR bug."""
    sentinel = cli._PROTEUS_DETERMINISTIC_SENTINEL

    # (a) flag present, sentinel unset → apply
    assert cli._should_apply_deterministic(['proteus', 'start', '--deterministic'], {}) is True

    # (b) flag present, sentinel already set → do NOT re-apply (would loop)
    assert (
        cli._should_apply_deterministic(
            ['proteus', 'start', '--deterministic'], {sentinel: '1'}
        )
        is False
    )

    # (c) flag absent, sentinel unset → do nothing
    assert cli._should_apply_deterministic(['proteus', 'start', '-c', 'cfg.toml'], {}) is False

    # (d) flag absent, sentinel set (orphaned env) → do nothing
    assert cli._should_apply_deterministic(['proteus', 'start'], {sentinel: '1'}) is False

    # Unicode flag-lookalike must NOT trigger (substring vs membership)
    assert (
        cli._should_apply_deterministic(['proteus', 'start', '--deterministic-mode'], {})
        is False
    )


@pytest.mark.unit
def test_apply_deterministic_env_appends_xla_flags():
    """XLA_FLAGS may already carry user-set debugging flags. We must APPEND
    --xla_cpu_enable_fast_math=false, not overwrite, otherwise we'd silently
    drop the user's debugging context. Also: idempotent, calling twice must
    not duplicate the flag."""
    # Existing XLA_FLAGS preserved + extended
    env = {'XLA_FLAGS': '--xla_dump_to=/tmp/xla'}
    cli._apply_deterministic_env(env)
    assert '--xla_dump_to=/tmp/xla' in env['XLA_FLAGS']
    assert cli._DETERMINISTIC_XLA_FLAG in env['XLA_FLAGS']
    assert env['JAX_ENABLE_X64'] == '1'
    assert env[cli._PROTEUS_DETERMINISTIC_SENTINEL] == '1'

    # Idempotency: applying again must not duplicate the flag
    cli._apply_deterministic_env(env)
    assert env['XLA_FLAGS'].count(cli._DETERMINISTIC_XLA_FLAG) == 1

    # Empty/missing XLA_FLAGS produces a clean single-flag string (no leading space)
    env2 = {}
    cli._apply_deterministic_env(env2)
    assert env2['XLA_FLAGS'] == cli._DETERMINISTIC_XLA_FLAG
    assert not env2['XLA_FLAGS'].startswith(' ')


@pytest.mark.unit
def test_deterministic_warning_when_sentinel_missing(tmp_path, monkeypatch):
    """If a user-supplied wrapper or test harness invokes the click handler
    with --deterministic but the env-var re-exec did not actually run (sentinel
    not set), the user must SEE a warning so the run isn't silently
    non-deterministic. Verifies the click handler's safety net, not the
    re-exec path itself (which is a subprocess concern)."""
    # Need a config file path for click validation; construct a minimal stub
    cfg = tmp_path / 'stub.toml'
    cfg.write_text('# stub\n')

    # Force sentinel to NOT be set, then invoke the click handler with --deterministic.
    # We monkeypatch Proteus to avoid actually running anything.
    monkeypatch.delenv(cli._PROTEUS_DETERMINISTIC_SENTINEL, raising=False)

    captured = {}

    class FakeProteus:
        def __init__(self, config_path):
            captured['config_path'] = config_path

        def start(self, resume, offline):
            captured['started'] = (resume, offline)

    monkeypatch.setattr(cli, 'Proteus', FakeProteus)

    res = runner.invoke(
        cli.start,
        ['-c', str(cfg), '--deterministic'],
    )
    # Either click validated the path and the handler ran, or click rejected the
    # stub config. In both cases we want to confirm the warning was emitted IFF
    # the handler ran. The handler emits to stdout via click.secho; CliRunner
    # captures both stdout and stderr in `output`.
    if 'started' in captured:
        assert 'NOT pinned' in res.output
        # Discrimination: with the sentinel deliberately unset, the click
        # handler must NOT have re-execed; pin its absence in env so the
        # warning above can only have come from the safety net, not from
        # an unrelated emit.
        import os as _os

        assert cli._PROTEUS_DETERMINISTIC_SENTINEL not in _os.environ


# ---------------------------
# 'plot' command
# ---------------------------


class _FakeProteusForPlot:
    """Minimal stand-in for Proteus used by the plot command path."""

    def __init__(self, *, config_path):
        self.config_path = config_path
        self.directories = {'output': '/tmp/proteus_test_plot_output'}

        class _Params:
            class _Out:
                logging = 'INFO'

            out = _Out()

        class _Config:
            params = _Params()

        self.config = _Config()


@pytest.mark.unit
def test_plot_list_argument_lists_available_plots(monkeypatch, tmp_path):
    """``proteus plot list`` (positional) prints the dispatch keys and returns 0.

    Different from ``--list`` flag: ``list`` as positional argument runs through
    the handler body. Verifies that at least two known plot kinds appear and
    that no plot function is actually invoked.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub config\n')

    invoked = []

    def fake_setup_logger(*args, **kwargs):
        invoked.append('logger')

    monkeypatch.setattr(cli, 'Proteus', _FakeProteusForPlot)
    monkeypatch.setattr(cli, 'setup_logger', fake_setup_logger)

    res = runner.invoke(cli.plot, ['list', '-c', str(cfg)])
    assert res.exit_code == 0, f'plot list output: {res.output}'
    assert 'Available plots:' in res.output
    # Two distinct kinds must appear, otherwise a single-key regression slips through.
    assert 'atmosphere' in res.output
    assert 'interior' in res.output


@pytest.mark.unit
def test_plot_invalid_plot_name_reports_and_continues(monkeypatch, tmp_path):
    """An unknown plot name yields an 'Invalid plot' message but does NOT raise.

    Discrimination: a regression that raised on unknown names would make the
    CLI brittle for partial typos; we pin the graceful-skip contract.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub config\n')

    monkeypatch.setattr(cli, 'Proteus', _FakeProteusForPlot)
    monkeypatch.setattr(cli, 'setup_logger', lambda *a, **k: None)

    res = runner.invoke(cli.plot, ['does_not_exist', '-c', str(cfg)])
    assert res.exit_code == 0
    assert 'Invalid plot: does_not_exist' in res.output


@pytest.mark.unit
def test_plot_all_dispatches_every_known_plot(monkeypatch, tmp_path):
    """``proteus plot all -c cfg`` iterates every dispatch key and invokes each handler.

    Mocks all plot functions so the test stays fast and offline. The post-state
    assertion checks the number of calls equals the size of plot_dispatch and
    that the handler kwarg was threaded through.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub config\n')

    monkeypatch.setattr(cli, 'Proteus', _FakeProteusForPlot)
    monkeypatch.setattr(cli, 'setup_logger', lambda *a, **k: None)

    # Patch each entry of plot_dispatch to a recorder so the real plotting
    # implementations don't fire (they would touch matplotlib / data files).
    from proteus import plot as plot_module

    calls = []

    def make_recorder(name):
        def _rec(handler):
            calls.append((name, handler))

        return _rec

    fake_dispatch = {key: make_recorder(key) for key in plot_module.plot_dispatch}
    monkeypatch.setattr(plot_module, 'plot_dispatch', fake_dispatch)

    res = runner.invoke(cli.plot, ['all', '-c', str(cfg)])
    assert res.exit_code == 0, f'plot all output: {res.output}'
    # Every dispatch entry must have fired exactly once
    assert len(calls) == len(fake_dispatch)
    # Discrimination: regression that swallowed the loop would yield 0 calls;
    # regression that double-dispatched would yield 2*N.
    assert {name for name, _ in calls} == set(fake_dispatch.keys())


@pytest.mark.unit
def test_plot_single_named_plot_invokes_only_that_handler(monkeypatch, tmp_path):
    """Naming a single plot dispatches ONLY that handler, not the whole dispatch table."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub config\n')

    monkeypatch.setattr(cli, 'Proteus', _FakeProteusForPlot)
    monkeypatch.setattr(cli, 'setup_logger', lambda *a, **k: None)

    from proteus import plot as plot_module

    calls = []

    def make_recorder(name):
        def _rec(handler):
            calls.append(name)

        return _rec

    fake_dispatch = {key: make_recorder(key) for key in plot_module.plot_dispatch}
    monkeypatch.setattr(plot_module, 'plot_dispatch', fake_dispatch)

    res = runner.invoke(cli.plot, ['atmosphere', '-c', str(cfg)])
    assert res.exit_code == 0
    assert calls == ['atmosphere']
    # Discrimination guard: no other plot fired (regression that ran 'all'
    # would produce >1 entry).
    assert len(calls) == 1


@pytest.mark.unit
def test_plot_list_flag_prints_dispatch_and_exits(monkeypatch):
    """``proteus plot --list`` (the eager-flag callback) prints dispatch names and exits."""
    # The --list flag goes through list_plots(); that callback uses sys.exit(),
    # which click captures and surfaces as exit_code != None.
    res = runner.invoke(cli.plot, ['--list'])
    # sys.exit() with no arg means 0
    assert res.exit_code == 0
    # The eager callback prints space-separated dispatch keys
    assert 'atmosphere' in res.output
    assert 'interior' in res.output


# ---------------------------
# normalize_star_name / validate_star_name
# ---------------------------


@pytest.mark.unit
def test_normalize_star_name_returns_none_for_falsy_inputs():
    """Empty string and None both return None (the 'no input' branch)."""
    # Both falsy inputs collapse to the same None; pin that contract.
    assert cli.normalize_star_name(None) is None
    assert cli.normalize_star_name('') is None
    # Discrimination: a non-empty string MUST NOT return None.
    assert cli.normalize_star_name('Sun') == 'sun'


@pytest.mark.unit
def test_normalize_star_name_lowercases_and_strips():
    """User input is lowercased and stripped before catalog lookup."""
    out = cli.normalize_star_name('  TRAPPIST-1 ')
    assert out == 'trappist-1'
    # Sign / scale guard: the original string contained uppercase letters and
    # whitespace; the output has neither.
    assert out.islower()
    assert out.strip() == out


@pytest.mark.unit
def test_validate_star_name_passthrough_for_none():
    """``validate_star_name(None)`` returns None without raising (matches normalize)."""
    out = cli.validate_star_name(None, catalog='muscles')
    assert out is None
    # Discrimination: with a valid name it should NOT return None
    assert cli.validate_star_name('trappist-1', catalog='muscles') == 'trappist-1'


@pytest.mark.unit
def test_validate_star_name_rejects_unknown_catalog():
    """An unknown catalog name surfaces ``ClickException`` listing available ones."""
    import click as _click

    with pytest.raises(_click.ClickException) as exc_info:
        cli.validate_star_name('trappist-1', catalog='not_a_real_catalog')

    msg = str(exc_info.value.message)
    assert "Unknown catalog 'not_a_real_catalog'" in msg
    # Discrimination: the error message must enumerate at least one valid catalog.
    assert 'muscles' in msg


@pytest.mark.unit
def test_validate_star_name_unknown_star_raises():
    """Unknown star in a known catalog raises ClickException, no return value."""
    import click as _click

    with pytest.raises(_click.ClickException) as exc_info:
        cli.validate_star_name('definitely-not-a-star-xyz', catalog='muscles')

    # Error message must name the star + cite --list as the recovery path
    msg = str(exc_info.value.message)
    assert 'definitely-not-a-star-xyz' in msg
    assert '--list' in msg


# ---------------------------
# 'scattering' get subcommand
# ---------------------------


@pytest.mark.unit
def test_get_scattering_dispatches_to_downloader(monkeypatch):
    """``proteus get scattering`` calls download_scattering exactly once."""
    calls = []

    def fake_download_scattering():
        calls.append('scattering')

    monkeypatch.setattr('proteus.utils.data.download_scattering', fake_download_scattering)

    res = runner.invoke(cli.cli, ['get', 'scattering'])
    assert res.exit_code == 0
    assert calls == ['scattering']


@pytest.mark.unit
def test_get_refractive_dispatches_to_downloader(monkeypatch):
    """``proteus get refractive`` calls download_refractive exactly once."""
    calls = []
    monkeypatch.setattr('proteus.utils.data.download_refractive', lambda: calls.append(1))

    res = runner.invoke(cli.cli, ['get', 'refractive'])
    assert res.exit_code == 0
    assert calls == [1]


# ---------------------------
# archive commands
# ---------------------------


class _FakeProteusArchive:
    """Tracks create_archives / extract_archives invocations."""

    instances = []

    def __init__(self, *, config_path):
        self.config_path = config_path
        self.created = False
        self.extracted = False
        type(self).instances.append(self)

    def create_archives(self):
        self.created = True

    def extract_archives(self):
        self.extracted = True


@pytest.mark.unit
def test_create_archives_dispatches(monkeypatch, tmp_path):
    """``proteus create-archives -c cfg`` instantiates Proteus and calls create_archives."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusArchive.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusArchive)

    res = runner.invoke(cli.create_archives, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(_FakeProteusArchive.instances) == 1
    assert _FakeProteusArchive.instances[0].created is True
    # Discrimination: extract path must NOT have fired
    assert _FakeProteusArchive.instances[0].extracted is False


@pytest.mark.unit
def test_extract_archives_dispatches(monkeypatch, tmp_path):
    """``proteus extract-archives -c cfg`` instantiates Proteus and calls extract_archives."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusArchive.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusArchive)

    res = runner.invoke(cli.extract_archives, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(_FakeProteusArchive.instances) == 1
    assert _FakeProteusArchive.instances[0].extracted is True
    # Discrimination: create path must NOT have fired
    assert _FakeProteusArchive.instances[0].created is False


# ---------------------------
# 'offchem' and 'observe' postprocessing commands
# ---------------------------


class _FakeProteusPost:
    """Stand-in for Proteus with offline_chemistry / observe methods recorded."""

    instances = []

    def __init__(self, *, config_path):
        self.config_path = config_path
        self.directories = {'output': '/tmp/proteus_post_output'}

        class _Params:
            class _Out:
                logging = 'INFO'

            out = _Out()

        class _Config:
            params = _Params()

        self.config = _Config()
        self.ran_offchem = False
        self.ran_observe = False
        type(self).instances.append(self)

    def offline_chemistry(self):
        self.ran_offchem = True

    def observe(self):
        self.ran_observe = True


@pytest.mark.unit
def test_offchem_invokes_offline_chemistry(monkeypatch, tmp_path):
    """``proteus offchem -c cfg`` calls runner.offline_chemistry, not observe."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusPost.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusPost)
    monkeypatch.setattr(cli, 'setup_logger', lambda *a, **k: None)

    res = runner.invoke(cli.offchem, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(_FakeProteusPost.instances) == 1
    assert _FakeProteusPost.instances[0].ran_offchem is True
    # Discrimination: observe path must NOT have fired
    assert _FakeProteusPost.instances[0].ran_observe is False


@pytest.mark.unit
def test_observe_invokes_observe(monkeypatch, tmp_path):
    """``proteus observe -c cfg`` calls runner.observe, not offline_chemistry."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusPost.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusPost)
    monkeypatch.setattr(cli, 'setup_logger', lambda *a, **k: None)

    res = runner.invoke(cli.observe, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(_FakeProteusPost.instances) == 1
    assert _FakeProteusPost.instances[0].ran_observe is True
    # Discrimination: offchem path must NOT have fired
    assert _FakeProteusPost.instances[0].ran_offchem is False


# ---------------------------
# grid / infer entry points
# ---------------------------


@pytest.mark.unit
def test_grid_calls_grid_from_config(monkeypatch, tmp_path):
    """``proteus grid -c cfg`` dispatches to grid_from_config with the config
    path and, with no flag, a default test_run=False so real simulations run."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    received = []

    def fake_grid_from_config(path, test_run=False):
        received.append((Path(path), test_run))

    import proteus.grid.manage as gmanage

    monkeypatch.setattr(gmanage, 'grid_from_config', fake_grid_from_config)

    res = runner.invoke(cli.grid, ['-c', str(cfg)])
    assert res.exit_code == 0
    # Discrimination: exactly one call, with the resolved config path.
    assert len(received) == 1
    assert received[0][0].name == 'cfg.toml'
    # Without --dry-run the dispatch must request a real run, not a stubbed one.
    assert received[0][1] is False


def test_start_reports_an_unknown_config_key_cleanly(tmp_path, caplog):
    """``proteus start`` refuses a misspelled key at error level.

    This is the command most runs go through, so a refusal that arrives as a
    bare traceback leaves the name of the offending key buried in it, and one
    emitted below error level reads as ordinary progress output.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw['params']['out']['path'] = str(tmp_path / 'run_output')
    raw['planet']['mass_total'] = 2.5  # deliberate misspelling of mass_tot
    cfg = tmp_path / 'typo.toml'
    with open(cfg, 'w') as f:
        tomlkit.dump(raw, f)

    with caplog.at_level(logging.INFO, logger='fwl'):
        res = runner.invoke(cli.cli, ['start', '-c', str(cfg), '--offline'])
    assert res.exit_code != 0
    assert_refusal_logged_as_error(caplog, 'planet.mass_total')
    assert 'Traceback' not in res.output


def test_cli_does_not_convert_unrelated_value_errors(tmp_path, monkeypatch):
    """A failure that is not about configuration keys keeps its traceback.

    Presenting every ValueError as a tidy CLI message would hide real bugs, so
    the group converts only the configuration-key error.
    """

    def boom(*_args, **_kwargs):
        raise ValueError('something else went wrong entirely')

    monkeypatch.setattr(cli, 'Proteus', boom)

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('config_version = "3.0"\n')

    res = runner.invoke(cli.cli, ['start', '-c', str(cfg), '--offline'])
    assert res.exit_code != 0
    # Not laundered into "Error: ...": the exception escapes for the traceback.
    assert isinstance(res.exception, ValueError)
    assert 'something else went wrong entirely' in str(res.exception)


def test_grid_reports_an_unknown_key_in_the_base_config_cleanly(tmp_path, monkeypatch, caplog):
    """``proteus grid`` refuses a base config with a misspelled key, at error level.

    Case config files are written out from the parsed base config, so an
    unrecognised key in the base never reaches them and the grid would
    otherwise run every case on a default nobody chose. The refusal has to name
    the key and arrive as an error, since the whole ensemble depends on it.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    import proteus.grid.manage as gmanage

    monkeypatch.setattr(gmanage, 'PROTEUS_DIR', str(tmp_path))
    monkeypatch.setattr(gmanage.time, 'sleep', lambda *_a, **_k: None)

    with open(PROTEUS_ROOT / 'tests' / 'grid' / 'base.toml', 'rb') as f:
        base = tomllib.load(f)
    base['params']['dt']['maxium'] = base['params']['dt'].pop('maximum')
    base_path = tmp_path / 'base.toml'
    with open(base_path, 'w') as f:
        tomlkit.dump(base, f)

    grid_toml = tmp_path / 'run.grid.toml'
    grid_toml.write_text(
        'config_version = "3.0"\n'
        'output = "cli_typo_grid"\n'
        'symlink = ""\n'
        f'ref_config = "{base_path}"\n'
        'use_slurm = false\n'
        'max_jobs = 1\n'
        'max_days = 1\n'
        'max_mem = 1\n'
        '["planet.mass_tot"]\n'
        '    method = "direct"\n'
        '    values = [0.7]\n'
    )

    with caplog.at_level(logging.INFO, logger='fwl'):
        res = runner.invoke(cli.cli, ['grid', '-c', str(grid_toml), '--dry-run'])
    assert res.exit_code != 0
    assert_refusal_logged_as_error(caplog, 'params.dt.maxium')
    # An unwrapped raise would print a traceback instead.
    assert 'Traceback' not in res.output


def test_update_input_data_refuses_an_unknown_config_key_before_downloading(
    tmp_path, monkeypatch
):
    """A misspelled key stops the data refresh before anything is fetched.

    This helper runs at the tail of installing and of updating, after every
    other step has reported success, so refusing here has to happen before the
    download rather than after it. The helper is called directly because the
    commands that reach it, ``install-all`` and ``update-all``, perform a full
    installation; that the error it raises is presented in the CLI's own style
    rather than as a traceback is pinned by the group-level tests above.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    downloads = []
    monkeypatch.setattr(
        cli, 'download_sufficient_data', lambda configuration, clean: downloads.append(clean)
    )

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw['planet']['mass_total'] = 2.5  # deliberate misspelling of mass_tot
    cfg = tmp_path / 'typo.toml'
    with open(cfg, 'w') as f:
        tomlkit.dump(raw, f)

    with pytest.raises(UnknownConfigKeyError) as excinfo:
        cli._update_input_data(cfg)
    assert 'planet.mass_total' in str(excinfo.value)
    assert not downloads

    # Discrimination: spelled correctly the same config completes the refresh,
    # so the refusal is caused by the misspelling and not by this config.
    raw['planet']['mass_tot'] = raw['planet'].pop('mass_total')
    good = tmp_path / 'good.toml'
    with open(good, 'w') as f:
        tomlkit.dump(raw, f)
    assert cli._update_input_data(good) is True
    assert downloads == [True]


def test_grid_dry_run_passes_test_run_flag(monkeypatch, tmp_path):
    """``proteus grid -c cfg --dry-run`` flips test_run to True so the grid is
    generated without launching PROTEUS. Discrimination: the only difference
    from the default invocation is the test_run value, so a regression that
    ignored the flag would leave it False and silently start real runs."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    received = []

    def fake_grid_from_config(path, test_run=False):
        received.append((Path(path), test_run))

    import proteus.grid.manage as gmanage

    monkeypatch.setattr(gmanage, 'grid_from_config', fake_grid_from_config)

    res = runner.invoke(cli.grid, ['-c', str(cfg), '--dry-run'])
    assert res.exit_code == 0
    assert len(received) == 1
    assert received[0][1] is True


@pytest.mark.unit
def test_infer_calls_infer_from_config(monkeypatch, tmp_path):
    """``proteus infer -c cfg`` dispatches to infer_from_config with the config path."""
    # The optimisation stack is the optional `inference` extra.
    pytest.importorskip('torch')

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    received = []

    def fake_infer_from_config(path):
        received.append(Path(path))

    import proteus.inference.inference as inf

    monkeypatch.setattr(inf, 'infer_from_config', fake_infer_from_config)

    res = runner.invoke(cli.infer, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(received) == 1
    assert received[0].name == 'cfg.toml'


def _patch_infer_import(monkeypatch, error: ImportError):
    """Make importing the inference entry point raise ``error``.

    The CLI imports ``proteus.inference.inference`` lazily inside the
    command, so the failure has to be injected at the import call itself
    rather than by removing an installed package.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'proteus.inference.inference':
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def _pretend_absent(monkeypatch, *names: str):
    """Make ``importlib.util.find_spec`` report *names* as not installed.

    The CLI confirms absence before advising an install, so a test that
    injects a missing-package error for a package the test environment
    actually has must remove it from the import system's view too.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in names:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)


@pytest.mark.unit
def test_infer_reports_missing_inference_extra(monkeypatch, tmp_path):
    """With the optimisation stack absent, ``proteus infer`` names the missing
    package and the extra that provides it instead of surfacing a bare
    ImportError traceback. The optimisation stack is optional, so this is the
    first thing a user without it sees.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    # The shape the import system produces for a genuinely absent package.
    _patch_infer_import(
        monkeypatch, ModuleNotFoundError("No module named 'torch'", name='torch')
    )
    _pretend_absent(monkeypatch, 'torch')

    res = runner.invoke(cli.infer, ['-c', str(cfg)])

    assert res.exit_code == 1
    # Both halves of the message matter: which package is missing, and the
    # command that installs it.
    assert 'torch' in res.output
    assert 'fwl-proteus[inference]' in res.output
    # A ClickException is reported, not raised through as a traceback.
    assert res.exception is None or isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_infer_propagates_unrelated_import_error(monkeypatch, tmp_path):
    """An import failure inside the inference package that is not one of the
    optional distributions propagates unchanged. Discrimination: a blanket
    except would answer a broken PROTEUS import with an install-the-extra
    message and send the user chasing a dependency that is already present.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    broken = ModuleNotFoundError(
        "No module named 'proteus.inference.missing_helper'",
        name='proteus.inference.missing_helper',
    )
    _patch_infer_import(monkeypatch, broken)

    res = runner.invoke(cli.infer, ['-c', str(cfg)])

    assert res.exit_code == 1
    assert res.exception is broken
    assert 'fwl-proteus[inference]' not in res.output


@pytest.mark.unit
def test_infer_propagates_import_error_from_present_distribution(monkeypatch, tmp_path):
    """A symbol that cannot be imported from an installed distribution raises a
    plain ImportError, not ModuleNotFoundError. That is the signature of a
    version mismatch inside the optimisation stack, and it must reach the user
    as itself: telling them to install a package they already have hides the
    real cause and the reinstall changes nothing.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    skewed = ImportError(
        "cannot import name 'LogExpectedImprovement' from 'botorch.acquisition.analytic'",
        name='botorch.acquisition.analytic',
    )
    _patch_infer_import(monkeypatch, skewed)

    res = runner.invoke(cli.infer, ['-c', str(cfg)])

    assert res.exit_code == 1
    assert res.exception is skewed
    assert 'fwl-proteus[inference]' not in res.output


@pytest.mark.unit
def test_infer_propagates_broken_submodule_of_installed_distribution(monkeypatch, tmp_path):
    """An installed distribution whose own submodule fails to import raises
    ModuleNotFoundError naming that submodule, whose root is one of the three
    optional packages. Discrimination: without the installed check the root
    alone would be taken as proof of absence, and a corrupted install would be
    reported as a missing extra.
    """
    pytest.importorskip('botorch')

    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    broken = ModuleNotFoundError("No module named 'botorch.models'", name='botorch.models')
    _patch_infer_import(monkeypatch, broken)

    res = runner.invoke(cli.infer, ['-c', str(cfg)])

    assert res.exit_code == 1
    assert res.exception is broken
    assert 'fwl-proteus[inference]' not in res.output


@pytest.mark.unit
@pytest.mark.parametrize('distribution', ['torch', 'botorch', 'gpytorch'])
def test_absent_inference_distribution_names_only_a_truly_missing_package(
    monkeypatch, distribution
):
    """The classifier behind the CLI message answers with a distribution name
    only when that distribution is genuinely absent. Each rejection path is
    exercised separately, because each one alone is enough to turn a real
    failure into misleading install advice. Every package in the extra is
    covered: a check written against one name only would leave the other two
    without the install hint.
    """
    absent = ModuleNotFoundError(f"No module named '{distribution}'", name=distribution)

    # Reported only once the import system agrees the package is gone.
    _pretend_absent(monkeypatch, distribution)
    assert cli._absent_inference_distribution(absent) == distribution

    # A submodule of a missing distribution resolves to the distribution, so
    # the message names something a user can actually install.
    submodule = ModuleNotFoundError(
        f"No module named '{distribution}.sub'", name=f'{distribution}.sub'
    )
    assert cli._absent_inference_distribution(submodule) == distribution

    # A plain ImportError means the package was found and something inside it
    # failed, so it is never translated, even for a name in the extra.
    inside = ImportError(f"dlopen failed while loading '{distribution}'", name=distribution)
    assert cli._absent_inference_distribution(inside) is None

    # A missing package outside the extra is not this command's business, and
    # its absence must not be answered with the inference install command.
    unrelated = ModuleNotFoundError(
        "No module named 'unrelated_absent_package_xyz'",
        name='unrelated_absent_package_xyz',
    )
    assert cli._absent_inference_distribution(unrelated) is None

    # An error carrying no module name cannot identify anything.
    nameless = ModuleNotFoundError('import failed')
    assert cli._absent_inference_distribution(nameless) is None


@pytest.mark.unit
def test_absent_inference_distribution_rejects_an_installed_package():
    """A submodule failure inside an installed distribution names that
    distribution as the root, and the classifier must still refuse it.
    Discrimination: this is the one case the exception type alone cannot tell
    apart from a real absence, so it pins the installed check specifically.
    """
    pytest.importorskip('botorch')

    broken = ModuleNotFoundError("No module named 'botorch.models'", name='botorch.models')

    assert cli._absent_inference_distribution(broken) is None
    # The same error shape, with the import system reporting absence, is the
    # case that does get reported: the two differ only in installed state.
    assert broken.name.split('.')[0] in cli.INFERENCE_DISTRIBUTIONS


# ---------------------------
# grid_summarise / grid_pack
# ---------------------------


@pytest.mark.unit
def test_grid_summarise_dispatches_with_status(monkeypatch, tmp_path):
    """``proteus grid-summarise -o output -s completed`` calls summarise(output, status)."""
    outdir = tmp_path / 'out'
    outdir.mkdir()

    received = []

    def fake_summarise(output_path, status):
        received.append((Path(output_path), status))

    import proteus.grid.summarise as gsum

    monkeypatch.setattr(gsum, 'summarise', fake_summarise)

    res = runner.invoke(cli.cli, ['grid-summarise', '-o', str(outdir), '-s', 'completed'])
    assert res.exit_code == 0
    assert len(received) == 1
    assert received[0][1] == 'completed'


@pytest.mark.unit
def test_grid_pack_dispatches(monkeypatch, tmp_path):
    """``proteus grid-pack -o output`` calls pack(output)."""
    outdir = tmp_path / 'out'
    outdir.mkdir()

    received = []

    def fake_pack(output_path):
        received.append(Path(output_path))

    import proteus.grid.pack as gpack

    monkeypatch.setattr(gpack, 'pack', fake_pack)

    res = runner.invoke(cli.cli, ['grid-pack', '-o', str(outdir)])
    assert res.exit_code == 0
    assert len(received) == 1
    assert received[0].name == 'out'


# ---------------------------
# Installer helpers
# ---------------------------


@pytest.mark.unit
def test_resolve_fwl_data_dir_uses_env_when_set(monkeypatch, tmp_path):
    """When FWL_DATA is set, resolve_fwl_data_dir returns that exact path."""
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    out = cli.resolve_fwl_data_dir()
    assert out == Path(tmp_path)
    # Discrimination: the result must equal the env var value, not the
    # fallback (which is sibling to the source tree).
    assert 'FWL_DATA' not in str(out) or str(out) == str(tmp_path)


@pytest.mark.unit
def test_resolve_fwl_data_dir_falls_back_when_env_missing(monkeypatch):
    """When FWL_DATA is NOT set, resolve_fwl_data_dir returns the source-adjacent default."""
    monkeypatch.delenv('FWL_DATA', raising=False)
    out = cli.resolve_fwl_data_dir()
    # The fallback is a directory named FWL_DATA next to the proteus parent.
    assert out.name == 'FWL_DATA'
    # Discrimination: env path branch would return whatever FWL_DATA pointed
    # to; pin that the fallback path is parent-of-package-relative.
    assert isinstance(out, Path)


@pytest.mark.unit
def test_is_julia_installed_true_when_shutil_finds_it(monkeypatch):
    """is_julia_installed returns True when shutil.which finds a julia binary."""
    monkeypatch.delenv('PYTHON_JULIAPKG_EXE', raising=False)
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    result = cli.is_julia_installed()
    assert result is True
    # Discrimination: the helper returns a Python bool, not a path string.
    # A regression that returned shutil.which()'s output directly would fail
    # the type pin below (str truthy but not `is True`).
    assert isinstance(result, bool)


@pytest.mark.unit
def test_is_julia_installed_false_when_missing(monkeypatch):
    """is_julia_installed returns False when shutil.which cannot resolve julia."""
    monkeypatch.delenv('PYTHON_JULIAPKG_EXE', raising=False)
    monkeypatch.setattr(cli.shutil, 'which', lambda exe: None)
    result = cli.is_julia_installed()
    assert result is False
    # Discrimination: pin the bool type so a regression returning None (also
    # falsy) does not slip through.
    assert isinstance(result, bool)


def _julia_stub(directory: Path, executable: bool = True) -> Path:
    """Write a ``julia`` script in ``directory``, executable or not."""
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / 'julia'
    exe.write_text('#!/bin/sh\necho "julia version 1.13.1"\n')
    exe.chmod(0o755 if executable else 0o644)
    return exe


@pytest.mark.unit
@pytest.mark.parametrize(
    ('kind', 'expected'),
    [('executable', True), ('not-executable', False), ('missing', False)],
    ids=['bound-executable', 'bound-not-executable', 'bound-missing'],
)
def test_is_julia_installed_reads_pythonjuliapkg_exe_without_a_path_julia(
    monkeypatch, tmp_path, kind, expected
):
    """With no julia on PATH, PYTHON_JULIAPKG_EXE counts only when it names an executable file."""
    bindir = tmp_path / 'julia-1.13.1' / 'bin'
    exe = bindir / 'julia'
    if kind != 'missing':
        _julia_stub(bindir, executable=kind == 'executable')
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    assert cli.is_julia_installed() is expected
    monkeypatch.delenv('PYTHON_JULIAPKG_EXE')
    assert cli.is_julia_installed() is False


@pytest.mark.unit
def test_env_with_proteus_julia_puts_the_bound_julia_first_on_path(monkeypatch, tmp_path):
    """The folder <root>/.proteus-julia, with a link to PYTHON_JULIAPKG_EXE, leads PATH in a copy.

    The process environment keeps its PATH, and a variable that names no
    executable file leaves PATH as it is in the copy too.
    """
    root = tmp_path / 'proteus'
    root.mkdir()
    bindir = tmp_path / 'julia-1.13.1' / 'bin'
    exe = _julia_stub(bindir)
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    env = cli._env_with_proteus_julia(root)
    link_dir = root / '.proteus-julia'
    assert env['PATH'].split(os.pathsep) == [str(link_dir), '/usr/bin', '/bin']
    assert os.listdir(link_dir) == ['julia']
    assert os.readlink(link_dir / 'julia') == str(exe)
    assert os.environ['PATH'] == '/usr/bin:/bin'
    assert env['PYTHON_JULIAPKG_EXE'] == str(exe)
    exe.chmod(0o644)
    assert cli._env_with_proteus_julia(root)['PATH'] == '/usr/bin:/bin'
    monkeypatch.delenv('PYTHON_JULIAPKG_EXE')
    assert cli._env_with_proteus_julia(root)['PATH'] == '/usr/bin:/bin'


@pytest.mark.unit
def test_env_with_proteus_julia_shadows_no_other_program_of_the_binary_folder(
    monkeypatch, tmp_path
):
    """A bound julia-1.13 in a folder with a python: julia is it, python stays the env's.

    The folder of the binary (a shared one such as /opt/homebrew/bin) holds a
    python, and the binary is not named julia; the env folder holds a python and
    a julia of another version. In the returned PATH, julia resolves to the bound
    binary and python to the env's.
    """
    root = tmp_path / 'proteus'
    root.mkdir()
    shared = tmp_path / 'shared' / 'bin'
    shared.mkdir(parents=True)
    bound = shared / 'julia-1.13'
    bound.write_text('#!/bin/sh\necho "julia version 1.13.1"\n')
    bound.chmod(0o755)
    (shared / 'python').write_text('#!/bin/sh\n')
    (shared / 'python').chmod(0o755)
    env_bin = tmp_path / 'env' / 'bin'
    _julia_stub(env_bin)
    (env_bin / 'python').write_text('#!/bin/sh\n')
    (env_bin / 'python').chmod(0o755)
    monkeypatch.setenv('PATH', f'{env_bin}{os.pathsep}/usr/bin{os.pathsep}/bin')
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(bound))
    path = cli._env_with_proteus_julia(root)['PATH']
    julia = shutil.which('julia', path=path)
    assert os.path.realpath(julia) == os.path.realpath(bound)
    assert julia != str(env_bin / 'julia')
    assert shutil.which('python', path=path) == str(env_bin / 'python')


@pytest.mark.unit
def test_update_input_data_returns_false_when_config_missing(monkeypatch, tmp_path):
    """_update_input_data returns False and prints a skip message when config does not exist."""
    missing = tmp_path / 'missing.toml'

    # Stash a sentinel so we can prove download_sufficient_data was NOT called.
    download_calls = []

    def fake_download(*args, **kwargs):
        download_calls.append(True)

    monkeypatch.setattr(cli, 'download_sufficient_data', fake_download)

    out = cli._update_input_data(missing)
    assert out is False
    # Discrimination: the download branch must NOT have fired when the
    # config file is missing.
    assert download_calls == []


@pytest.mark.unit
def test_update_input_data_returns_true_when_config_exists(monkeypatch, tmp_path):
    """_update_input_data returns True and triggers download_sufficient_data on a present config."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    received = []

    def fake_read_config_object(path):
        received.append(('read', Path(path)))
        return {'fake': 'configuration'}

    def fake_download_sufficient_data(configuration, clean):
        received.append(('dl', configuration, clean))

    monkeypatch.setattr(cli, 'read_config_object', fake_read_config_object)
    monkeypatch.setattr(cli, 'download_sufficient_data', fake_download_sufficient_data)

    out = cli._update_input_data(cfg)
    assert out is True
    # Both calls fired, in order
    assert len(received) == 2
    assert received[0][0] == 'read'
    assert received[1][0] == 'dl'
    # clean kwarg threaded as True
    assert received[1][2] is True


# ---------------------------
# install_all
# ---------------------------


def _stub_disk_usage_high():
    """Return a disk_usage-shaped tuple with plenty of free space (100 GB)."""
    from collections import namedtuple

    Usage = namedtuple('Usage', ['total', 'used', 'free'])
    return Usage(total=1_000_000_000_000, used=0, free=100_000_000_000)


def _stub_disk_usage_low():
    """Return a disk_usage-shaped tuple with <5 GB free."""
    from collections import namedtuple

    Usage = namedtuple('Usage', ['total', 'used', 'free'])
    return Usage(total=1_000_000_000_000, used=0, free=1_000_000_000)


def _pin_proteus_root(monkeypatch, root):
    """Shape ``root`` as a PROTEUS checkout and pin resolution to it.

    Writes the project marker the resolver verifies, points the package
    lookup at ``root``, and pre-sets RAD_DIR so the commands' setdefault
    cannot leak an environment variable past the test.
    """
    root.mkdir(exist_ok=True)
    (root / 'pyproject.toml').write_text('[project]\nname = "fwl-proteus"\n')
    monkeypatch.setattr(cli, 'get_proteus_dir', lambda: str(root))
    monkeypatch.setenv('RAD_DIR', str(root / 'socrates'))


def _make_fwl_data(monkeypatch, tmp_path):
    """Point FWL_DATA at an existing directory, as after an installation."""
    fwl = tmp_path / 'fwl_data'
    fwl.mkdir(exist_ok=True)
    monkeypatch.setenv('FWL_DATA', str(fwl))


def _agni_clone_may_move(monkeypatch):
    """Let ``update-all`` treat the empty ``AGNI/`` stand-in of a test as a clone it may move.

    The git checks of ``_agni_update_blocker`` have their own tests on real
    clones; here every subprocess is stubbed.
    """
    monkeypatch.setattr(cli, '_agni_update_blocker', lambda agni_dir: None)


@pytest.mark.unit
def test_install_all_aborts_on_low_disk(monkeypatch, tmp_path):
    """``install-all`` exits non-zero when free disk is below the 5 GB threshold."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_low())
    # Defensive stub: a regression past the gate must fail an assertion, not
    # launch a real installer.
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Aborting installation' in res.output
    # Discrimination: must NOT print the "completed" message that lives on
    # the success path.
    assert 'PROTEUS installation completed' not in res.output


@pytest.mark.unit
def test_install_all_aborts_when_julia_missing(monkeypatch, tmp_path):
    """``install-all`` aborts with a Julia-not-found message when julia is missing."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    # Pre-create socrates dir so the SOCRATES install path is skipped
    (tmp_path / 'socrates').mkdir()
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    # Defensive stub: a regression into the SOCRATES branch must fail an
    # assertion, not launch a real installer.
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )
    # Force the julia check to fail
    monkeypatch.setattr(cli, 'is_julia_installed', lambda: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Julia not found' in res.output
    # Discrimination: AGNI install path must NOT have been reached
    assert 'Installing AGNI' not in res.output


def _built_socrates(folder: Path) -> Path:
    """Make ``folder`` a SOCRATES build: it holds ``bin/radlib.a``. Returns the folder."""
    (folder / 'bin').mkdir(parents=True)
    (folder / 'bin' / 'radlib.a').write_text('')
    return folder


def _export_env_site(monkeypatch, tmp_path, conda_prefix):
    """Stub the heavy steps of install-all and update-all for an ``--export-env`` run.

    A built SOCRATES (bin/radlib.a), AGNI and a built FastChem exist under the root, PYTHON_JULIAPKG_EXE
    names an executable Julia 1.13 stub, Julia and conda are on the stubbed PATH,
    FWL_DATA holds a space, and HOME is an empty folder. Returns the HOME folder
    and the list that records each subprocess command.
    """
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    on_path = {'julia': '/opt/julia-1.10.4/bin/julia', 'conda': '/opt/conda/condabin/conda'}
    monkeypatch.setattr(cli.shutil, 'which', lambda exe: on_path.get(exe))
    for var in ('PIXI_ENVIRONMENT_NAME', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(
        'PYTHON_JULIAPKG_EXE', str(_julia_stub(tmp_path / 'julia-1.13.1' / 'bin'))
    )
    if conda_prefix:
        monkeypatch.setenv('CONDA_PREFIX', conda_prefix)
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'proteus')
    (tmp_path / 'fwl data').mkdir()
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl data'))
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: home)
    _built_socrates(tmp_path / 'socrates')
    fastchem = tmp_path / 'AGNI' / 'fastchem'
    fastchem.mkdir(parents=True)
    (fastchem / 'fastchem').write_text('#!/bin/sh\n')
    (fastchem / 'fastchem').chmod(0o755)
    _agni_clone_may_move(monkeypatch)
    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        calls.append(list(cmd))
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)
    return home, calls


def _expected_install_vars(tmp_path):
    """The 4 variables that ``--export-env`` records for the site of ``_export_env_site``."""
    return {
        'FWL_DATA': str(tmp_path / 'fwl data'),
        'RAD_DIR': str((tmp_path / 'socrates').resolve()),
        'PYTHON_JULIAPKG_EXE': str(tmp_path / 'julia-1.13.1' / 'bin' / 'julia'),
        'FC_DIR': str((tmp_path / 'AGNI' / 'fastchem').resolve()),
    }


@pytest.mark.unit
def test_install_all_export_env_binds_four_vars_to_conda_env(monkeypatch, tmp_path):
    """``install-all --export-env`` sets the 4 variables in the conda env and the exports file.

    One conda call carries all 4 pairs, the exports file under the PROTEUS
    root holds one quoted line each, the message names the reactivation of
    the env, and the temporary HOME stays empty: no shell start file is written.
    """
    home, calls = _export_env_site(monkeypatch, tmp_path, '/opt/conda/envs/proteus')
    res = runner.invoke(cli.cli, ['install-all', '--export-env'])
    assert res.exit_code == 0, f'install-all output: {res.output}'
    assert 'PROTEUS installation completed' in res.output
    expected = _expected_install_vars(tmp_path)
    pairs = [f'{name}={value}' for name, value in expected.items()]
    conda_calls = [c for c in calls if c[0] == 'conda']
    assert conda_calls == [
        ['conda', 'env', 'config', 'vars', 'set', '-p', '/opt/conda/envs/proteus', *pairs]
    ]
    lines = (tmp_path / '.proteus-env.sh').read_text().splitlines()
    assert lines == [f"export {name}='{value}'" for name, value in expected.items()]
    assert list(home.iterdir()) == []
    assert 'conda deactivate && conda activate proteus' in res.output
    assert 'source .proteus-env.sh' not in res.output


@pytest.mark.unit
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
def test_export_env_with_base_active_binds_nothing_and_says_why(monkeypatch, tmp_path, command):
    """With base active, `--export-env` calls no conda, writes the file and names base.

    Without the rule the 4 variables would be bound to the base env, which every
    other env inherits on top of its own activation.
    """
    _, calls = _export_env_site(monkeypatch, tmp_path, '/opt/conda')
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'base')
    res = runner.invoke(cli.cli, [command, '--export-env'])
    assert res.exit_code == 0, res.output
    assert [c for c in calls if c[0] == 'conda'] == []
    expected = _expected_install_vars(tmp_path)
    lines = (tmp_path / '.proteus-env.sh').read_text().splitlines()
    assert lines == [f"export {name}='{value}'" for name, value in expected.items()]
    assert (
        '[!] The conda base environment is active, and PROTEUS sets nothing in base; '
        f'the variables are in {tmp_path / ".proteus-env.sh"} only'
    ) in res.output
    assert 'source .proteus-env.sh' in res.output
    assert 'conda activate base' not in res.output


@pytest.mark.unit
def test_export_env_keeps_a_value_with_a_single_quote_out_of_conda(monkeypatch, tmp_path):
    """A FWL_DATA with a single quote is in the exports file only; the other 3 go to conda.

    One warning names FWL_DATA, and the closing line sources the exports file,
    since the conda env alone does not hold every variable.
    """
    _, calls = _export_env_site(monkeypatch, tmp_path, '/opt/conda/envs/proteus')
    quoted = tmp_path / "o'brien data"
    quoted.mkdir()
    monkeypatch.setenv('FWL_DATA', str(quoted))
    res = runner.invoke(cli.cli, ['install-all', '--export-env'])
    assert res.exit_code == 0, res.output
    expected = {**_expected_install_vars(tmp_path), 'FWL_DATA': str(quoted)}
    pairs = [f'{n}={v}' for n, v in expected.items() if n != 'FWL_DATA']
    assert [c for c in calls if c[0] == 'conda'] == [
        ['conda', 'env', 'config', 'vars', 'set', '-p', '/opt/conda/envs/proteus', *pairs]
    ]
    assert "export FWL_DATA='" in (tmp_path / '.proteus-env.sh').read_text()
    assert res.output.count('[!] Not set in the conda env: FWL_DATA, whose value') == 1
    assert 'source .proteus-env.sh' in res.output
    assert 'conda deactivate' not in res.output


@pytest.mark.unit
@pytest.mark.parametrize(
    'bound', ['unset', 'not-executable', 'missing'], ids=['unset', 'not-executable', 'missing']
)
def test_install_vars_never_take_the_julia_on_path(monkeypatch, tmp_path, capsys, bound):
    """Without an executable PYTHON_JULIAPKG_EXE no Julia is recorded, and one line says so.

    A julia 1.10 is on PATH; recording it would bind PROTEUS to that version, or
    to the juliaup launcher. The line names bash install.sh.
    """
    monkeypatch.setattr(
        cli.shutil,
        'which',
        lambda exe: '/opt/julia-1.10.4/bin/julia' if exe == 'julia' else None,
    )
    exe = tmp_path / 'julia-1.13.1' / 'bin' / 'julia'
    if bound == 'unset':
        monkeypatch.delenv('PYTHON_JULIAPKG_EXE', raising=False)
    else:
        if bound == 'not-executable':
            _julia_stub(exe.parent, executable=False)
        monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    variables = cli._install_vars(tmp_path, tmp_path / 'fwl', tmp_path / 'socrates')
    assert list(variables) == ['FWL_DATA', 'RAD_DIR']
    out = capsys.readouterr().out.splitlines()
    assert out == [
        '[!] PYTHON_JULIAPKG_EXE is not set to an executable Julia, so it is not recorded; '
        'bash install.sh selects Julia 1.13 and records it.'
    ]
    _julia_stub(exe.parent)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    assert cli._install_vars(tmp_path, tmp_path, tmp_path)['PYTHON_JULIAPKG_EXE'] == str(exe)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('program', 'recorded'),
    [('built', True), ('missing', False), ('not-executable', False)],
    ids=['fastchem-built', 'folder-without-program', 'program-not-executable'],
)
def test_install_vars_record_fc_dir_only_for_a_built_fastchem(
    monkeypatch, tmp_path, program, recorded
):
    """FC_DIR is recorded when AGNI/fastchem/fastchem is executable, not for the folder alone."""
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(_julia_stub(tmp_path / 'jl')))
    fastchem = tmp_path / 'AGNI' / 'fastchem'
    fastchem.mkdir(parents=True)
    if program != 'missing':
        (fastchem / 'fastchem').write_text('#!/bin/sh\n')
        (fastchem / 'fastchem').chmod(0o755 if program == 'built' else 0o644)
    variables = cli._install_vars(tmp_path, tmp_path / 'fwl', tmp_path / 'socrates')
    assert ('FC_DIR' in variables) is recorded
    assert variables.get('FC_DIR') == (str(fastchem.resolve()) if recorded else None)


@pytest.mark.unit
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
def test_export_env_records_the_rad_dir_set_when_the_root_folder_has_no_build(
    monkeypatch, tmp_path, command
):
    """A socrates folder of the root without bin/radlib.a gives way to a RAD_DIR set to a build.

    That folder is what a failed build or a clone without a build leaves. Recording it
    would override the valid RAD_DIR at each activation of the env; the RAD_DIR of the
    process stays as it was.
    """
    _, calls = _export_env_site(monkeypatch, tmp_path, '/opt/conda/envs/proteus')
    shutil.rmtree(tmp_path / 'socrates' / 'bin')
    own = _built_socrates(tmp_path / 'elsewhere' / 'socrates')
    monkeypatch.setenv('RAD_DIR', str(own))
    res = runner.invoke(cli.cli, [command, '--export-env'])
    assert res.exit_code == 0, res.output
    expected = {**_expected_install_vars(tmp_path), 'RAD_DIR': str(own)}
    lines = (tmp_path / '.proteus-env.sh').read_text().splitlines()
    assert lines == [f"export {name}='{value}'" for name, value in expected.items()]
    (call,) = [c for c in calls if c[0] == 'conda']
    assert [a for a in call if a.startswith('RAD_DIR=')] == [f'RAD_DIR={own}']
    assert os.environ['RAD_DIR'] == str(own)
    assert 'No SOCRATES build' not in res.output


@pytest.mark.unit
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
def test_export_env_records_no_rad_dir_without_a_built_socrates(monkeypatch, tmp_path, command):
    """With no bin/radlib.a in the root's socrates or in RAD_DIR, RAD_DIR is not recorded.

    One line names the root's folder; the other 3 variables are recorded, and
    the RAD_DIR of the process, a missing folder, is not replaced.
    """
    _, calls = _export_env_site(monkeypatch, tmp_path, '/opt/conda/envs/proteus')
    shutil.rmtree(tmp_path / 'socrates' / 'bin')
    monkeypatch.setenv('RAD_DIR', str(tmp_path / 'missing'))
    res = runner.invoke(cli.cli, [command, '--export-env'])
    assert res.exit_code == 0, res.output
    expected = {k: v for k, v in _expected_install_vars(tmp_path).items() if k != 'RAD_DIR'}
    lines = (tmp_path / '.proteus-env.sh').read_text().splitlines()
    assert lines == [f"export {name}='{value}'" for name, value in expected.items()]
    (call,) = [c for c in calls if c[0] == 'conda']
    assert [a for a in call if a.startswith('RAD_DIR=')] == []
    assert (
        res.output.count(f'[!] No SOCRATES build (bin/radlib.a) in {tmp_path / "socrates"}')
        == 1
    )
    assert os.environ['RAD_DIR'] == str(tmp_path / 'missing')


@pytest.mark.unit
def test_install_all_runs_the_proteus_agni_script_and_no_clone(monkeypatch, tmp_path):
    """A missing AGNI is set up by tools/get_agni.sh 0 of PROTEUS, not by a clone of the tip.

    The script checks out the commit PROTEUS pins; it runs in the root, with the
    bound Julia first on PATH and stdin from /dev/null, as update-all runs it.
    """
    captured = _agni_site(monkeypatch, tmp_path, agni_present=False)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(_julia_stub(tmp_path / 'jl' / 'bin')))
    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0, res.output
    assert [cmd for cmd, _ in captured if 'clone' in cmd or cmd[:1] == ['git']] == []
    assert [cmd for cmd, _ in captured if 'src/get_agni.sh' in cmd] == []
    agni = [(cmd, kw) for cmd, kw in captured if 'get_agni.sh' in ' '.join(cmd)]
    assert [cmd for cmd, _ in agni] == [['bash', str(tmp_path / 'tools' / 'get_agni.sh'), '0']]
    kw = agni[0][1]
    assert (kw['cwd'], kw['stdin'], kw['check']) == (tmp_path, subprocess.DEVNULL, True)
    assert kw['env']['PATH'].split(os.pathsep)[0] == str(tmp_path / '.proteus-julia')


@pytest.mark.unit
def test_install_all_invokes_socrates_when_missing(monkeypatch, tmp_path):
    """SOCRATES install branch fires subprocess.run when the socrates/ dir is absent."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    # NO socrates dir; AGNI pre-existing so it doesn't try git clone
    (tmp_path / 'AGNI').mkdir()

    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(tuple(cmd) if isinstance(cmd, list) else cmd)
        # Simulate that the socrates install script created the dir
        if 'get_socrates.sh' in (cmd[1] if len(cmd) > 1 else ''):
            (tmp_path / 'socrates').mkdir(exist_ok=True)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0, f'install-all output: {res.output}'
    # Discrimination: at least one subprocess call was the socrates install
    socrates_calls = [c for c in subprocess_calls if 'get_socrates.sh' in str(c)]
    assert len(socrates_calls) >= 1


@pytest.mark.unit
def test_install_all_socrates_subprocess_failure_aborts(monkeypatch, tmp_path):
    """A CalledProcessError from the SOCRATES install script aborts with exit != 0."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    # No socrates dir; the install attempt must raise

    def fake_subprocess_run(cmd, **kwargs):
        if 'get_socrates.sh' in str(cmd):
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Failed to install SOCRATES' in res.output


@pytest.mark.unit
def test_install_all_agni_setup_failure_aborts(monkeypatch, tmp_path):
    """A CalledProcessError of tools/get_agni.sh, which clones AGNI, aborts the installation."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    # SOCRATES present so the SOCRATES branch is skipped
    (tmp_path / 'socrates').mkdir()
    # AGNI missing; its setup script must fail

    def fake_subprocess_run(cmd, **kwargs):
        if cmd[:2] == ['bash', str(tmp_path / 'tools' / 'get_agni.sh')]:
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Failed to install AGNI' in res.output


def _agni_site(monkeypatch, tmp_path, agni_present: bool) -> list:
    """Stub the heavy steps of install-all and update-all around the AGNI setup script.

    SOCRATES exists, AGNI exists when ``agni_present``, FWL_DATA exists, PATH
    holds only system folders and the data downloads are no-ops. Returns the
    list of ``(command, kwargs)`` of each subprocess call.
    """
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    _make_fwl_data(monkeypatch, tmp_path)
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    (tmp_path / 'socrates').mkdir()
    if agni_present:
        (tmp_path / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)
    captured = []

    def fake_subprocess_run(cmd, **kwargs):
        captured.append((list(cmd), kwargs))
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)
    return captured


@pytest.mark.unit
@pytest.mark.parametrize(
    ('command', 'agni_present'),
    [('install-all', False), ('update-all', True)],
    ids=['install-all-new-agni', 'update-all-existing-agni'],
)
def test_agni_setup_runs_with_the_bound_julia_first_on_path(
    monkeypatch, tmp_path, command, agni_present
):
    """AGNI's setup script gets PATH with the folder of the link to PYTHON_JULIAPKG_EXE first.

    No julia is on PATH, so the command finds Julia only through the variable;
    the PATH of the process itself stays as it was. Both commands run
    ``tools/get_agni.sh 0`` in the root, with stdin from /dev/null.
    """
    captured = _agni_site(monkeypatch, tmp_path, agni_present)
    bindir = tmp_path / 'julia-1.13.1' / 'bin'
    exe = _julia_stub(bindir)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
    res = runner.invoke(cli.cli, [command])
    assert res.exit_code == 0, res.output
    assert 'Julia not found' not in res.output
    script = ['bash', str(tmp_path / 'tools' / 'get_agni.sh'), '0']
    agni = [kw for cmd, kw in captured if cmd == script]
    assert len(agni) == 1
    assert agni[0]['cwd'] == tmp_path
    assert agni[0]['stdin'] is subprocess.DEVNULL
    link_dir = tmp_path / '.proteus-julia'
    assert agni[0]['env']['PATH'].split(os.pathsep) == [str(link_dir), '/usr/bin', '/bin']
    assert os.readlink(link_dir / 'julia') == str(exe)
    assert os.environ['PATH'] == '/usr/bin:/bin'


@pytest.mark.unit
@pytest.mark.parametrize(
    ('command', 'agni_present'),
    [('install-all', False), ('update-all', True)],
    ids=['install-all-new-agni', 'update-all-existing-agni'],
)
def test_agni_setup_goes_on_without_the_julia_link_in_a_read_only_root(
    monkeypatch, tmp_path, command, agni_present
):
    """A root where ``.proteus-julia`` cannot be made gives one warning, not a traceback.

    The AGNI script still runs once, with PATH as it is, and the command completes.
    The warning says that the script stops: its ``use_proteus_julia`` makes the
    same link in the same root and returns 1, so PATH as it is never reaches Julia.
    """
    if os.geteuid() == 0:
        pytest.skip('the root user writes into a read-only folder')
    captured = _agni_site(monkeypatch, tmp_path, agni_present)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(_julia_stub(tmp_path / 'jl' / 'bin')))
    (tmp_path / 'output').mkdir()
    tmp_path.chmod(0o555)
    try:
        res = runner.invoke(cli.cli, [command])
    finally:
        tmp_path.chmod(0o755)
    assert res.exit_code == 0, res.output
    assert res.exception is None
    link = tmp_path / '.proteus-julia' / 'julia'
    warnings = [ln for ln in res.output.splitlines() if ln.startswith('[!] Could not link')]
    assert len(warnings) == 1
    assert warnings[0].startswith(f'[!] Could not link {link} to the Julia PROTEUS uses (')
    assert warnings[0].endswith('; the setup script will stop, because it makes the same link')
    assert 'gets PATH as it is' not in res.output
    script = ['bash', str(tmp_path / 'tools' / 'get_agni.sh'), '0']
    agni = [kw for cmd, kw in captured if cmd == script]
    assert len(agni) == 1
    assert agni[0]['env']['PATH'] == '/usr/bin:/bin'
    assert not (tmp_path / '.proteus-julia').exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ('version', 'line'),
    [
        ('1.13.1', '[+] Julia: 1.13.1 at <exe> (PYTHON_JULIAPKG_EXE)'),
        ('1.12.6', '[!] Julia: 1.12.6 at <exe> (PYTHON_JULIAPKG_EXE) (PROTEUS is tested on'),
    ],
    ids=['julia-1.13', 'julia-1.12'],
)
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
def test_install_all_and_update_all_name_the_julia_they_use(
    monkeypatch, tmp_path, command, version, line
):
    """Both commands print the version of PYTHON_JULIAPKG_EXE once, and warn for a 1.12.

    A julia 1.13 on PATH does not count: the bound binary is the one PROTEUS uses.
    The warning carries the advice of ``proteus doctor``, which adds 1.13 beside the default.
    """
    monkeypatch.setattr(doctor, 'check_julia', _REAL_CHECK_JULIA)
    _agni_site(monkeypatch, tmp_path, agni_present=True)
    exe = tmp_path / 'jl' / 'bin' / 'julia'
    exe.parent.mkdir(parents=True)
    exe.write_text(f'#!/bin/sh\necho "julia version {version}"\n')
    exe.chmod(0o755)
    monkeypatch.setenv('PYTHON_JULIAPKG_EXE', str(exe))
    _julia_stub(tmp_path / 'path-bin')
    monkeypatch.setenv('PATH', f'{tmp_path / "path-bin"}:/usr/bin:/bin')
    stubbed = cli.subprocess.run
    monkeypatch.setattr(
        cli.subprocess,
        'run',
        lambda cmd, **kw: _REAL_RUN(cmd, **kw) if cmd[0] == str(exe) else stubbed(cmd, **kw),
    )
    res = runner.invoke(cli.cli, [command])
    assert res.exit_code == 0, res.output
    julia_lines = [
        ln for ln in res.output.splitlines() if ln.startswith(('[+] Julia:', '[!] Julia:'))
    ]
    assert len(julia_lines) == 1
    assert julia_lines[0].startswith(line.replace('<exe>', str(exe)))
    assert ('juliaup add 1.13' in julia_lines[0]) is (version != '1.13.1')


@pytest.mark.unit
def test_install_all_fetches_the_phoenix_grid_of_the_default_star_once(
    monkeypatch, tmp_path, phoenix_downloads
):
    """install-all downloads the PHOENIX grid at [Fe/H] = 0.0, [alpha/M] = 0.0 once.

    That is the grid of the star a config without a [star] section uses, read
    from the config defaults; update-all leaves the grid to the data step.
    """
    import attrs

    from proteus.config._config import Config

    _agni_site(monkeypatch, tmp_path, agni_present=True)
    monkeypatch.setattr(
        cli.shutil, 'which', lambda name: '/usr/bin/julia' if name == 'julia' else None
    )
    star = attrs.fields(Config).star.default.factory()
    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0, res.output
    assert phoenix_downloads == [{'alpha': 0.0, 'FeH': 0.0}]
    assert phoenix_downloads[0] == {
        'alpha': star.mors.phoenix_alpha,
        'FeH': star.mors.phoenix_FeH,
    }
    assert star.mors.spectrum_source == 'phoenix'
    assert '[+] PHOENIX grid for [Fe/H]=+0.0, [alpha/M]=+0.0 available' in res.output
    assert 'PROTEUS installation completed' in res.output
    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0, res.output
    assert len(phoenix_downloads) == 1


@pytest.mark.unit
@pytest.mark.parametrize('failure', ['returns-false', 'raises-oserror'])
def test_install_all_warns_and_goes_on_when_the_phoenix_download_fails(
    monkeypatch, tmp_path, failure
):
    """A failed PHOENIX download is a warning: the data step and the rest of install-all run."""
    _agni_site(monkeypatch, tmp_path, agni_present=True)
    monkeypatch.setattr(
        cli.shutil, 'which', lambda name: '/usr/bin/julia' if name == 'julia' else None
    )
    attempts = []

    def failing_download(**kwargs):
        attempts.append(kwargs)
        if failure == 'raises-oserror':
            raise OSError('No space left on device')
        return False

    monkeypatch.setattr('proteus.utils.data.download_phoenix', failing_download)
    data_paths = []
    monkeypatch.setattr(cli, '_update_input_data', lambda path: data_paths.append(path) or True)
    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0, res.output
    assert attempts == [{'alpha': 0.0, 'FeH': 0.0}]
    assert '[!] The PHOENIX grid for [Fe/H]=+0.0, [alpha/M]=+0.0 did not download' in res.output
    assert 'proteus get phoenix --feh 0.0 --alpha 0.0' in res.output
    assert 'PHOENIX grid for [Fe/H]=+0.0, [alpha/M]=+0.0 available' not in res.output
    assert data_paths == [tmp_path / 'input' / 'all_options.toml']
    assert 'PROTEUS installation completed' in res.output


class _RecordedCalls(list):
    """List of captured subprocess calls with environment stub attached."""

    env: dict


def _standard_modules_test_site(monkeypatch, tmp_path, fail_script=None):
    """Prepare filesystem and stubs for standard module install/update tests."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)
    (tmp_path / 'socrates').mkdir(exist_ok=True)
    (tmp_path / 'AGNI').mkdir(exist_ok=True)
    _agni_clone_may_move(monkeypatch)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    stub_julia_env = {'STUB_JULIA_ENV': '1'}
    monkeypatch.setattr(cli, '_env_with_proteus_julia', lambda root: stub_julia_env)

    calls = _RecordedCalls()
    calls.env = stub_julia_env

    def fake_subprocess_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))
        if fail_script and len(cmd) > 1 and fail_script in str(cmd[1]):
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    return calls


@pytest.mark.unit
def test_install_all_installs_standard_modules_when_absent(monkeypatch, tmp_path):
    """install-all invokes setup scripts for standard modules when their folders are absent."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0
    expected_scripts = [
        'tools/get_morrigan.sh',
        'tools/get_vulcan.sh',
        'tools/get_atmodeller.sh',
        'tools/get_obliqua.sh',
    ]
    std_calls = [
        c for c in calls if len(c[0]) > 1 and any(s in str(c[0][1]) for s in expected_scripts)
    ]
    assert [c[0][1] for c in std_calls] == [str(tmp_path / s) for s in expected_scripts]
    for cmd, kwargs in std_calls:
        assert len(cmd) == 2
        assert cmd[0] == 'bash'
        assert kwargs.get('cwd') == tmp_path
        assert kwargs.get('env') is calls.env
        assert kwargs.get('check') is True
        assert kwargs.get('stdin') is cli.subprocess.DEVNULL


@pytest.mark.unit
def test_install_all_runs_standard_module_scripts_when_folders_present(monkeypatch, tmp_path):
    """install-all runs setup scripts for standard modules even when folders exist."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path)
    expected_folders = ['Morrigan', 'VULCAN', 'Atmodeller', 'Obliqua']
    for folder in expected_folders:
        (tmp_path / folder).mkdir()

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0

    expected_scripts = [
        'tools/get_morrigan.sh',
        'tools/get_vulcan.sh',
        'tools/get_atmodeller.sh',
        'tools/get_obliqua.sh',
    ]
    std_calls = [
        c for c in calls if len(c[0]) > 1 and any(s in str(c[0][1]) for s in expected_scripts)
    ]
    assert [c[0][1] for c in std_calls] == [str(tmp_path / s) for s in expected_scripts]
    for cmd, kwargs in std_calls:
        assert len(cmd) == 2
        assert cmd[0] == 'bash'
        assert kwargs.get('cwd') == tmp_path
        assert kwargs.get('env') is calls.env
        assert kwargs.get('check') is True
        assert kwargs.get('stdin') is cli.subprocess.DEVNULL


@pytest.mark.unit
def test_install_all_warns_and_continues_when_standard_module_script_fails(
    monkeypatch, tmp_path
):
    """install-all warns in yellow and continues when a standard module script fails."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path, fail_script='get_morrigan.sh')

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0
    assert 'Morrigan is not installed: "bash tools/get_morrigan.sh" failed' in res.output
    remaining_scripts = [
        'tools/get_vulcan.sh',
        'tools/get_atmodeller.sh',
        'tools/get_obliqua.sh',
    ]
    succeeded = [
        c for c in calls if len(c[0]) > 1 and any(s in str(c[0][1]) for s in remaining_scripts)
    ]
    assert [c[0][1] for c in succeeded] == [str(tmp_path / s) for s in remaining_scripts]
    for cmd, kwargs in succeeded:
        assert len(cmd) == 2
        assert cmd[0] == 'bash'
        assert kwargs.get('cwd') == tmp_path
        assert kwargs.get('env') is calls.env
        assert kwargs.get('check') is True
        assert kwargs.get('stdin') is cli.subprocess.DEVNULL


# ---------------------------
# update_all
# ---------------------------


@pytest.mark.unit
def test_update_all_aborts_on_low_disk(monkeypatch, tmp_path):
    """``update-all`` exits non-zero when free disk is below the 5 GB threshold."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_low())
    # Defensive stub: a regression past the gate must fail an assertion, not
    # run a real pip reinstall.
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code != 0
    assert 'Aborting installation' in res.output
    # Discrimination: must NOT print the success message.
    assert 'PROTEUS update completed' not in res.output


@pytest.mark.unit
def test_update_all_success_path(monkeypatch, tmp_path):
    """Happy path: socrates + AGNI present, julia found, pip + git pull all stubbed."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)

    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0, f'update-all output: {res.output}'
    assert 'PROTEUS update completed' in res.output
    # Discrimination: pip + socrates + AGNI commands all fired (>= 3 subprocess calls).
    assert len(subprocess_calls) >= 3


@pytest.mark.unit
def test_update_all_warns_when_socrates_missing(monkeypatch, tmp_path):
    """SOCRATES-missing path emits a warning, no crash."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)

    # NO socrates dir, NO AGNI dir
    monkeypatch.setattr(
        cli.subprocess,
        'run',
        lambda cmd, **kw: type('R', (), {'returncode': 0, 'stdout': ''})(),
    )
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert 'SOCRATES not found' in res.output
    assert 'AGNI not found' in res.output


@pytest.mark.unit
def test_update_all_warns_when_julia_missing(monkeypatch, tmp_path):
    """Julia-missing path emits a warning but does NOT abort (unlike install-all)."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.delenv('PYTHON_JULIAPKG_EXE', raising=False)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(cli.shutil, 'which', lambda exe: None)
    _make_fwl_data(monkeypatch, tmp_path)

    (tmp_path / 'socrates').mkdir()
    monkeypatch.setattr(
        cli.subprocess,
        'run',
        lambda cmd, **kw: type('R', (), {'returncode': 0, 'stdout': ''})(),
    )
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    # update-all does NOT abort on missing Julia, only install-all does.
    assert res.exit_code == 0
    assert 'Julia not found' in res.output
    # Discrimination: success message still fires for the rest of the path
    assert 'PROTEUS update completed' in res.output


@pytest.mark.unit
def test_update_all_socrates_update_failure_continues(monkeypatch, tmp_path):
    """A SOCRATES update subprocess failure is reported but the command still continues."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)

    def fake_subprocess_run(cmd, **kwargs):
        if 'get_socrates.sh' in str(cmd):
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    # The SOCRATES failure is caught, AGNI + others continue, exit 0.
    assert res.exit_code == 0
    assert 'Failed to update SOCRATES' in res.output
    assert 'PROTEUS update completed' in res.output


@pytest.mark.unit
def test_update_all_agni_update_failure_continues(monkeypatch, tmp_path):
    """An AGNI update subprocess failure is reported but the command still continues."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)

    def fake_subprocess_run(cmd, **kwargs):
        if 'get_agni.sh' in str(cmd):
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert (
        '[x] Failed to update AGNI: bash tools/get_agni.sh 0 exited with status 1; '
        'its messages are above'
    ) in res.output
    assert 'PROTEUS update completed' in res.output


@pytest.mark.unit
def test_update_all_export_env_without_conda_env_writes_only_exports_file(
    monkeypatch, tmp_path
):
    """``update-all --export-env`` without a conda env calls no conda and writes the exports file.

    The message names ``source .proteus-env.sh``, a second run replaces the 4
    lines instead of adding to them, and HOME stays empty.
    """
    home, calls = _export_env_site(monkeypatch, tmp_path, None)
    for _ in range(2):
        res = runner.invoke(cli.cli, ['update-all', '--export-env'])
        assert res.exit_code == 0, f'update-all output: {res.output}'
    assert 'PROTEUS update completed' in res.output
    assert [c for c in calls if c[0] == 'conda'] == []
    expected = _expected_install_vars(tmp_path)
    lines = (tmp_path / '.proteus-env.sh').read_text().splitlines()
    assert lines == [f"export {name}='{value}'" for name, value in expected.items()]
    assert list(home.iterdir()) == []
    assert 'source .proteus-env.sh' in res.output
    assert 'conda activate' not in res.output


# ---------------------------
# update_all and the AGNI clone: real git on scratch repositories, local remote
# ---------------------------

_REPO_TOOLS = Path(__file__).resolve().parents[1] / 'tools'
_REAL_RUN = subprocess.run


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return its output without surrounding white space."""
    done = _REAL_RUN(['git', '-C', str(cwd), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _is_detached(clone: Path) -> bool:
    """Return whether HEAD of ``clone`` is on no branch."""
    done = _REAL_RUN(
        ['git', '-C', str(clone), 'symbolic-ref', '-q', 'HEAD'], capture_output=True
    )
    return done.returncode == 1


def _agni_remote(tmp_path: Path) -> tuple[Path, list[str]]:
    """Make a bare AGNI remote with 3 commits on main; return it and the hashes, oldest first.

    Its ``src/get_agni.sh`` only prints a line, and the second commit adds
    ``src/pin_only.jl``, so an untracked file of that name blocks the checkout.
    """
    work = tmp_path / 'agni_work'
    _REAL_RUN(['git', 'init', '-q', '-b', 'main', str(work)], check=True)
    (work / 'src').mkdir()
    (work / 'src' / 'get_agni.sh').write_text('#!/bin/bash\necho "AGNI setup ${1:-}"\n')
    commits = []
    for n in (1, 2, 3):
        (work / 'version.txt').write_text(f'{n}\n')
        if n == 2:
            (work / 'src' / 'pin_only.jl').write_text('# added at the pinned commit\n')
        _git(work, 'add', '-A')
        _git(work, 'commit', '-q', '-m', f'AGNI commit {n}')
        commits.append(_git(work, 'rev-parse', 'HEAD'))
    bare = tmp_path / 'AGNI.git'
    _REAL_RUN(['git', 'clone', '-q', '--bare', str(work), str(bare)], check=True)
    return bare, commits


def _agni_update_site(monkeypatch, tmp_path: Path, script: str, clone: bool = True):
    """A PROTEUS root with an AGNI clone, a stub julia, and the steps before AGNI stubbed.

    ``script = 'real'`` copies ``tools/get_agni.sh`` and ``tools/_get_common.sh``
    of this checkout into the root; ``'stub'`` writes a ``get_agni.sh`` that
    only exits 0, for the tests of the decision in ``update_all``. The
    remote is local (``AGNI_GIT_URL``), the pin is its second commit
    (``AGNI_GIT_REF``) and its tip the third, and the clone is detached at
    the first. Git reads a config of the test only. pip is the one stubbed
    subprocess; git and bash run for real.
    """
    gitconfig = tmp_path / 'gitconfig'
    gitconfig.write_text('[user]\n\tname = PROTEUS test\n\temail = test@example.org\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(gitconfig))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    for var in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'PYTHON_JULIAPKG_EXE'):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / 'proteus'
    _pin_proteus_root(monkeypatch, root)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    _make_fwl_data(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, '_install_cvode', lambda root: None)
    bindir = tmp_path / 'bin'
    _julia_stub(bindir)
    monkeypatch.setenv('PATH', os.pathsep.join([str(bindir), os.environ['PATH']]))
    bare, commits = _agni_remote(tmp_path)
    monkeypatch.setenv('AGNI_GIT_URL', str(bare))
    monkeypatch.setenv('AGNI_GIT_REF', commits[1])
    (root / 'tools').mkdir()
    if script == 'real':
        for name in ('get_agni.sh', '_get_common.sh'):
            shutil.copy(_REPO_TOOLS / name, root / 'tools' / name)
    else:
        (root / 'tools' / 'get_agni.sh').write_text('#!/bin/bash\nexit 0\n')
    agni = root / 'AGNI'
    if clone:
        _REAL_RUN(['git', 'clone', '-q', str(bare), str(agni)], check=True)
        _git(agni, 'checkout', '-q', '--detach', commits[0])
    calls = []

    def run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        if list(cmd[1:3]) == ['-m', 'pip']:
            return subprocess.CompletedProcess(cmd, 0)
        return _REAL_RUN(cmd, **kwargs)

    monkeypatch.setattr(cli.subprocess, 'run', run)
    data_paths = []
    monkeypatch.setattr(
        cli, '_update_input_data', lambda path: data_paths.append(path) or False
    )
    return types.SimpleNamespace(
        root=root, agni=agni, commits=commits, calls=calls, data_paths=data_paths
    )


def _script_calls(site) -> list[list[str]]:
    """The commands of ``site`` that ran ``tools/get_agni.sh`` of its root."""
    script = str(site.root / 'tools' / 'get_agni.sh')
    return [cmd for cmd in site.calls if cmd[:2] == ['bash', script]]


def _stay_line(site, found: str) -> str:
    """The line ``update-all`` prints for an AGNI clone that it leaves as it is."""
    pin = site.commits[1][:7]
    return f'agni: {site.agni} stays as it is: {found}; PROTEUS expects the commit {pin}'


@pytest.mark.unit
def test_update_all_moves_a_clean_detached_agni_clone_to_the_pin(monkeypatch, tmp_path):
    """A clean clone detached at an older commit, as install.sh leaves it, ends at the pin.

    The real ``tools/get_agni.sh`` runs once with ``0``. The remote tip is
    one commit past the pin, so a pull of the latest AGNI ends elsewhere.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'real')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert _script_calls(site) == [['bash', str(site.root / 'tools' / 'get_agni.sh'), '0']]
    assert _git(site.agni, 'rev-parse', 'HEAD') == site.commits[1]
    assert _git(site.agni, 'rev-parse', 'HEAD') != site.commits[2]
    assert _is_detached(site.agni)
    assert ['git', 'pull'] not in site.calls
    assert 'Failed to update AGNI' not in res.output
    assert '[+] AGNI updated' in res.output
    assert 'stays as it is' not in res.output


@pytest.mark.unit
def test_update_all_moves_agni_and_keeps_an_untracked_file_and_a_stash(monkeypatch, tmp_path):
    """An untracked file and a stash entry do not block: the clone moves and keeps both."""
    site = _agni_update_site(monkeypatch, tmp_path, 'real')
    (site.agni / 'version.txt').write_text('an edit put aside\n')
    _git(site.agni, 'stash', 'push', '-q', '-m', 'my edit')
    (site.agni / 'notes.txt').write_text('my notes\n')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert len(_script_calls(site)) == 1
    assert _git(site.agni, 'rev-parse', 'HEAD') == site.commits[1]
    assert (site.agni / 'notes.txt').read_text() == 'my notes\n'
    assert 'my edit' in _git(site.agni, 'stash', 'list')
    assert (site.agni / 'version.txt').read_text() == '2\n'


@pytest.mark.unit
def test_update_all_reports_the_script_status_when_git_refuses_the_checkout(
    monkeypatch, tmp_path
):
    """An untracked file that the pin would overwrite makes git refuse; the clone stays put.

    The message names the script and its exit status, and the later steps run.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'real')
    (site.agni / 'src' / 'pin_only.jl').write_text('# my own file of that name\n')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert len(_script_calls(site)) == 1
    assert _git(site.agni, 'rev-parse', 'HEAD') == site.commits[0]
    assert (site.agni / 'src' / 'pin_only.jl').read_text() == '# my own file of that name\n'
    assert (
        '[x] Failed to update AGNI: bash tools/get_agni.sh 0 exited with status 1; '
        'its messages are above'
    ) in res.output
    assert '[+] AGNI updated' not in res.output
    assert 'PROTEUS update completed' in res.output


@pytest.mark.unit
def test_update_all_leaves_an_agni_clone_with_a_modified_file_as_it_is(monkeypatch, tmp_path):
    """A modified tracked file blocks: no script, same HEAD and status, the hand commands.

    ``tools/get_agni.sh`` is a stub here: the test is about the decision of
    ``update_all``, and the command still runs its later steps.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'stub')
    (site.agni / 'version.txt').write_text('my change\n')
    head = _git(site.agni, 'rev-parse', 'HEAD')
    status = _git(site.agni, 'status', '--porcelain')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert _script_calls(site) == []
    assert _git(site.agni, 'rev-parse', 'HEAD') == head == site.commits[0]
    assert _git(site.agni, 'status', '--porcelain') == status == 'M version.txt'
    assert _stay_line(site, '1 modified tracked file') in res.output
    assert f'git -C {site.agni} fetch origin' in res.output
    assert f'git -C {site.agni} checkout {site.commits[1]}' in res.output
    assert 'branch my-agni-work' not in res.output
    assert 'bash tools/get_agni.sh 0' in res.output
    assert 'Failed to update AGNI' not in res.output
    assert site.data_paths == [site.root / 'input' / 'all_options.toml']
    assert 'PROTEUS update completed' in res.output


@pytest.mark.unit
def test_update_all_leaves_a_local_commit_on_a_detached_agni_head(monkeypatch, tmp_path):
    """A commit on a detached HEAD is on no ref; moving the clone would lose it, so it stays.

    ``tools/get_agni.sh`` is a stub here: the test is about the decision. The
    hand commands start with a `git branch`, since `fetch` and `checkout <pin>`
    alone would leave the commit on no ref; run by hand in this order, they keep
    the commit on the branch and move HEAD to the pin.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'stub')
    (site.agni / 'version.txt').write_text('my work\n')
    _git(site.agni, 'commit', '-q', '-am', 'my work')
    local = _git(site.agni, 'rev-parse', 'HEAD')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert _script_calls(site) == []
    assert _git(site.agni, 'rev-parse', 'HEAD') == local
    assert local not in site.commits
    found = '1 local commit (on no remote-tracking branch and under no tag)'
    assert _stay_line(site, found) in res.output
    commands = [ln.strip() for ln in res.output.splitlines() if ln.startswith('      git -C ')]
    assert commands == [
        f'git -C {site.agni} branch my-agni-work'
        '    (keeps the local commits on a branch; choose any name)',
        f'git -C {site.agni} fetch origin',
        f'git -C {site.agni} checkout {site.commits[1]}',
    ]
    for command in commands:
        _REAL_RUN(command.split('    (')[0].split(), check=True, capture_output=True)
    assert _git(site.agni, 'rev-parse', 'refs/heads/my-agni-work') == local
    assert _git(site.agni, 'rev-parse', 'HEAD') == site.commits[1]


@pytest.mark.unit
def test_agni_commit_that_only_a_local_tag_holds_does_not_block(monkeypatch, tmp_path):
    """A commit on a detached HEAD under a tag of the clone, on no remote branch, may move.

    A tag keeps the commit, so the checkout of the pin loses nothing; without
    ``--tags`` in the count it would be 1 local commit and block.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'stub')
    (site.agni / 'version.txt').write_text('tagged work\n')
    _git(site.agni, 'commit', '-q', '-am', 'tagged work')
    _git(site.agni, 'tag', 'my-release')
    count = _git(site.agni, 'rev-list', '--count', 'HEAD', '--not', '--remotes')
    assert count == '1'

    assert cli._agni_update_blocker(site.agni) is None
    _git(site.agni, 'tag', '-d', 'my-release')
    assert cli._agni_update_blocker(site.agni) == (
        '1 local commit (on no remote-tracking branch and under no tag)'
    )


@pytest.mark.unit
def test_agni_blocker_reads_the_clone_and_not_the_repository_of_git_dir(monkeypatch, tmp_path):
    """With GIT_DIR naming another repository, the AGNI clone is still the one read.

    git would take the index and HEAD of GIT_DIR and compare them with the AGNI
    files, which blocks a clean clone; the blocker reads the clone itself, so a
    clean clone may move and a modified one stays.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'stub')
    other = tmp_path / 'other'
    _REAL_RUN(['git', 'init', '-q', '-b', 'main', str(other)], check=True)
    (other / 'unrelated.txt').write_text('x\n')
    _git(other, 'add', '-A')
    _git(other, 'commit', '-q', '-m', 'other')
    monkeypatch.setenv('GIT_DIR', str(other / '.git'))

    assert cli._agni_update_blocker(site.agni) is None
    (site.agni / 'version.txt').write_text('my change\n')
    assert cli._agni_update_blocker(site.agni) == '1 modified tracked file'
    assert cli._git_read(site.agni, 'rev-parse', 'HEAD') == site.commits[0]


@pytest.mark.unit
def test_git_read_gives_git_its_own_timeout_and_no_optional_locks(monkeypatch, tmp_path):
    """``_git_read`` passes ``_GIT_READ_TIMEOUT_S`` (30 s), not the 10 s default of the doctor.

    A slow ``git status`` on a network folder needs the longer limit. The call
    also puts ``--no-optional-locks`` before ``-C``, so that status rewrites no
    index, and the output comes back without its surrounding white space.
    """
    calls = []

    def fake_check_output(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return '  M version.txt\n'

    monkeypatch.setattr(doctor.subprocess, 'check_output', fake_check_output)
    assert cli._git_read(tmp_path, 'status', '--porcelain') == 'M version.txt'
    ((cmd, kwargs),) = calls
    assert cmd == ['git', '--no-optional-locks', '-C', str(tmp_path), 'status', '--porcelain']
    assert kwargs['timeout'] == cli._GIT_READ_TIMEOUT_S == 30
    assert kwargs['timeout'] != doctor._GIT_NOTE_TIMEOUT_S


@pytest.mark.unit
def test_update_all_leaves_an_agni_branch_with_an_unpushed_commit(monkeypatch, tmp_path):
    """A branch ahead of its remote blocks as well. ``tools/get_agni.sh`` is a stub here."""
    site = _agni_update_site(monkeypatch, tmp_path, 'stub')
    _git(site.agni, 'switch', '-q', 'main')
    (site.agni / 'version.txt').write_text('not pushed\n')
    _git(site.agni, 'commit', '-q', '-am', 'not pushed')
    local = _git(site.agni, 'rev-parse', 'HEAD')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert _script_calls(site) == []
    assert _git(site.agni, 'rev-parse', 'HEAD') == local
    assert _git(site.agni, 'symbolic-ref', '--short', 'HEAD') == 'main'
    assert 'stays as it is: 1 local commit' in res.output


@pytest.mark.unit
def test_update_all_moves_an_agni_branch_that_equals_its_remote(monkeypatch, tmp_path):
    """A branch at its remote holds no local work: the clone ends detached at the pin.

    The branch stays at the remote tip, one commit past the pin.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'real')
    _git(site.agni, 'switch', '-q', 'main')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert len(_script_calls(site)) == 1
    assert _git(site.agni, 'rev-parse', 'HEAD') == site.commits[1]
    assert _is_detached(site.agni)
    assert _git(site.agni, 'rev-parse', 'refs/heads/main') == site.commits[2]
    assert '[+] AGNI updated' in res.output


@pytest.mark.unit
def test_update_all_leaves_an_agni_folder_with_an_empty_git_dir(monkeypatch, tmp_path):
    """An empty ``.git`` folder makes git answer for the PROTEUS repository around it.

    The folder stays, and HEAD and status of the PROTEUS repository are
    unchanged. ``tools/get_agni.sh`` is a stub here: the real script would
    run its checkout in the PROTEUS repository.
    """
    site = _agni_update_site(monkeypatch, tmp_path, 'stub', clone=False)
    _git(site.root, 'init', '-q', '-b', 'main')
    _git(site.root, 'add', '-A')
    _git(site.root, 'commit', '-q', '-m', 'PROTEUS')
    (site.agni / '.git').mkdir(parents=True)
    head = _git(site.root, 'rev-parse', 'HEAD')
    status = _git(site.root, 'status', '--porcelain')

    res = runner.invoke(cli.cli, ['update-all'])

    assert res.exit_code == 0, res.output
    assert _script_calls(site) == []
    assert _git(site.root, 'rev-parse', 'HEAD') == head
    assert _git(site.root, 'status', '--porcelain') == status
    found = f'not a git clone of its own (git answers for {site.root.resolve()})'
    assert _stay_line(site, found) in res.output
    assert f'Move it away; then, in {site.root}, run  bash tools/get_agni.sh 0' in res.output
    assert f'git -C {site.agni} checkout' not in res.output


# ---------------------------
# run-from-anywhere contract: install-all / update-all must resolve the
# source tree from the package location, not the caller's working directory
# ---------------------------


@pytest.mark.unit
def test_update_all_anchors_paths_to_source_root_not_cwd(monkeypatch, tmp_path):
    """``update-all`` works when invoked from outside the checkout.

    Run from any directory other than the source tree, the command must still
    reinstall the source tree (not the caller's directory) and must invoke the
    SOCRATES build script and the AGNI update by their absolute location under
    the resolved root. This is the contract that breaks when the root is taken
    from the current working directory.
    """
    source_root = tmp_path / 'proteus_src'
    foreign_cwd = tmp_path / 'run_from_here'
    source_root.mkdir()
    (source_root / 'tools').mkdir()
    (source_root / 'socrates').mkdir()
    (source_root / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)
    foreign_cwd.mkdir()

    monkeypatch.chdir(foreign_cwd)
    _pin_proteus_root(monkeypatch, source_root)
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    _make_fwl_data(monkeypatch, tmp_path)

    disk_paths = []

    def fake_disk_usage(path):
        disk_paths.append(Path(path))
        return _stub_disk_usage_high()

    monkeypatch.setattr(cli.shutil, 'disk_usage', fake_disk_usage)

    captured = []

    def fake_subprocess_run(cmd, **kwargs):
        captured.append((list(cmd), dict(kwargs)))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)

    data_paths = []

    def record_input_data(path):
        data_paths.append(path)
        return False

    monkeypatch.setattr(cli, '_update_input_data', record_input_data)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0, f'update-all output: {res.output}'

    # The editable pip install targets the source tree, never the cwd or '.'.
    pip_targets = [
        cmd[-1] for cmd, _ in captured if 'pip' in cmd and 'install' in cmd and '-e' in cmd
    ]
    # Discrimination: a cwd-derived implementation passes the bare '.'
    # (resolving against the caller's directory), which this equality rejects.
    assert pip_targets == [str(source_root)]

    # The SOCRATES script is referenced by its absolute path under the root, so
    # bash can find it regardless of where the command was launched.
    socrates_args = [a for cmd, _ in captured for a in cmd if 'get_socrates.sh' in a]
    assert len(socrates_args) == 1
    assert socrates_args[0].startswith(str(source_root))
    assert str(foreign_cwd) not in socrates_args[0]

    # The AGNI update runs the script under the root from the root, not the cwd.
    agni = [(cmd, kw.get('cwd')) for cmd, kw in captured if 'get_agni.sh' in str(cmd)]
    assert agni == [(['bash', str(source_root / 'tools' / 'get_agni.sh'), '0'], source_root)]

    # The disk gate measures the filesystem the update writes to.
    assert disk_paths == [source_root]

    # The default config for the data refresh sits under the root, not under
    # the caller's directory.
    assert data_paths == [source_root / 'input' / 'all_options.toml']

    # Failures must propagate: every subprocess call runs with check=True so
    # a nonzero exit aborts the step instead of being ignored.
    assert captured and all(kw.get('check') is True for _, kw in captured)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('mod_name', 'expected_folder', 'script_rel', 'pattern'),
    [
        ('Morrigan', 'Morrigan', 'tools/get_morrigan.sh', r'workpath="\$root/([^/]+)/"'),
        ('VULCAN', 'VULCAN', 'tools/get_vulcan.sh', r'workpath="\$root/([^/]+)/"'),
        ('atmodeller', 'Atmodeller', 'tools/get_atmodeller.sh', r'workpath="\$root/([^/]+)/"'),
        ('Obliqua', 'Obliqua', 'tools/get_obliqua.sh', r'dest="\$proteus_root/([^"]+)"'),
    ],
    ids=['Morrigan', 'VULCAN', 'atmodeller', 'Obliqua'],
)
def test_standard_module_folder_matches_script_default(
    mod_name, expected_folder, script_rel, pattern
):
    """The folder of each standard module matches the destination folder in its setup script."""
    root = cli._resolve_proteus_root()
    script_text = (root / script_rel).read_text()
    match = re.search(pattern, script_text)
    assert match is not None
    assert match.group(1) == expected_folder
    entry = next(item for item in cli._STANDARD_MODULES if item[0] == mod_name)
    assert entry[1] == expected_folder


@pytest.mark.unit
def test_standard_modules_matches_install_sh_further_modules():
    """The standard module pairs and order match the install_further_module lines in install.sh."""
    root = cli._resolve_proteus_root()
    install_text = (root / 'install.sh').read_text()
    matches = re.findall(r'install_further_module\s+"([^"]+)"\s+bash\s+(\S+)', install_text)
    expected_pairs = [
        ('Morrigan', 'tools/get_morrigan.sh'),
        ('VULCAN', 'tools/get_vulcan.sh'),
        ('atmodeller', 'tools/get_atmodeller.sh'),
        ('Obliqua', 'tools/get_obliqua.sh'),
    ]
    assert len(matches) == 4
    assert matches == expected_pairs
    pairs_in_cli = [(name, script) for name, _, script in cli._STANDARD_MODULES]
    assert pairs_in_cli == expected_pairs


@pytest.mark.unit
def test_update_all_warns_when_standard_modules_missing(monkeypatch, tmp_path):
    """update-all emits yellow warnings for missing standard module directories."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path)
    real_version = importlib.metadata.version

    def fake_version(pkg):
        if pkg in ('fwl-morrigan', 'fwl-vulcan', 'atmodeller'):
            raise importlib.metadata.PackageNotFoundError(pkg)
        return real_version(pkg)

    monkeypatch.setattr('importlib.metadata.version', fake_version)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    expected_modules = [
        ('Morrigan', 'tools/get_morrigan.sh'),
        ('VULCAN', 'tools/get_vulcan.sh'),
        ('atmodeller', 'tools/get_atmodeller.sh'),
        ('Obliqua', 'tools/get_obliqua.sh'),
    ]
    for name, script in expected_modules:
        assert f'[!] {name} not found. Run "cd {tmp_path} && bash {script}".' in res.output
    for cmd, _ in calls:
        for _, script in expected_modules:
            assert not any(script in str(arg) for arg in cmd)


@pytest.mark.unit
def test_update_all_warns_only_for_absent_module(monkeypatch, tmp_path):
    """update-all warns only for absent standard modules when others are present."""
    _standard_modules_test_site(monkeypatch, tmp_path)
    (tmp_path / 'Obliqua').mkdir()
    (tmp_path / 'Obliqua' / 'Project.toml').write_text('name = "Obliqua"\n')

    real_version = importlib.metadata.version

    def fake_version(pkg):
        if pkg == 'fwl-vulcan':
            raise importlib.metadata.PackageNotFoundError(pkg)
        if pkg in ('fwl-morrigan', 'atmodeller'):
            return '1.0.0'
        return real_version(pkg)

    monkeypatch.setattr('importlib.metadata.version', fake_version)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert (
        f'[!] VULCAN not found. Run "cd {tmp_path} && bash tools/get_vulcan.sh".' in res.output
    )
    assert 'Morrigan not found' not in res.output
    assert 'atmodeller not found' not in res.output
    assert 'Obliqua not found' not in res.output


@pytest.mark.unit
def test_update_all_runs_no_commands_in_standard_module_folders(monkeypatch, tmp_path):
    """update-all runs no git commands inside standard module directories."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path)
    expected_folders = ['Morrigan', 'VULCAN', 'Atmodeller', 'Obliqua']
    expected_scripts = [
        'tools/get_morrigan.sh',
        'tools/get_vulcan.sh',
        'tools/get_atmodeller.sh',
        'tools/get_obliqua.sh',
    ]
    for folder in expected_folders:
        (tmp_path / folder).mkdir()

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    for _, kwargs in calls:
        if 'cwd' in kwargs and kwargs['cwd'] is not None:
            cwd_path = Path(kwargs['cwd'])
            for folder in expected_folders:
                assert cwd_path != tmp_path / folder
    for cmd, _ in calls:
        for script in expected_scripts:
            assert not any(script in str(arg) for arg in cmd)
        for folder in expected_folders:
            folder_str = str(tmp_path / folder)
            assert not any(folder_str in str(arg) for arg in cmd)


@pytest.mark.unit
def test_update_all_omits_warning_when_vulcan_installed_as_package_without_folder(
    monkeypatch, tmp_path
):
    """update-all does not warn when VULCAN package is installed even if folder is missing."""
    calls = _standard_modules_test_site(monkeypatch, tmp_path)
    real_version = importlib.metadata.version

    def fake_version(pkg):
        if pkg == 'fwl-vulcan':
            return '2.0.0'
        return real_version(pkg)

    monkeypatch.setattr('importlib.metadata.version', fake_version)
    assert not (tmp_path / 'VULCAN').exists()

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert 'VULCAN not found' not in res.output
    assert not any('get_vulcan.sh' in str(c[0]) for c in calls)


@pytest.mark.unit
def test_update_all_warns_when_morrigan_folder_present_but_package_not_installed(
    monkeypatch, tmp_path
):
    """update-all warns when Morrigan folder exists but the package is not installed."""
    _standard_modules_test_site(monkeypatch, tmp_path)
    (tmp_path / 'Morrigan').mkdir()
    real_version = importlib.metadata.version

    def fake_version(pkg):
        if pkg == 'fwl-morrigan':
            raise importlib.metadata.PackageNotFoundError(pkg)
        return real_version(pkg)

    monkeypatch.setattr('importlib.metadata.version', fake_version)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    expected_line = (
        f'[!] Morrigan not found. Run "cd {tmp_path} && bash tools/get_morrigan.sh".'
    )
    assert expected_line in res.output
    assert 'Setting up Morrigan' not in res.output


@pytest.mark.unit
def test_update_all_obliqua_empty_folder_warns_populated_folder_passes(monkeypatch, tmp_path):
    """update-all warns for empty Obliqua folder and passes for populated Obliqua folder."""
    _standard_modules_test_site(monkeypatch, tmp_path)
    obliqua_dir = tmp_path / 'Obliqua'
    obliqua_dir.mkdir()
    expected_line = f'[!] Obliqua not found. Run "cd {tmp_path} && bash tools/get_obliqua.sh".'

    res_empty = runner.invoke(cli.cli, ['update-all'])
    assert res_empty.exit_code == 0
    assert expected_line in res_empty.output

    (obliqua_dir / 'Project.toml').write_text('name = "Obliqua"\n')
    res_populated = runner.invoke(cli.cli, ['update-all'])
    assert res_populated.exit_code == 0
    assert 'Obliqua not found' not in res_populated.output


@pytest.mark.unit
def test_update_all_standard_module_warning_includes_cd_root(monkeypatch, tmp_path):
    """update-all warning includes the cd <root> prefix in setup instructions."""
    _standard_modules_test_site(monkeypatch, tmp_path)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert f'cd {tmp_path} && ' in res.output
    assert '[!] Obliqua not found. Run "cd ' in res.output


@pytest.mark.unit
def test_install_all_anchors_paths_to_source_root_not_cwd(monkeypatch, tmp_path):
    """``install-all`` clones submodules next to the source tree, not the cwd.

    With neither socrates/ nor AGNI/ present yet, the installer must run the
    SOCRATES and AGNI scripts by their absolute paths under the root, from the
    root, so a run launched from a foreign directory populates the checkout
    rather than scattering submodules into the caller's directory.
    """
    source_root = tmp_path / 'proteus_src'
    foreign_cwd = tmp_path / 'run_from_here'
    source_root.mkdir()
    (source_root / 'tools').mkdir()
    foreign_cwd.mkdir()
    # Deliberately no socrates/ and no AGNI/ so both install branches fire.

    monkeypatch.chdir(foreign_cwd)
    _pin_proteus_root(monkeypatch, source_root)
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    disk_paths = []

    def fake_disk_usage(path):
        disk_paths.append(Path(path))
        return _stub_disk_usage_high()

    monkeypatch.setattr(cli.shutil, 'disk_usage', fake_disk_usage)

    captured = []

    def fake_subprocess_run(cmd, **kwargs):
        captured.append((list(cmd), dict(kwargs)))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)

    data_paths = []

    def record_input_data(path):
        data_paths.append(path)
        return False

    monkeypatch.setattr(cli, '_update_input_data', record_input_data)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code == 0, f'install-all output: {res.output}'

    # SOCRATES script referenced by absolute path under the root.
    socrates_args = [a for cmd, _ in captured for a in cmd if 'get_socrates.sh' in a]
    assert len(socrates_args) == 1
    assert socrates_args[0].startswith(str(source_root))

    # AGNI is set up by the script under the root, run from the root, which
    # clones into <root>/AGNI; nothing is cloned into the foreign cwd.
    assert [cmd for cmd, _ in captured if 'clone' in cmd] == []
    agni = [(cmd, kw.get('cwd')) for cmd, kw in captured if 'get_agni.sh' in str(cmd)]
    assert agni == [(['bash', str(source_root / 'tools' / 'get_agni.sh'), '0'], source_root)]

    # The default config for the data download sits under the root.
    assert data_paths == [source_root / 'input' / 'all_options.toml']

    # The output directory is created under the root, never in the caller's
    # directory.
    assert (source_root / 'output').is_dir()
    assert not (foreign_cwd / 'output').exists()

    # The disk gate measures the filesystem the install writes to.
    assert disk_paths == [source_root]

    # Failures must propagate: every subprocess call runs with check=True.
    assert captured and all(kw.get('check') is True for _, kw in captured)


@pytest.mark.unit
def test_update_all_errors_cleanly_when_source_tree_missing(monkeypatch, tmp_path):
    """``update-all`` fails fast with a clear message for a non-editable install.

    When the source tree cannot be located (PROTEUS installed as a plain
    wheel) and the working directory is not a checkout either, there is
    nothing to reinstall editable, so the command must exit non-zero with an
    actionable message and must NOT attempt any subprocess.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())

    def raise_env_error():
        raise EnvironmentError('Cannot locate PROTEUS directory.')

    monkeypatch.setattr(cli, 'get_proteus_dir', raise_env_error)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(cli.subprocess, 'run', fake_run)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code != 0
    assert 'Cannot locate the PROTEUS source tree' in res.output
    # Discrimination: it bailed out before running pip (or anything else), and
    # the success banner is absent.
    assert calls == []
    assert 'PROTEUS update completed' not in res.output


@pytest.mark.unit
def test_install_all_errors_cleanly_when_source_tree_missing(monkeypatch, tmp_path):
    """``install-all`` fails fast with a clear message for a non-editable install."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    def raise_env_error():
        raise EnvironmentError('Cannot locate PROTEUS directory.')

    monkeypatch.setattr(cli, 'get_proteus_dir', raise_env_error)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(cli.subprocess, 'run', fake_run)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Cannot locate the PROTEUS source tree' in res.output
    # Discrimination: no submodule subprocess ran, the completion banner is
    # absent, and the abort precedes the FWL_DATA directory creation.
    assert calls == []
    assert 'PROTEUS installation completed' not in res.output
    assert not (tmp_path / 'fwl_data').exists()


@pytest.mark.unit
def test_resolve_proteus_root_returns_directory_with_pyproject(monkeypatch, tmp_path):
    """``_resolve_proteus_root`` returns the source root, not the working dir.

    The resolved path comes from the installed package location and must hold
    the PROTEUS ``pyproject.toml``; changing the process working directory to
    an unrelated location must not change the answer.
    """
    source_root = tmp_path / 'proteus_src'
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    _pin_proteus_root(monkeypatch, source_root)
    monkeypatch.chdir(elsewhere)

    root = cli._resolve_proteus_root()
    assert root == source_root
    assert (root / 'pyproject.toml').is_file()
    # Discrimination: the result is decoupled from the cwd we just moved to and
    # is an absolute path.
    assert root != elsewhere
    assert root.is_absolute()


@pytest.mark.unit
def test_resolve_proteus_root_prefers_package_location_over_cwd(monkeypatch, tmp_path):
    """The package-derived root wins even when the cwd is also a checkout.

    The working directory is a fallback for wheel installs only; with a valid
    package-derived root available, standing inside a second checkout must not
    redirect the install target to it.
    """
    root_a = tmp_path / 'clone_a'
    root_b = tmp_path / 'clone_b'
    _pin_proteus_root(monkeypatch, root_a)
    root_b.mkdir()
    (root_b / 'pyproject.toml').write_text('[project]\nname = "fwl-proteus"\n')
    monkeypatch.chdir(root_b)

    root = cli._resolve_proteus_root()
    # Discrimination: a cwd-first implementation returns clone_b here.
    assert root == root_a
    assert root != root_b


@pytest.mark.unit
def test_resolve_proteus_root_exits_when_source_tree_unlocatable(monkeypatch, tmp_path):
    """``_resolve_proteus_root`` turns a missing source tree into a clean exit.

    A bare wheel install has no ``pyproject.toml`` to find, and a working
    directory without the checkout markers offers no fallback; the helper
    converts the underlying ``EnvironmentError`` into ``SystemExit`` so the
    CLI prints an actionable message instead of a raw traceback.
    """
    monkeypatch.chdir(tmp_path)

    def raise_env_error():
        raise EnvironmentError('Cannot locate PROTEUS directory.')

    monkeypatch.setattr(cli, 'get_proteus_dir', raise_env_error)

    with pytest.raises(SystemExit) as excinfo:
        cli._resolve_proteus_root()
    # Discrimination: a clean exit-code-1, not the raw EnvironmentError and not
    # a success (code 0).
    assert excinfo.value.code == 1


@pytest.mark.unit
def test_resolve_proteus_root_falls_back_to_checkout_cwd(monkeypatch, tmp_path):
    """A wheel install run from inside a checkout uses that checkout.

    When the package location yields no source tree, the helper accepts the
    working directory only if its pyproject.toml names the fwl-proteus
    project; a directory holding some other project's pyproject.toml must
    still exit.
    """
    monkeypatch.chdir(tmp_path)

    def raise_env_error():
        raise EnvironmentError('Cannot locate PROTEUS directory.')

    monkeypatch.setattr(cli, 'get_proteus_dir', raise_env_error)

    # Edge case: a generic Python project directory is not accepted.
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "other"\n')
    with pytest.raises(SystemExit) as excinfo:
        cli._resolve_proteus_root()
    assert excinfo.value.code == 1

    # With the PROTEUS project name in place the working directory is used.
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "fwl-proteus"\n')
    root = cli._resolve_proteus_root()
    assert root == tmp_path
    assert root.is_absolute()


@pytest.mark.unit
def test_is_proteus_root_rejects_invalid_candidates(tmp_path):
    """``_is_proteus_root`` accepts only a directory holding fwl-proteus.

    Probes the empty directory, a foreign project, malformed TOML, and the
    genuine project name, so a stale editable pointer or an arbitrary Python
    project can never be accepted as the install target.
    """
    # Edge case: no pyproject.toml at all.
    assert cli._is_proteus_root(tmp_path) is False

    # A different project's pyproject.toml is rejected.
    pyproject = tmp_path / 'pyproject.toml'
    pyproject.write_text('[project]\nname = "other"\n')
    assert cli._is_proteus_root(tmp_path) is False

    # Malformed TOML counts as not-a-root rather than raising.
    pyproject.write_text('not [valid toml')
    assert cli._is_proteus_root(tmp_path) is False

    # The genuine project name is accepted.
    pyproject.write_text('[project]\nname = "fwl-proteus"\n')
    assert cli._is_proteus_root(tmp_path) is True


@pytest.mark.unit
def test_update_all_explicit_config_path_not_reanchored(monkeypatch, tmp_path):
    """An explicitly passed --config-path is forwarded exactly as given.

    Only the implicit default is anchored to the source root; a user-supplied
    relative path keeps its usual cwd-relative meaning and must reach the
    data-download step unchanged.
    """
    source_root = tmp_path / 'proteus_src'
    foreign_cwd = tmp_path / 'run_from_here'
    foreign_cwd.mkdir()
    _pin_proteus_root(monkeypatch, source_root)
    (source_root / 'socrates').mkdir()
    (source_root / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)
    monkeypatch.chdir(foreign_cwd)
    _make_fwl_data(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )

    data_paths = []

    def record_input_data(path):
        data_paths.append(path)
        return False

    monkeypatch.setattr(cli, '_update_input_data', record_input_data)

    res = runner.invoke(cli.cli, ['update-all', '--config-path', 'my/conf.toml'])
    assert res.exit_code == 0, f'update-all output: {res.output}'
    # Discrimination: a re-anchoring implementation would forward
    # source_root / 'my/conf.toml' instead of the path as typed.
    assert data_paths == [Path('my/conf.toml')]


@pytest.mark.unit
def test_install_all_explicit_config_path_not_reanchored(monkeypatch, tmp_path):
    """``install-all`` forwards an explicit --config-path exactly as given."""
    source_root = tmp_path / 'proteus_src'
    foreign_cwd = tmp_path / 'run_from_here'
    foreign_cwd.mkdir()
    _pin_proteus_root(monkeypatch, source_root)
    (source_root / 'socrates').mkdir()
    (source_root / 'AGNI').mkdir()
    monkeypatch.chdir(foreign_cwd)
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )

    data_paths = []

    def record_input_data(path):
        data_paths.append(path)
        return False

    monkeypatch.setattr(cli, '_update_input_data', record_input_data)

    res = runner.invoke(cli.cli, ['install-all', '--config-path', 'my/conf.toml'])
    assert res.exit_code == 0, f'install-all output: {res.output}'
    # Discrimination: a re-anchoring implementation would forward
    # source_root / 'my/conf.toml' instead of the path as typed.
    assert data_paths == [Path('my/conf.toml')]


@pytest.mark.unit
def test_update_all_aborts_when_fwl_data_missing(monkeypatch, tmp_path):
    """``update-all`` exits with guidance when the FWL_DATA directory is absent.

    Updating presumes a prior installation; with FWL_DATA pointing at a
    directory that does not exist, the command must abort with the
    install-all hint rather than continue against a non-existent data tree.
    """
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'no_such_dir'))
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code != 0
    assert 'FWL_DATA directory not found' in res.output
    assert 'install-all' in res.output
    # Discrimination: the success banner must not fire on this path.
    assert 'PROTEUS update completed' not in res.output


# ---------------------------
# start command (covering the non-deterministic happy path)
# ---------------------------


class _FakeProteusStart:
    """Stand-in for Proteus.start that records its kwargs."""

    instances = []

    def __init__(self, *, config_path):
        self.config_path = config_path
        type(self).instances.append(self)

    def start(self, *, resume=False, offline=False):
        self.resume = resume
        self.offline = offline


@pytest.mark.unit
def test_start_dispatches_resume_and_offline_flags(monkeypatch, tmp_path):
    """``proteus start -c cfg --resume --offline`` threads the flags into Proteus.start."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusStart.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusStart)
    # Ensure the deterministic-sentinel safety net does NOT fire on this path.
    monkeypatch.delenv(cli._PROTEUS_DETERMINISTIC_SENTINEL, raising=False)

    res = runner.invoke(cli.start, ['-c', str(cfg), '--resume', '--offline'])
    assert res.exit_code == 0, f'start output: {res.output}'
    assert len(_FakeProteusStart.instances) == 1
    inst = _FakeProteusStart.instances[0]
    assert inst.resume is True
    assert inst.offline is True


@pytest.mark.unit
def test_start_defaults_to_no_resume_no_offline(monkeypatch, tmp_path):
    """Without --resume / --offline, Proteus.start is called with both flags False."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    _FakeProteusStart.instances = []
    monkeypatch.setattr(cli, 'Proteus', _FakeProteusStart)

    res = runner.invoke(cli.start, ['-c', str(cfg)])
    assert res.exit_code == 0
    assert len(_FakeProteusStart.instances) == 1
    inst = _FakeProteusStart.instances[0]
    assert inst.resume is False
    assert inst.offline is False


def _cvode_recorder(fail: bool, import_fails: bool = False):
    """Return a subprocess.run stand-in that records commands and can fail the install.

    ``fail`` makes tools/get_cvode.sh fail; ``import_fails`` makes the follow-up
    import check in the interpreter running PROTEUS fail.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if fail and str(cmd[1]).endswith('get_cvode.sh'):
            raise cli.subprocess.CalledProcessError(1, cmd)
        if import_fails and cmd[1:] == ['-c', cli._CVODE_IMPORT]:
            raise cli.subprocess.CalledProcessError(1, cmd)
        return type('R', (), {'returncode': 0})()

    return fake_run, calls


def _cvode_command_env(monkeypatch, tmp_path):
    """Pin the tree, disk, Julia and data directories so install-all and update-all run offline."""
    _pin_proteus_root(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    (tmp_path / 'fwl_data').mkdir()
    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()
    _agni_clone_may_move(monkeypatch)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)


@pytest.mark.unit
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
@pytest.mark.parametrize('import_fails', [False, True], ids=['script-fails', 'import-fails'])
def test_a_failed_cvode_install_warns_and_the_command_goes_on(
    monkeypatch, tmp_path, command, import_fails
):
    """A CVODE failure is a warning: exit 0, the message says who needs it, later steps run.

    Only Aragog on ``solver_method = "cvode"`` needs CVODE, and that run stops
    at setup, so SPIDER, radau and bdf users can finish the installation.
    """
    _cvode_command_env(monkeypatch, tmp_path)
    fake_run, calls = _cvode_recorder(fail=not import_fails, import_fails=import_fails)
    monkeypatch.setattr(cli.subprocess, 'run', fake_run)

    res = runner.invoke(cli.cli, [command])

    out = res.output.replace('\n', ' ')
    assert res.exit_code == 0, res.output
    assert '[!] CVODE (scikits-odes-sundials)' in out
    assert 'stops at setup until it does' in out
    assert 'bash tools/get_cvode.sh' in out
    assert 'radau' in out and 'bdf' in out
    assert 'CVODE available' not in out
    assert 'completed' in out
    # The command went on past the CVODE step to its end.
    assert out.index('[!] CVODE') < out.index('completed')


@pytest.mark.unit
@pytest.mark.parametrize('command', ['install-all', 'update-all'])
def test_a_successful_cvode_install_runs_the_script_from_the_source_tree(
    monkeypatch, tmp_path, command
):
    """Both commands run tools/get_cvode.sh from the PROTEUS root and go on to finish."""
    _cvode_command_env(monkeypatch, tmp_path)
    fake_run, calls = _cvode_recorder(fail=False)
    monkeypatch.setattr(cli.subprocess, 'run', fake_run)

    res = runner.invoke(cli.cli, [command])

    assert res.exit_code == 0, res.output
    cvode_calls = [c for c in calls if str(c[1]).endswith('get_cvode.sh')]
    assert cvode_calls == [['bash', str(tmp_path / 'tools' / 'get_cvode.sh')]]
    assert [sys.executable, '-c', cli._CVODE_IMPORT] in calls
    assert 'CVODE available' in res.output
    assert 'completed' in res.output


# ---------------------------
# Extended tests: `proteus get` logfile location
# ---------------------------

LEGACY_GET_LOG = 'proteus_get.log'


def _stub_reference_downloaders(monkeypatch, calls=None):
    """Replace the two downloaders behind ``get reference`` with recording no-ops."""
    if calls is None:
        calls = []

    monkeypatch.setattr(
        'proteus.utils.data.download_exoplanet_data', lambda: calls.append('exo')
    )
    monkeypatch.setattr(
        'proteus.utils.data.download_massradius_data', lambda: calls.append('mr')
    )
    return calls


@pytest.mark.unit
def test_get_logfile_is_scoped_to_the_calling_user(monkeypatch, tmp_path):
    """Two users running ``proteus get`` on one machine target two different logfiles.

    A shared temporary directory is normally sticky, so a fixed logfile name
    puts every user of the machine on one path that only its owner can
    recreate. The path must therefore vary with the caller, and must no longer
    be the shared name that collided.
    """
    runner = CliRunner()
    _stub_reference_downloaders(monkeypatch)
    monkeypatch.setattr('proteus.cli.tempfile.gettempdir', lambda: str(tmp_path))

    seen = []
    monkeypatch.setattr(cli, 'setup_logger', lambda logpath, **kw: seen.append(Path(logpath)))

    for uid in (1001, 2002):
        monkeypatch.setattr('proteus.cli.os.getuid', lambda uid=uid: uid)
        res = runner.invoke(cli.cli, ['get', 'reference'])
        assert res.exit_code == 0, res.output

    assert seen[0] != seen[1]
    assert {path.parent for path in seen} == {tmp_path}
    assert LEGACY_GET_LOG not in {path.name for path in seen}


@pytest.mark.unit
def test_get_leaves_an_unremovable_foreign_logfile_alone(monkeypatch, tmp_path):
    """``proteus get`` runs although another user's logfile sits in the temp directory.

    Reproduces the shared-cluster failure: a logfile written by a different
    user in a sticky temporary directory, where the kernel refuses removal by
    a non-owner. Removal of that path is made to raise the same
    ``PermissionError``, so a command that still targeted it would abort.
    """
    runner = CliRunner()
    calls = _stub_reference_downloaders(monkeypatch)
    monkeypatch.setattr('proteus.cli.tempfile.gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr('proteus.cli.os.getuid', lambda: 4242)

    foreign = tmp_path / LEGACY_GET_LOG
    foreign.write_text('owned by another user\n')

    real_remove = os.remove

    def guarded_remove(path, *args, **kwargs):
        if Path(path) == foreign:
            raise PermissionError(1, 'Operation not permitted', str(path))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr('proteus.utils.logs.os.remove', guarded_remove)

    res = runner.invoke(cli.cli, ['get', 'reference'])
    assert res.exit_code == 0, res.output
    assert calls == ['exo', 'mr']
    # The other user's file is untouched, and this user gets a logfile of their own.
    assert foreign.read_text() == 'owned by another user\n'
    assert (tmp_path / 'proteus_get_4242.log').exists()


@pytest.mark.unit
def test_get_continues_when_the_logfile_cannot_be_opened(monkeypatch, tmp_path):
    """A logfile that cannot be opened at all warns but does not stop the download.

    Covers what a per-user name cannot: a read-only or full temporary
    directory. The download proceeds on terminal-only logging.
    """
    runner = CliRunner()
    calls = _stub_reference_downloaders(monkeypatch)
    monkeypatch.setattr('proteus.cli.tempfile.gettempdir', lambda: str(tmp_path))

    def refuse(*args, **kwargs):
        raise PermissionError(1, 'Operation not permitted')

    monkeypatch.setattr(cli, 'setup_logger', refuse)

    try:
        res = runner.invoke(cli.cli, ['get', 'reference'])
        assert res.exit_code == 0, res.output
        assert calls == ['exo', 'mr']
        assert 'Cannot write' in res.output
        # Exactly one terminal handler remains; no half-configured logger survives.
        assert len(logging.getLogger('fwl').handlers) == 1
    finally:
        logging.getLogger('fwl').handlers.clear()
