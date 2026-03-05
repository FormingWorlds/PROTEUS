# Test PROTEUS terminal CLI and commands
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from proteus import __version__ as proteus_version
from proteus import cli

runner = CliRunner()


@pytest.mark.unit
def test_doctor():
    # run PROTEUS doctor command
    response = runner.invoke(cli.doctor, [])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
    assert 'Packages' in response.output
    assert 'AGNI' in response.output
    assert 'fwl-mors' in response.output


@pytest.mark.unit
def test_version():
    # run PROTEUS version command
    response = runner.invoke(cli.cli, ['--version'])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
    assert str(proteus_version) in response.output


@pytest.mark.unit
def test_get():
    # run PROTEUS get command
    response = runner.invoke(cli.get, ['reference'])
    assert response.exit_code == 0

    response = runner.invoke(cli.get, ['surfaces'])
    assert response.exit_code == 0

    response = runner.invoke(cli.get, ['spectral', '-n', 'Frostflow', '-b', '16'])
    assert response.exit_code == 0

    response = runner.invoke(cli.get, ['muscles', '--star', 'trappist-1'])
    assert response.exit_code == 0

    response = runner.invoke(cli.get, ['phoenix', '--feh', '0.0', '--alpha', '0.0'])
    assert response.exit_code == 0

    response = runner.invoke(cli.get, ['solar'])
    assert response.exit_code == 0


# ---------------------------
# Extended tests: proteus get subcommands
# ---------------------------


@pytest.mark.unit
def test_get_help_extended():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', '--help'])
    assert res.exit_code == 0
    assert 'Get data and modules' in res.output
    # Subcommands should be listed
    for sub in ['reference', 'surfaces', 'spectral', 'muscles', 'phoenix', 'solar', 'stellar']:
        assert sub in res.output


@pytest.mark.unit
def test_get_unknown_subcommand():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'does-not-exist'])
    assert res.exit_code != 0
    assert 'No such command' in res.output


# ---- spectral ----


@pytest.mark.unit
def test_get_spectral_defaults(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_download_spectral_file(name, bands):
        calls.append((name, bands))

    # imported inside command: from .utils.data import download_spectral_file
    monkeypatch.setattr(
        'proteus.utils.data.download_spectral_file', fake_download_spectral_file
    )

    res = runner.invoke(cli.cli, ['get', 'spectral'])
    assert res.exit_code == 0
    assert calls == [(None, None)]


@pytest.mark.unit
def test_get_spectral_forwards_args(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_download_spectral_file(name, bands):
        calls.append((name, bands))

    monkeypatch.setattr(
        'proteus.utils.data.download_spectral_file', fake_download_spectral_file
    )

    res = runner.invoke(cli.cli, ['get', 'spectral', '-n', 'Frostflow', '-b', '16'])
    assert res.exit_code == 0
    assert calls == [('Frostflow', '16')]


# ---- surfaces / reference ----


@pytest.mark.unit
def test_get_surfaces_calls_downloader(monkeypatch):
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
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles'])
    assert res.exit_code != 0
    assert 'Provide --star NAME or use --all.' in res.output


@pytest.mark.unit
def test_get_muscles_list_outputs_names():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--list'])
    assert res.exit_code == 0
    # spot-check a couple known entries
    assert 'trappist-1' in res.output
    assert 'gj876' in res.output


@pytest.mark.unit
def test_get_muscles_star_normalization_and_download(monkeypatch):
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
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--star', 'gj87'])
    assert res.exit_code != 0
    assert 'Unknown MUSCLES star' in res.output
    # Implementation adds either a Did-you-mean hint or at least the --list hint
    assert ('Did you mean:' in res.output) or ('Use --list' in res.output)


@pytest.mark.unit
def test_get_muscles_all_and_star_mutually_exclusive():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['get', 'muscles', '--all', '--star', 'trappist-1'])
    assert res.exit_code != 0
    assert 'Use either --all or --star NAME' in res.output


@pytest.mark.unit
def test_get_muscles_all_reports_failures(monkeypatch):
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
    runner = CliRunner()

    def fake_GetFWLData():
        return tmp_path

    def fake_download_stellar_spectra(folders=('solar',)):
        solar_dir = tmp_path / 'stellar_spectra' / 'solar'
        solar_dir.mkdir(parents=True, exist_ok=True)
        (solar_dir / 'dummy.txt').write_text('ok')

    monkeypatch.setattr('proteus.utils.data.GetFWLData', fake_GetFWLData)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'solar'])
    assert res.exit_code == 0
    assert 'Solar spectra downloaded successfully.' in res.output
    assert str(tmp_path / 'stellar_spectra' / 'solar') in res.output


@pytest.mark.unit
def test_get_solar_raises_if_no_files_found(monkeypatch, tmp_path):
    runner = CliRunner()

    def fake_GetFWLData():
        return tmp_path

    def fake_download_stellar_spectra(folders=('solar',)):
        # create directory but no files
        (tmp_path / 'stellar_spectra' / 'solar').mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr('proteus.utils.data.GetFWLData', fake_GetFWLData)
    monkeypatch.setattr(
        'proteus.utils.data.download_stellar_spectra', fake_download_stellar_spectra
    )

    res = runner.invoke(cli.cli, ['get', 'solar'])
    assert res.exit_code != 0
    assert 'no files were found' in res.output.lower()
    assert 'zenodo_download.log' in res.output
    assert 'zenodo_validate.log' in res.output


@pytest.mark.unit
def test_get_solar_wraps_downloader_exception(monkeypatch, tmp_path):
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
    runner = CliRunner()
    calls = []

    def fake():
        calls.append(subcommand)

    monkeypatch.setattr(target, fake)

    res = runner.invoke(cli.cli, ['get', subcommand])
    assert res.exit_code == 0
    assert calls == [subcommand]
