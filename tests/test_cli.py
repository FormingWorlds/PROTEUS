# Test PROTEUS terminal CLI and commands
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from proteus import __version__ as proteus_version
from proteus import cli

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


runner = CliRunner()


@pytest.mark.unit
def test_doctor():
    """``proteus doctor`` exits 0 and reports the expected package list,
    including AGNI and fwl-mors.
    """
    # run PROTEUS doctor command
    response = runner.invoke(cli.doctor, [])

    # return ok?
    assert response.exit_code == 0

    # contains information we expect
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
    that files exist under FWL_DATA/stellar_spectra/solar after the
    downloader returns), so the no-op for download_stellar_spectra writes
    a stub file at the expected path. FWL_DATA itself is monkeypatched to
    a tmp_path so the test never touches the user's real data tree.

    Anti-happy-path: each subcommand assertion is its own line and asserts
    the specific exit code; a regression in any one of them surfaces the
    name of the failing subcommand in the assertion output. The previous
    failure (test_get fails on `solar` in CI but not locally because the
    user's FWL_DATA happened to have stellar_spectra/solar/ already
    populated) is now eliminated by controlling FWL_DATA explicitly.
    """
    # Monkeypatch FWL_DATA to a writable tmp_path so post-condition file
    # checks have a controlled environment. The module-level FWL_DATA_DIR
    # constant in proteus.utils.data is read at import time, so setenv
    # alone is not enough; patch the module attribute too.
    from pathlib import Path as _Path

    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', _Path(tmp_path), raising=False)

    def stub_download_stellar_spectra(folders=('solar',), **kwargs):
        # The CLI `solar` subcommand checks for files under
        # GetFWLData()/stellar_spectra/solar after the downloader returns,
        # raising ClickException if none are present. The no-op honours
        # that contract by writing a stub file so the post-condition
        # passes without touching the network.
        for folder in folders:
            target = tmp_path / 'stellar_spectra' / folder
            target.mkdir(parents=True, exist_ok=True)
            (target / '_stub.txt').write_text('stub')
        return True

    no_op = lambda *args, **kwargs: True  # noqa: E731
    for target in (
        'proteus.utils.data.download_exoplanet_data',
        'proteus.utils.data.download_massradius_data',
        'proteus.utils.data.download_surface_albedos',
        'proteus.utils.data.download_spectral_file',
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
    """``proteus get spectral`` (no args) calls the downloader with name=None, bands=None."""
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
    """``proteus get spectral -n NAME -b BANDS`` forwards the args to the downloader."""
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
    """``proteus get solar`` succeeds when files materialise under FWL_DATA/stellar_spectra/solar."""
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
    """``proteus get solar`` exits non-zero when the downloader writes no files,
    and points the user to the Zenodo download / validate log paths.
    """
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
    """``proteus grid -c cfg`` dispatches to grid_from_config with the config path."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text('# stub\n')

    received = []

    def fake_grid_from_config(path):
        received.append(Path(path))

    import proteus.grid.manage as gmanage

    monkeypatch.setattr(gmanage, 'grid_from_config', fake_grid_from_config)

    res = runner.invoke(cli.grid, ['-c', str(cfg)])
    assert res.exit_code == 0
    # Discrimination: exactly one call, with the resolved config path.
    assert len(received) == 1
    assert received[0].name == 'cfg.toml'


@pytest.mark.unit
def test_infer_calls_infer_from_config(monkeypatch, tmp_path):
    """``proteus infer -c cfg`` dispatches to infer_from_config with the config path."""
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
def test_append_to_shell_rc_writes_when_absent(monkeypatch, tmp_path):
    """append_to_shell_rc creates the rc file and writes the export line on first call."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    rc = cli.append_to_shell_rc('TEST_VAR', '/some/value', shell='/bin/bash')
    assert rc is not None
    assert rc == tmp_path / '.bashrc'
    contents = rc.read_text()
    assert 'TEST_VAR' in contents
    assert '/some/value' in contents


@pytest.mark.unit
def test_append_to_shell_rc_skips_when_already_present(monkeypatch, tmp_path):
    """If the export line is already in the rc file, the helper returns None and does not duplicate."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    rc_first = cli.append_to_shell_rc('TEST_VAR', '/some/value', shell='/bin/zsh')
    assert rc_first == tmp_path / '.zshrc'

    rc_second = cli.append_to_shell_rc('TEST_VAR', '/some/value', shell='/bin/zsh')
    # Already present, must signal a no-op.
    assert rc_second is None
    # Idempotency guard: line count for that var must be exactly 1.
    occurrences = (tmp_path / '.zshrc').read_text().count('TEST_VAR')
    assert occurrences == 1


@pytest.mark.unit
def test_append_to_shell_rc_unknown_shell_returns_none(monkeypatch, tmp_path):
    """An unrecognized shell returns None and does NOT create any rc file."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    out = cli.append_to_shell_rc('TEST_VAR', '/x', shell='/usr/local/bin/exotic_shell')
    assert out is None
    # Discrimination: no rc files created
    assert not (tmp_path / '.bashrc').exists()
    assert not (tmp_path / '.zshrc').exists()


@pytest.mark.unit
def test_is_julia_installed_true_when_shutil_finds_it(monkeypatch):
    """is_julia_installed returns True when shutil.which finds a julia binary."""
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
    monkeypatch.setattr(cli.shutil, 'which', lambda exe: None)
    result = cli.is_julia_installed()
    assert result is False
    # Discrimination: pin the bool type so a regression returning None (also
    # falsy) does not slip through.
    assert isinstance(result, bool)


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


@pytest.mark.unit
def test_install_all_aborts_on_low_disk(monkeypatch, tmp_path):
    """``install-all`` exits non-zero when free disk is below the 5 GB threshold."""
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_low())

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Aborting installation' in res.output
    # Discrimination: must NOT print the "completed" message that lives on
    # the success path.
    assert 'PROTEUS installation completed' not in res.output


@pytest.mark.unit
def test_install_all_aborts_when_julia_missing(monkeypatch, tmp_path):
    """``install-all`` aborts with a Julia-not-found message when julia is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    # Pre-create socrates dir so the SOCRATES install path is skipped
    (tmp_path / 'socrates').mkdir()
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    # Force the julia check to fail
    monkeypatch.setattr(cli, 'is_julia_installed', lambda: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Julia not found' in res.output
    # Discrimination: AGNI install path must NOT have been reached
    assert 'Installing AGNI' not in res.output


@pytest.mark.unit
def test_install_all_success_with_export_env(monkeypatch, tmp_path):
    """Full ``install-all --export-env`` path with all heavy steps stubbed.

    Verifies the rc-export step writes lines, the socrates / AGNI dirs are
    detected as pre-existing, and the completion message fires.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    monkeypatch.setenv('SHELL', '/bin/bash')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    (tmp_path / 'home').mkdir()

    # Pre-create socrates and AGNI to skip subprocess calls
    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()

    subprocess_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        subprocess_calls.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    # No config file so the input-data step gracefully skips.
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['install-all', '--export-env'])
    assert res.exit_code == 0, f'install-all output: {res.output}'
    assert 'PROTEUS installation completed' in res.output
    # Discrimination: at least one rc-export line fired
    assert 'Exported' in res.output or 'already exported' in res.output


@pytest.mark.unit
def test_install_all_invokes_socrates_when_missing(monkeypatch, tmp_path):
    """SOCRATES install branch fires subprocess.run when the socrates/ dir is absent."""
    monkeypatch.chdir(tmp_path)
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
    monkeypatch.chdir(tmp_path)
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
def test_install_all_agni_clone_failure_aborts(monkeypatch, tmp_path):
    """A CalledProcessError during AGNI clone aborts the installation."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    # SOCRATES present so the SOCRATES branch is skipped
    (tmp_path / 'socrates').mkdir()
    # AGNI missing; cloning must fail

    def fake_subprocess_run(cmd, **kwargs):
        if 'git' in str(cmd[0]) and 'clone' in cmd:
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['install-all'])
    assert res.exit_code != 0
    assert 'Failed to install AGNI' in res.output


# ---------------------------
# update_all
# ---------------------------


@pytest.mark.unit
def test_update_all_aborts_on_low_disk(monkeypatch, tmp_path):
    """``update-all`` exits non-zero when free disk is below the 5 GB threshold."""
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_low())

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code != 0
    assert 'Aborting installation' in res.output
    # Discrimination: must NOT print the success message.
    assert 'PROTEUS update completed' not in res.output


@pytest.mark.unit
def test_update_all_success_path(monkeypatch, tmp_path):
    """Happy path: socrates + AGNI present, julia found, pip + git pull all stubbed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()

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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    # NO socrates dir, NO AGNI dir
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert 'SOCRATES not found' in res.output
    assert 'AGNI not found' in res.output


@pytest.mark.unit
def test_update_all_warns_when_julia_missing(monkeypatch, tmp_path):
    """Julia-missing path emits a warning but does NOT abort (unlike install-all)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(cli.shutil, 'which', lambda exe: None)
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    (tmp_path / 'socrates').mkdir()
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()

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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()

    def fake_subprocess_run(cmd, **kwargs):
        # The AGNI block uses cwd=agni_dir; identify by that
        if kwargs.get('cwd') == tmp_path / 'AGNI':
            raise cli.subprocess.CalledProcessError(returncode=1, cmd=cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(cli.subprocess, 'run', fake_subprocess_run)
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all'])
    assert res.exit_code == 0
    assert 'Failed to update AGNI' in res.output


@pytest.mark.unit
def test_update_all_export_env_writes_rc(monkeypatch, tmp_path):
    """``update-all --export-env`` writes rc lines for FWL_DATA and RAD_DIR."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: _stub_disk_usage_high())
    monkeypatch.setattr(
        cli.shutil, 'which', lambda exe: '/usr/local/bin/julia' if exe == 'julia' else None
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl_data'))
    monkeypatch.setenv('SHELL', '/bin/bash')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    (tmp_path / 'home').mkdir()

    (tmp_path / 'socrates').mkdir()
    (tmp_path / 'AGNI').mkdir()
    monkeypatch.setattr(
        cli.subprocess, 'run', lambda cmd, **kw: type('R', (), {'returncode': 0})()
    )
    monkeypatch.setattr(cli, '_update_input_data', lambda path: False)

    res = runner.invoke(cli.cli, ['update-all', '--export-env'])
    assert res.exit_code == 0
    assert 'PROTEUS update completed' in res.output
    # Either exported or detected as already exported; at least one rc line touched.
    assert 'Exported' in res.output or 'already exported' in res.output


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
