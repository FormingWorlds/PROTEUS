from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock, patch

import pytest
import requests
from packaging.version import Version

from proteus.doctor import (
    BasePackage,
    GitPackage,
    PythonPackage,
    _editable_checkout_path,
    _git_checkout_state,
    doctor_entry,
)


class DummyPackage(BasePackage):
    def __init__(self, name: str, current: str, latest: str):
        super().__init__(name=name)
        self._current = current
        self._latest = latest

    def current_version(self) -> Version:
        return Version(self._current)

    def latest_version(self) -> Version:
        return Version(self._latest)


@pytest.mark.unit
def test_get_status_message_reports_update_for_newer_release():
    package = DummyPackage('fwl-proteus', current='25.10.15', latest='25.11.19')

    message = package.get_status_message()

    assert 'Update available 25.10.15 -> 25.11.19' in message


@pytest.mark.unit
def test_get_status_message_does_not_suggest_downgrade():
    package = DummyPackage('fwl-proteus', current='25.11.19', latest='25.10.15')

    message = package.get_status_message()

    assert 'Update available' not in message
    assert 'Local version 25.11.19 is newer than latest release 25.10.15' in message


@pytest.mark.unit
def test_get_status_message_handles_unparseable_versions():
    package = DummyPackage('fwl-proteus', current='main', latest='v25.11.19')

    message = package.get_status_message()

    assert 'Update available' not in message
    assert "InvalidVersion - Invalid version: 'main'" in message


@pytest.mark.unit
def test_python_package_latest_version_reads_json_response():
    package = PythonPackage(name='fwl-proteus')
    response = Mock(ok=True)
    response.json.return_value = {'info': {'version': '25.11.19'}}

    with patch('proteus.doctor.requests.get', return_value=response):
        assert package.latest_version() == Version('25.11.19')


@pytest.mark.unit
def test_python_package_latest_version_raises_on_bad_http_status():
    package = PythonPackage(name='fwl-proteus')
    response = Mock(ok=False)
    response.raise_for_status.side_effect = requests.HTTPError('boom')

    with patch('proteus.doctor.requests.get', return_value=response):
        with pytest.raises(requests.HTTPError, match='boom'):
            package.latest_version()


@pytest.mark.unit
def test_git_package_current_version_converts_missing_repo_to_package_not_found():
    package = GitPackage(
        name='SOCRATES',
        owner='FormingWorlds',
        version_getter=Mock(side_effect=FileNotFoundError()),
    )

    with pytest.raises(PackageNotFoundError, match='SOCRATES is not installed'):
        package.current_version()


@pytest.mark.unit
def test_git_package_latest_version_reads_tag_name_from_json_response():
    package = GitPackage(name='SOCRATES', owner='FormingWorlds', version_getter=Mock())
    response = Mock(ok=True)
    response.json.return_value = {'tag_name': 'v2026.01'}

    with patch('proteus.doctor.requests.get', return_value=response):
        assert package.latest_version() == Version('v2026.01')


@pytest.mark.unit
def test_git_package_latest_version_raises_on_bad_http_status():
    package = GitPackage(name='SOCRATES', owner='FormingWorlds', version_getter=Mock())
    response = Mock(ok=False)
    response.raise_for_status.side_effect = requests.HTTPError('nope')

    with patch('proteus.doctor.requests.get', return_value=response):
        with pytest.raises(requests.HTTPError, match='nope'):
            package.latest_version()


@pytest.mark.unit
def test_git_package_falls_back_to_tags_when_releases_latest_404():
    """SOCRATES has no formal GitHub releases, only tags.

    The /releases/latest endpoint returns 404 for repos without
    formal releases. The doctor must then read /tags and pick the
    most recent tag rather than reporting SOCRATES as broken on
    every health check.
    """
    package = GitPackage(name='SOCRATES', owner='FormingWorlds', version_getter=Mock())

    releases_response = Mock(status_code=404)
    tags_response = Mock(status_code=200)
    tags_response.json.return_value = [
        {'name': '25.05'},  # /tags returns most-recent-first
        {'name': '25.04'},
        {'name': '25.03'},
    ]

    with patch(
        'proteus.doctor.requests.get',
        side_effect=[releases_response, tags_response],
    ):
        assert package.latest_version() == Version('25.05')


@pytest.mark.unit
def test_git_package_fallback_raises_when_no_tags_either():
    """A repo with no releases AND no tags must surface an explicit error.

    Without this, an empty list from /tags would IndexError out of
    ``tags[0]['name']`` with a stack trace that obscures the real
    diagnosis (the upstream repo has no version metadata).
    """
    package = GitPackage(name='SOCRATES', owner='FormingWorlds', version_getter=Mock())

    releases_response = Mock(status_code=404)
    tags_response = Mock(status_code=200)
    tags_response.json.return_value = []

    with patch(
        'proteus.doctor.requests.get',
        side_effect=[releases_response, tags_response],
    ):
        with pytest.raises(RuntimeError, match='no releases or tags found'):
            package.latest_version()


@pytest.mark.unit
def test_git_package_fallback_propagates_tags_http_error():
    """If /tags itself fails the doctor must not swallow the HTTPError."""
    package = GitPackage(name='SOCRATES', owner='FormingWorlds', version_getter=Mock())

    releases_response = Mock(status_code=404)
    tags_response = Mock(status_code=500)
    tags_response.raise_for_status.side_effect = requests.HTTPError('tags 500')

    with patch(
        'proteus.doctor.requests.get',
        side_effect=[releases_response, tags_response],
    ):
        with pytest.raises(requests.HTTPError, match='tags 500'):
            package.latest_version()


def _make_editable_distribution(file_url: str) -> Mock:
    """Build a Mock importlib.metadata.Distribution for an editable install.

    ``direct_url.json`` (PEP 610) is the only stable signal pip records
    for editable installs. ``dir_info.editable = True`` plus a
    ``file://`` URL is what ``pip install -e <path>`` writes.
    """
    dist = Mock()
    dist.read_text.return_value = f'{{"url": "{file_url}", "dir_info": {{"editable": true}}}}'
    return dist


@pytest.mark.unit
def test_editable_checkout_path_returns_none_for_missing_package():
    """Packages that are not installed must not crash the doctor."""
    with patch(
        'proteus.doctor.importlib.metadata.distribution',
        side_effect=PackageNotFoundError('fwl-aragog'),
    ):
        assert _editable_checkout_path('fwl-aragog') is None


@pytest.mark.unit
def test_editable_checkout_path_returns_none_for_wheel_install():
    """A package installed from a wheel has no ``direct_url.json``."""
    dist = Mock()
    dist.read_text.return_value = None
    with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
        assert _editable_checkout_path('fwl-aragog') is None


@pytest.mark.unit
def test_editable_checkout_path_rejects_non_editable_vcs_install():
    """A VCS install pinned to a tag is not editable and must not be reported.

    pip writes ``direct_url.json`` for ``pip install git+https://.../@tag`` too,
    but ``dir_info.editable`` is False. Reporting it as editable would
    mislead users into thinking a sibling checkout exists when none does.
    """
    dist = Mock()
    dist.read_text.return_value = (
        '{"url": "https://github.com/FormingWorlds/aragog.git",'
        ' "vcs_info": {"vcs": "git", "commit_id": "abc123"}}'
    )
    with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
        assert _editable_checkout_path('fwl-aragog') is None


@pytest.mark.unit
def test_editable_checkout_path_returns_unquoted_filesystem_path():
    """File URLs with percent-encoded spaces must round-trip to a real path."""
    dist = _make_editable_distribution('file:///Users/Tim%20L/git/aragog')
    with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
        path = _editable_checkout_path('fwl-aragog')
    assert path == '/Users/Tim L/git/aragog'


@pytest.mark.unit
def test_editable_checkout_path_returns_none_for_malformed_json():
    """A corrupted ``direct_url.json`` must not raise out of doctor."""
    dist = Mock()
    dist.read_text.return_value = '{not valid json'
    with patch('proteus.doctor.importlib.metadata.distribution', return_value=dist):
        assert _editable_checkout_path('fwl-aragog') is None


@pytest.mark.unit
def test_git_checkout_state_reports_clean_tree():
    """A clean working tree returns ``(short_hash, False)``."""

    def fake_check_output(cmd, **_kwargs):
        if cmd[-1] == 'HEAD':
            return 'd051902\n'
        if 'status' in cmd:
            return ''
        raise AssertionError(f'unexpected git invocation: {cmd}')

    with patch('proteus.doctor.subprocess.check_output', side_effect=fake_check_output):
        state = _git_checkout_state('/tmp/aragog')
    assert state == ('d051902', False)


@pytest.mark.unit
def test_git_checkout_state_reports_dirty_tree():
    """Any non-whitespace porcelain output marks the tree dirty."""

    def fake_check_output(cmd, **_kwargs):
        if cmd[-1] == 'HEAD':
            return 'd051902\n'
        if 'status' in cmd:
            return ' M src/aragog/solver/entropy_solver.py\n?? scratch.py\n'
        raise AssertionError(f'unexpected git invocation: {cmd}')

    with patch('proteus.doctor.subprocess.check_output', side_effect=fake_check_output):
        state = _git_checkout_state('/tmp/aragog')
    assert state == ('d051902', True)


@pytest.mark.unit
def test_git_checkout_state_returns_none_when_git_unavailable():
    """A missing ``git`` binary or a non-repo path must not crash doctor."""
    with patch(
        'proteus.doctor.subprocess.check_output',
        side_effect=FileNotFoundError('git'),
    ):
        assert _git_checkout_state('/tmp/not-a-repo') is None


@pytest.mark.unit
def test_git_checkout_state_returns_none_when_not_a_git_repo():
    """``git rev-parse`` exits non-zero outside a working tree."""
    with patch(
        'proteus.doctor.subprocess.check_output',
        side_effect=subprocess.CalledProcessError(128, 'git'),
    ):
        assert _git_checkout_state('/tmp/not-a-repo') is None


@pytest.mark.unit
def test_python_package_status_unannotated_for_wheel_install():
    """Wheel installs render the plain ``name: ok`` form (no annotation)."""
    package = PythonPackage(name='fwl-aragog')
    with (
        patch.object(package, 'current_version', return_value=Version('26.5.13')),
        patch.object(package, 'latest_version', return_value=Version('26.5.13')),
        patch('proteus.doctor._editable_checkout_path', return_value=None),
    ):
        message = package.get_status_message()
    assert 'editable' not in message
    assert 'ok' in message


@pytest.mark.unit
def test_python_package_status_annotates_clean_editable_install():
    """An editable install adds ``[editable @ basename -> hash]``."""
    package = PythonPackage(name='fwl-aragog')
    with (
        patch.object(package, 'current_version', return_value=Version('26.5.13')),
        patch.object(package, 'latest_version', return_value=Version('26.5.13')),
        patch(
            'proteus.doctor._editable_checkout_path',
            return_value='/Users/tim/git/PROTEUS/aragog',
        ),
        patch('proteus.doctor._git_checkout_state', return_value=('d051902', False)),
    ):
        message = package.get_status_message()
    assert 'editable @ aragog -> d051902' in message
    assert '(dirty)' not in message


@pytest.mark.unit
def test_python_package_status_marks_dirty_editable_install():
    """A dirty editable install appends ``(dirty)`` so users see uncommitted work."""
    package = PythonPackage(name='fwl-zalmoxis')
    with (
        patch.object(package, 'current_version', return_value=Version('26.5.13')),
        patch.object(package, 'latest_version', return_value=Version('26.5.13')),
        patch(
            'proteus.doctor._editable_checkout_path',
            return_value='/Users/tim/git/PROTEUS/Zalmoxis',
        ),
        patch('proteus.doctor._git_checkout_state', return_value=('39308df', True)),
    ):
        message = package.get_status_message()
    assert 'editable @ Zalmoxis -> 39308df (dirty)' in message


@pytest.mark.unit
def test_python_package_status_handles_editable_without_git_metadata():
    """An editable checkout outside any git repo still gets a partial annotation."""
    package = PythonPackage(name='fwl-vulcan')
    with (
        patch.object(package, 'current_version', return_value=Version('26.4.22')),
        patch.object(package, 'latest_version', return_value=Version('26.4.22')),
        patch(
            'proteus.doctor._editable_checkout_path',
            return_value='/tmp/VULCAN',
        ),
        patch('proteus.doctor._git_checkout_state', return_value=None),
    ):
        message = package.get_status_message()
    assert 'editable @ VULCAN' in message
    assert '->' not in message.split('editable @ VULCAN', 1)[1]


@pytest.mark.unit
def test_doctor_entry_prints_environment_variables_before_packages():
    fake_package = Mock()
    fake_package.get_status_message.return_value = 'pkg: ok'
    outputs: list[str] = []

    with (
        patch('proteus.doctor.VARIABLES', ('RAD_DIR',)),
        patch('proteus.doctor.PACKAGES', (fake_package,)),
        patch('proteus.doctor.get_env_var_status_message', return_value='RAD_DIR: ok'),
        patch('click.secho', side_effect=lambda text, **_: outputs.append(text)),
        patch('click.echo', side_effect=lambda text: outputs.append(text)),
    ):
        doctor_entry()

    assert outputs == ['Environment variables', 'RAD_DIR: ok', '\nPackages', 'pkg: ok']
