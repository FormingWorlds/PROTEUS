from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock, patch

import pytest
import requests
from packaging.version import Version

from proteus.doctor import BasePackage, GitPackage, PythonPackage, doctor_entry


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
