from __future__ import annotations

import pytest

from proteus.doctor import VARIABLES, BasePackage


class DummyPackage(BasePackage):
    def __init__(self, name: str, current: str, latest: str):
        super().__init__(name=name)
        self._current = current
        self._latest = latest

    def current_version(self) -> str:
        return self._current

    def latest_version(self) -> str:
        return self._latest


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
    assert 'Could not compare versions main and v25.11.19' in message


@pytest.mark.unit
def test_doctor_checks_expected_environment_variables():
    assert set(VARIABLES) == {'FWL_DATA', 'RAD_DIR', 'ZALMOXIS_ROOT', 'FC_DIR', 'LA_DIR'}
