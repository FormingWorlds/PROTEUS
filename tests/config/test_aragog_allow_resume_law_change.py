"""The aragog allow_resume_law_change option over its input domain.

Source: ``proteus.config._interior.Aragog.allow_resume_law_change`` and the bool structure
hook of ``proteus.config``. Contract: the default is false, a TOML true or false
passes the reader, and any other TOML value stops the load with the field named, so a
string "false" or a 1 cannot turn the switch on.
"""

from __future__ import annotations

import math

import pytest
import tomlkit
from helpers import PROTEUS_ROOT

from proteus.config import read_config_object
from proteus.config._interior import Aragog

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load(tmp_path, value):
    """Read input/minimal.toml with allow_resume_law_change set to ``value``."""
    doc = tomlkit.parse((PROTEUS_ROOT / 'input' / 'minimal.toml').read_text())
    doc.setdefault('interior_energetics', tomlkit.table()).setdefault(
        'aragog', tomlkit.table()
    )['allow_resume_law_change'] = value
    path = tmp_path / 'allow.toml'
    path.write_text(tomlkit.dumps(doc))
    return read_config_object(path).interior_energetics.aragog.allow_resume_law_change


@pytest.mark.parametrize('value', [True, False])
def test_the_reader_takes_a_toml_boolean(tmp_path, value):
    """A TOML true or false reaches the config unchanged, and a file without the field reads false."""
    assert _load(tmp_path, value) is value
    path = PROTEUS_ROOT / 'input' / 'minimal.toml'
    assert read_config_object(path).interior_energetics.aragog.allow_resume_law_change is False


@pytest.mark.parametrize(
    'value',
    [1, 0, 1.0, 0.0, 'true', 'false', 'yes', '', math.nan, math.inf],
    ids=[
        'int 1',
        'int 0',
        'float 1',
        'float 0',
        'string true',
        'string false',
        'yes',
        'empty',
        'nan',
        'inf',
    ],
)
def test_a_value_that_is_not_a_toml_boolean_stops_the_load(tmp_path, value):
    """A number or a string stops the load with the bool-field message and the dotted path;
    the string 'false' does not read as true."""
    with pytest.raises(
        ValueError,
        match=r'interior_energetics\.aragog\.allow_resume_law_change: must be true or false, got ',
    ) as err:
        _load(tmp_path, value)
    assert repr(value) in str(err.value)


def test_construction_and_assignment_refuse_a_non_boolean():
    """The attrs validator refuses a non-boolean at construction and by assignment."""
    hypothesis = pytest.importorskip('hypothesis')
    st = hypothesis.strategies

    @hypothesis.settings(derandomize=True, deadline=None)
    @hypothesis.given(
        value=st.one_of(
            st.text(), st.integers(), st.floats(allow_nan=True, allow_infinity=True), st.none()
        )
    )
    def refused(value):
        with pytest.raises(TypeError, match='allow_resume_law_change` must be a bool'):
            Aragog(allow_resume_law_change=value)
        a = Aragog(allow_resume_law_change=True)
        with pytest.raises(TypeError, match='allow_resume_law_change` must be a bool'):
            a.allow_resume_law_change = value
        assert a.allow_resume_law_change is True

    refused()
