"""
Unit tests for proteus.interior_energetics.resume_laws: the check that a resumed
aragog run keeps the separation laws it was written with.

Invariant: a resumed run continues under one separation law. The stored laws are the
``interior_energetics.aragog`` fields of the run's ``init_coupler.toml``; a stored value
that differs from the new config refuses the resume, unless
``allow_resume_law_change`` is set. An aragog record without ``separation_viscosity``
counts as ``'melt'``; a law the run did not record, or a run without a record that names
the interior module, resumes only when the new config file sets the field.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import tomlkit

pytest.importorskip('hypothesis')
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from proteus.config import read_config_object  # noqa: E402
from proteus.interior_energetics.resume_laws import (  # noqa: E402
    RECORD_OVERRIDES,
    ResumeSeparationLawError,
    check_resume_separation_laws,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

MINIMAL = Path(__file__).parents[2] / 'input' / 'minimal.toml'
LOGGER = 'fwl.proteus.interior_energetics.resume_laws'


def _user_config(folder, **aragog):
    """A config file from input/minimal.toml with the given aragog fields set; returns (Config, path)."""
    text = MINIMAL.read_text()
    if aragog:
        text += '\n[interior_energetics.aragog]\n' + ''.join(
            f'    {k} = {tomlkit.item(v).as_string()}\n' for k, v in aragog.items()
        )
    path = Path(folder) / 'user.toml'
    path.write_text(text)
    return read_config_object(path), path


def _stored_run(folder, drop=(), **aragog):
    """A run directory whose init_coupler.toml records the given aragog fields; ``drop`` removes fields."""
    out = Path(folder) / 'run'
    out.mkdir()
    config, _ = _user_config(out, **aragog)
    config.write(out / 'init_coupler.toml')
    if drop:
        doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
        for field in drop:
            del doc['interior_energetics']['aragog'][field]
        (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    return out


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'field, stored, new',
    [
        ('separation_buoyancy', 'solid', 'mixture'),
        ('separation_buoyancy', 'mixture', 'solid'),
        ('separation_viscosity', 'melt', 'mixture'),
        ('separation_viscosity', 'mixture', 'melt'),
    ],
    ids=[
        'buoyancy solid to mixture',
        'buoyancy mixture to solid',
        'drag melt to mixture',
        'drag mixture to melt',
    ],
)
def test_a_changed_separation_law_refuses_the_resume(tmp_path, field, stored, new):
    """A resume under a separation law other than the stored one stops and names the field and both laws."""
    out = _stored_run(tmp_path, **{field: stored})
    config, path = _user_config(tmp_path, **{field: new})
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    message = str(err.value)
    assert f'interior_energetics.aragog.{field}' in message
    assert f"'{stored}'" in message and f"'{new}'" in message
    assert 'allow_resume_law_change' in message
    assert isinstance(err.value, RuntimeError)


def test_unchanged_laws_resume_without_a_record(tmp_path, caplog):
    """The same laws as stored give no switch and no log record from the check."""
    out = _stored_run(tmp_path, separation_buoyancy='solid', separation_viscosity='melt')
    config, path = _user_config(
        tmp_path, separation_buoyancy='solid', separation_viscosity='melt'
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert [r for r in caplog.records if r.name == LOGGER] == []


@pytest.mark.parametrize(
    'new, refused',
    [
        ({}, True),
        ({'separation_viscosity': 'mixture'}, True),
        ({'separation_viscosity': 'melt'}, False),
    ],
    ids=['default mixture', 'explicit mixture', 'explicit melt'],
)
def test_a_run_without_a_stored_viscosity_ran_under_melt(tmp_path, new, refused):
    """A run older than the separation_viscosity field ran under aragog's 'melt' drag viscosity."""
    out = _stored_run(tmp_path, drop=('separation_viscosity',))
    config, path = _user_config(tmp_path, **new)
    if refused:
        with pytest.raises(ResumeSeparationLawError, match="separation_viscosity is 'mixture'"):
            check_resume_separation_laws(config, path, out)
    else:
        assert check_resume_separation_laws(config, path, out) == []


@pytest.mark.parametrize('law', ['mixture', 'solid'])
def test_a_run_without_a_stored_buoyancy_needs_the_field_in_the_config_file(
    tmp_path, caplog, law
):
    """Without a stored buoyancy the resume stops unless the config file sets the field, whatever its value."""
    out = _stored_run(tmp_path, drop=('separation_buoyancy',))
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    assert 'separation_buoyancy is not recorded' in str(err.value)
    assert '26.10.22' in str(err.value) and 'aragog version' in str(err.value)
    assert 'before this attempt' in str(err.value)  # the refused attempt writes a log too
    config, path = _user_config(tmp_path, separation_buoyancy=law)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert any(f"'{law}'" in r.getMessage() for r in caplog.records if r.name == LOGGER)


@pytest.mark.parametrize(
    'content',
    [None, b'\x00 not toml [', b"[interior]\nmodule = 'aragog'\n"],
    ids=['no file', 'unreadable file', 'record in the [interior] layout'],
)
def test_a_run_without_a_usable_record_needs_both_laws_in_the_config_file(tmp_path, content):
    """With no record that names the interior module, neither law is guessed: the resume stops
    until the config file sets both, and then takes them as set, 'mixture' viscosity included."""
    out = tmp_path / 'run'
    out.mkdir()
    if content is not None:
        (out / 'init_coupler.toml').write_bytes(content)
    config, path = _user_config(tmp_path, separation_buoyancy='mixture')
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    message = str(err.value)
    assert 'separation_viscosity is not recorded' in message
    assert "'melt'" not in message and 'separation_buoyancy' not in message
    config, path = _user_config(
        tmp_path, separation_buoyancy='mixture', separation_viscosity='mixture'
    )
    assert check_resume_separation_laws(config, path, out) == []


def _record_without_module(tmp_path):
    """A run directory whose init_coupler.toml holds both laws but no interior module."""
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'init_coupler.toml').write_text(
        "[interior_energetics.aragog]\nseparation_viscosity = 'mixture'\n"
        "separation_buoyancy = 'solid'\n"
    )
    return out


def test_a_change_from_a_record_that_names_no_module_is_refused(tmp_path):
    """A record with separation laws but no interior_energetics.module is compared like any
    record; only the 'melt' inference needs a record that names Aragog."""
    out = _record_without_module(tmp_path)
    config, path = _user_config(tmp_path, separation_viscosity='mixture')
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    assert "separation_buoyancy is 'mixture'" in str(err.value)
    assert "written with 'solid'" in str(err.value)


def test_the_recorded_laws_of_a_record_that_names_no_module_resume_quietly(tmp_path, caplog):
    """The same record and the laws it holds resume with no switch and no note."""
    out = _record_without_module(tmp_path)
    config, path = _user_config(
        tmp_path, separation_buoyancy='solid', separation_viscosity='mixture'
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_a_refused_resume_logs_no_note_for_a_law_the_config_file_sets(tmp_path, caplog):
    """A law the config file sets for an unrecorded field is noted only when the resume goes on."""
    out = _stored_run(tmp_path, drop=('separation_buoyancy',), separation_viscosity='melt')
    config, path = _user_config(tmp_path, separation_buoyancy='solid')
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        with pytest.raises(ResumeSeparationLawError, match="separation_viscosity is 'mixture'"):
            check_resume_separation_laws(config, path, out)
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_the_record_overrides_fit_the_config_schema(tmp_path):
    """start() writes init_coupler.toml with RECORD_OVERRIDES; a real config takes them and
    reads the allow switch back as false."""
    config, _ = _user_config(tmp_path, allow_resume_law_change=True)
    config.write(tmp_path / 'record.toml', overrides=RECORD_OVERRIDES)
    reread = read_config_object(tmp_path / 'record.toml').interior_energetics.aragog
    assert reread.allow_resume_law_change is False
    assert config.interior_energetics.aragog.allow_resume_law_change is True  # config unchanged


def test_the_allow_switch_does_not_cover_an_unrecorded_buoyancy(tmp_path, caplog):
    """The allow switch lets a recorded law change but does not stand in for a missing record,
    and a refused resume logs no switch."""
    out = _stored_run(tmp_path, drop=('separation_buoyancy',), separation_viscosity='melt')
    config, path = _user_config(tmp_path, allow_resume_law_change=True)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        with pytest.raises(ResumeSeparationLawError) as err:
            check_resume_separation_laws(config, path, out)
    assert 'separation_buoyancy is not recorded' in str(err.value)
    assert 'allow_resume_law_change does not cover' in str(err.value)
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_the_allow_switch_lets_a_changed_law_resume_and_reports_it(tmp_path, caplog):
    """With allow_resume_law_change the resume goes on, returns the switch and logs a warning."""
    out = _stored_run(tmp_path, separation_buoyancy='solid')
    config, path = _user_config(tmp_path, allow_resume_law_change=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        switches = check_resume_separation_laws(config, path, out)
    assert switches == [('separation_buoyancy', 'solid', 'mixture')]
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warned) == 1 and "'solid'" in warned[0] and "'mixture'" in warned[0]


@pytest.mark.parametrize(
    'stored_module, new_module', [('spider', 'aragog'), ('aragog', 'spider')]
)
def test_the_check_applies_only_to_an_aragog_run_resumed_under_aragog(
    tmp_path, stored_module, new_module
):
    """A resume that does not run aragog on both sides is not checked for the aragog laws."""
    out = _stored_run(tmp_path, separation_buoyancy='solid')
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    doc['interior_energetics']['module'] = stored_module
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    config, path = _user_config(tmp_path)
    config.interior_energetics.module = new_module
    assert (
        config.interior_energetics.aragog.separation_buoyancy == 'mixture'
    )  # differs from the record
    assert check_resume_separation_laws(config, path, out) == []


@pytest.fixture(scope='module')
def stored_template():
    """The init_coupler.toml text of a default run, edited per example below."""
    with tempfile.TemporaryDirectory() as folder:
        out = _stored_run(folder)
        return (out / 'init_coupler.toml').read_text(), _user_config(folder)[0]


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    stored=st.one_of(
        st.sampled_from(['mixture', 'solid', 'melt', 'Mixture', ' mixture', '']),
        st.text(max_size=12),
        st.booleans(),
        st.integers(-3, 3),
        st.floats(allow_nan=False),
    )
)
def test_any_stored_buoyancy_other_than_the_new_string_refuses(stored_template, stored):
    """A stored buoyancy resumes only when it equals the new value as a string; TOML true, 0 and 'Mixture' do not."""
    text, config = stored_template
    doc = tomlkit.parse(text)
    doc['interior_energetics']['aragog']['separation_buoyancy'] = stored
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
        if stored == 'mixture' and isinstance(stored, str):
            assert check_resume_separation_laws(config, MINIMAL, folder) == []
        else:
            with pytest.raises(ResumeSeparationLawError, match='separation_buoyancy is '):
                check_resume_separation_laws(config, MINIMAL, folder)
