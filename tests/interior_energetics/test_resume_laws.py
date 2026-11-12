"""
Unit tests for proteus.interior_energetics.resume_laws: the check that a resumed
aragog run keeps the separation laws it was written with.

Invariant: a resumed run continues under one separation law. The stored laws are the
``interior_energetics.aragog`` fields of the run's ``init_coupler.toml``; a stored value
that differs from the new config refuses the resume, unless
``allow_resume_law_change`` is set. An aragog record without ``separation_viscosity``
counts as ``'melt'``; any other law the run did not record, and both laws of a record of
another interior module, resume only when the new config file sets the field.
"""

from __future__ import annotations

import logging
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
    """The same record and the laws it holds resume with no switch and no note; its Aragog
    table holds no cell fields, so the run was on the uniform mesh, which the resume keeps."""
    out = _record_without_module(tmp_path)
    config, path = _user_config(
        tmp_path,
        separation_buoyancy='solid',
        separation_viscosity='mixture',
        cmb_cell_thickness=0.0,
        surface_cell_thickness=0.0,
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


def test_two_switched_laws_give_one_warning_each(tmp_path, caplog):
    """With allow_resume_law_change both laws can switch at one resume: both switches come
    back and each gets its own warning naming its key."""
    out = _stored_run(tmp_path, separation_viscosity='melt', separation_buoyancy='solid')
    config, path = _user_config(tmp_path, allow_resume_law_change=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        switches = check_resume_separation_laws(config, path, out)
    assert sorted(switches) == [
        ('separation_buoyancy', 'solid', 'mixture'),
        ('separation_viscosity', 'melt', 'mixture'),
    ]
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warned) == 2
    assert sum('separation_viscosity' in w for w in warned) == 1
    assert sum('separation_buoyancy' in w for w in warned) == 1


def test_a_record_whose_module_is_not_a_name_names_no_module(tmp_path):
    """A record with a non-string interior module, such as module = 1, is read as naming no
    interior module."""
    out = _stored_run(tmp_path, drop=('separation_buoyancy',))
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    doc['interior_energetics']['module'] = 1
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    assert 'names no interior module' in str(err.value)
    assert 'interior module 1' not in str(err.value)


def test_a_resume_that_does_not_use_aragog_is_not_checked(tmp_path):
    """A resume under another interior module leaves the aragog laws unchecked."""
    out = _stored_run(tmp_path, separation_buoyancy='solid')
    config, path = _user_config(tmp_path)
    config.interior_energetics.module = 'spider'
    assert config.interior_energetics.aragog.separation_buoyancy == 'mixture'  # differs
    assert check_resume_separation_laws(config, path, out) == []


@pytest.mark.parametrize('stored_module', ['dummy', 'spider'])
def test_a_record_of_another_interior_module_holds_no_aragog_law(tmp_path, stored_module):
    """An Aragog resume of a record written under another interior module, such as a dummy
    resume between two Aragog segments, takes both laws as unrecorded: the config file
    must set them, and the aragog values in that record are not compared."""
    out = _stored_run(tmp_path, separation_buoyancy='solid')
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    doc['interior_energetics']['module'] = stored_module
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError) as err:
        check_resume_separation_laws(config, path, out)
    message = str(err.value)
    assert 'separation_viscosity is not recorded' in message
    assert 'separation_buoyancy is not recorded' in message
    assert f"names interior module '{stored_module}'" in message
    config, path = _user_config(
        tmp_path, separation_viscosity='mixture', separation_buoyancy='mixture'
    )
    assert check_resume_separation_laws(config, path, out) == []


def test_an_allowed_switch_recorded_by_a_real_write_resumes_under_the_new_law(tmp_path):
    """An allowed switch, recorded by the real Config.write with RECORD_OVERRIDES as start()
    does, lets the next resume run under the new law without the switch and refuses the old
    law again."""
    out = _stored_run(tmp_path)
    config, path = _user_config(
        tmp_path, separation_buoyancy='solid', allow_resume_law_change=True
    )
    assert check_resume_separation_laws(config, path, out) == [
        ('separation_buoyancy', 'mixture', 'solid')
    ]
    config.write(out / 'init_coupler.toml', overrides=RECORD_OVERRIDES)
    config, path = _user_config(tmp_path, separation_buoyancy='solid')
    assert check_resume_separation_laws(config, path, out) == []
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError, match="written with 'solid'"):
        check_resume_separation_laws(config, path, out)


def test_a_record_that_names_no_module_but_holds_the_laws_needs_no_law_in_the_file(tmp_path):
    """A record without interior_energetics.module that holds both laws is compared like any
    record, so a config file that sets neither law resumes when its defaults match. The record
    holds no cell fields (the uniform mesh), which the file keeps."""
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'init_coupler.toml').write_text(
        "[interior_energetics.aragog]\nseparation_viscosity = 'mixture'\n"
        "separation_buoyancy = 'mixture'\n"
    )
    config, path = _user_config(tmp_path, cmb_cell_thickness=0.0, surface_cell_thickness=0.0)
    assert 'separation_' not in path.read_text()  # the file sets neither law
    assert check_resume_separation_laws(config, path, out) == []


@pytest.fixture(scope='module')
def stored_template(tmp_path_factory):
    """The init_coupler.toml text of a default run, edited per example below."""
    folder = tmp_path_factory.mktemp('template')
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
def test_any_stored_buoyancy_other_than_the_new_string_refuses(
    stored_template, tmp_path, stored
):
    """A stored buoyancy resumes only when it equals the new value as a string; TOML true, 0 and 'Mixture' do not."""
    text, config = stored_template
    doc = tomlkit.parse(text)
    doc['interior_energetics']['aragog']['separation_buoyancy'] = stored
    (tmp_path / 'init_coupler.toml').write_text(tomlkit.dumps(doc))  # rewritten per example
    if stored == 'mixture' and isinstance(stored, str):
        assert check_resume_separation_laws(config, MINIMAL, tmp_path) == []
    else:
        with pytest.raises(ResumeSeparationLawError, match='separation_buoyancy is '):
            check_resume_separation_laws(config, MINIMAL, tmp_path)


def _rheology_run(folder, enabled):
    """A run directory recording the rheology switch, or no rheology table when None."""
    out = Path(folder) / 'run'
    out.mkdir()
    config, _ = _user_config(out)
    config.write(out / 'init_coupler.toml')
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    if enabled is None:
        del doc['interior_energetics']['aragog']['rheology']
    else:
        doc['interior_energetics']['aragog']['rheology']['enabled'] = enabled
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    return out


@pytest.mark.parametrize('stored', [False, None])
def test_resume_refuses_to_turn_the_rheology_on(tmp_path, stored):
    """A record with the rheology off, or with no rheology table, refuses a resume with it on."""
    out = _rheology_run(tmp_path, stored)
    text = MINIMAL.read_text() + '\n[interior_energetics.aragog.rheology]\n    enabled = true\n'
    path = tmp_path / 'user.toml'
    path.write_text(text)
    with pytest.raises(ResumeSeparationLawError, match='rheology.enabled') as err:
        check_resume_separation_laws(read_config_object(path), path, out)
    assert 'allow_resume_law_change' in str(err.value)


def test_resume_refuses_to_turn_the_rheology_off(tmp_path):
    """A run written with the rheology on keeps it at a resume."""
    out = _rheology_run(tmp_path, True)
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError, match='rheology.enabled') as err:
        check_resume_separation_laws(config, path, out)
    assert 'written with True' in str(err.value)


def test_resume_keeps_an_unchanged_rheology(tmp_path, caplog):
    """The same rheology switch on both sides gives no switch and no warning."""
    out = _rheology_run(tmp_path, False)
    config, path = _user_config(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert [
        r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING
    ] == []


def test_the_allow_switch_lets_the_rheology_switch_and_reports_it(tmp_path, caplog):
    """With allow_resume_law_change a run written with the rheology on resumes with it off,
    returns the switch and logs one warning naming the key and both values."""
    out = _rheology_run(tmp_path, True)
    config, path = _user_config(tmp_path, allow_resume_law_change=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        switches = check_resume_separation_laws(config, path, out)
    assert switches == [('rheology.enabled', True, False)]
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warned) == 1
    assert 'rheology.enabled' in warned[0] and 'True' in warned[0] and 'False' in warned[0]


def _lid_case(folder, stored_lid, new_lid, allow=False):
    """A run directory recording lid_mush_base (no field when None) and a resume config file,
    both with the rheology on, so that only the lid switch differs."""
    out = _rheology_run(folder, True)
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    if stored_lid is None:
        doc['interior_energetics']['aragog'].pop('lid_mush_base', None)
    else:
        doc['interior_energetics']['aragog']['lid_mush_base'] = stored_lid
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    text = MINIMAL.read_text() + (
        '\n[interior_energetics.aragog]\n'
        f'    lid_mush_base = {str(new_lid).lower()}\n'
        f'    allow_resume_law_change = {str(allow).lower()}\n'
        '[interior_energetics.aragog.rheology]\n    enabled = true\n'
    )
    path = Path(folder) / 'user.toml'
    path.write_text(text)
    return read_config_object(path), path, out


@pytest.mark.parametrize(('stored', 'new'), [(True, False), (False, True), (None, True)])
def test_resume_refuses_to_switch_the_lid(tmp_path, stored, new):
    """A run keeps its lid switch at a resume; a record without the field ran without the lid."""
    config, path, out = _lid_case(tmp_path, stored, new)
    with pytest.raises(ResumeSeparationLawError, match='lid_mush_base') as err:
        check_resume_separation_laws(config, path, out)
    assert f'written with {bool(stored)!r}' in str(err.value)


def test_the_allow_switch_lets_the_lid_switch_and_reports_it(tmp_path, caplog):
    """With allow_resume_law_change a run written with the lid resumes without it, returns the
    switch and logs one warning naming the key."""
    config, path, out = _lid_case(tmp_path, True, False, allow=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == [
            ('lid_mush_base', True, False)
        ]
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warned) == 1 and 'lid_mush_base' in warned[0]


def test_resume_keeps_an_unchanged_lid(tmp_path, caplog):
    """The same lid switch on both sides gives no switch and no warning."""
    config, path, out = _lid_case(tmp_path, True, True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert [
        r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING
    ] == []


@pytest.mark.parametrize(
    'content',
    [None, b'\x00 not toml [', b"[interior]\nmodule = 'aragog'\n"],
    ids=['no file', 'unreadable file', 'record in the [interior] layout'],
)
def test_a_run_without_a_usable_record_does_not_check_the_rheology_or_the_lid(
    tmp_path, caplog, content
):
    """With no record of the Aragog table the rheology and lid switches are not compared with
    a guessed value: the resume with both on goes ahead and logs one note for each."""
    out = tmp_path / 'run'
    out.mkdir()
    if content is not None:
        (out / 'init_coupler.toml').write_bytes(content)
    text = MINIMAL.read_text() + (
        '\n[interior_energetics.aragog]\n'
        "    separation_viscosity = 'mixture'\n    separation_buoyancy = 'mixture'\n"
        '    lid_mush_base = true\n'
        '[interior_energetics.aragog.rheology]\n    enabled = true\n'
    )
    path = tmp_path / 'user.toml'
    path.write_text(text)
    config = read_config_object(path)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    noted = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    for key in ('rheology.enabled', 'lid_mush_base'):
        mine = [m for m in noted if key in m]
        assert len(mine) == 1 and 'not recorded' in mine[0] and 'True' in mine[0]


def _mesh_run(folder, drop=(), **aragog):
    """A run directory whose record holds the given aragog fields, minus ``drop``."""
    return _stored_run(folder, drop=drop, **aragog)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('allow', [False, True])
@pytest.mark.parametrize(
    'field, stored, new',
    [
        ('cmb_cell_thickness', 1000.0, 0.0),
        ('cmb_cell_thickness', 1000.0, 500.0),
        ('surface_cell_thickness', 'uniform', 0.0),
        ('surface_cell_thickness', 20000.0, 'uniform'),
    ],
)
def test_a_changed_mesh_end_cell_refuses_the_resume_even_when_laws_may_change(
    tmp_path, allow, field, stored, new
):
    """The snapshot entropy is put back node by node, so a resume on other end cells would
    move every node: refused, also with allow_resume_law_change."""
    out = _mesh_run(tmp_path, **{field: stored})
    config, path = _user_config(tmp_path, **{field: new, 'allow_resume_law_change': allow})
    with pytest.raises(ResumeSeparationLawError, match=field) as err:
        check_resume_separation_laws(config, path, out)
    assert 'node' in str(err.value)


def test_unchanged_end_cells_resume_quietly(tmp_path, caplog):
    """The default end cells on both sides give no switch and no warning."""
    out = _mesh_run(tmp_path)
    config, path = _user_config(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == []
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_a_record_without_the_end_cells_ran_on_the_uniform_mesh(tmp_path):
    """A record that names the Aragog module and lacks both cell fields ran on the uniform
    mesh: a resume under the new defaults is refused, one with both cells at 0 goes on."""
    out = _mesh_run(tmp_path, drop=('cmb_cell_thickness', 'surface_cell_thickness'))
    config, path = _user_config(tmp_path)
    with pytest.raises(ResumeSeparationLawError, match='cmb_cell_thickness') as err:
        check_resume_separation_laws(config, path, out)
    assert 'surface_cell_thickness' in str(err.value)
    config, path = _user_config(tmp_path, cmb_cell_thickness=0.0, surface_cell_thickness=0.0)
    assert check_resume_separation_laws(config, path, out) == []


def test_uniform_and_0_at_the_surface_with_no_cmb_cell_are_the_same_mesh(tmp_path):
    """With the CMB cell at 0, "uniform" and 0 at the surface both give the uniform mesh, so a
    resume may change one into the other."""
    out = _mesh_run(tmp_path, cmb_cell_thickness=0.0, surface_cell_thickness=0.0)
    config, path = _user_config(tmp_path, cmb_cell_thickness=0.0)
    assert config.interior_energetics.aragog.surface_cell_thickness == 'uniform'
    assert check_resume_separation_laws(config, path, out) == []


def test_a_run_without_a_usable_record_warns_about_the_end_cells(tmp_path, caplog):
    """With no record of the Aragog table the end cells of the run are unknown: the resume
    goes on and warns once for each cell field."""
    out = tmp_path / 'run'
    out.mkdir()
    text = MINIMAL.read_text() + (
        '\n[interior_energetics.aragog]\n'
        "    separation_viscosity = 'mixture'\n    separation_buoyancy = 'mixture'\n"
    )
    path = tmp_path / 'user.toml'
    path.write_text(text)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(read_config_object(path), path, out) == []
    noted = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    for key in ('cmb_cell_thickness', 'surface_cell_thickness'):
        assert len([m for m in noted if key in m and 'not recorded' in m]) == 1


def _slope_case(folder, stored, new, allow=False):
    """A rheology-on run recording slope ``stored`` (None: no slope field) and its resume."""
    out = Path(folder) / 'run'
    out.mkdir()
    config, _ = _user_config(out)
    config.write(out / 'init_coupler.toml')
    doc = tomlkit.parse((out / 'init_coupler.toml').read_text())
    rheology = {'enabled': True}
    if stored is not None:
        rheology['mlt_bottom_slope'] = stored
    doc['interior_energetics']['aragog']['rheology'] = rheology
    (out / 'init_coupler.toml').write_text(tomlkit.dumps(doc))
    text = MINIMAL.read_text() + (
        f'\n[interior_energetics.aragog]\n    allow_resume_law_change = {str(allow).lower()}\n'
        f'[interior_energetics.aragog.rheology]\n    enabled = true\n    mlt_bottom_slope = {new}\n'
    )
    path = Path(folder) / 'user.toml'
    path.write_text(text)
    return read_config_object(path), path, out


@pytest.mark.parametrize('stored, new', [(0.82, 1.0), (1.0, 0.82), (None, 0.82)])
def test_a_changed_slope_refuses_the_resume(tmp_path, stored, new):
    """The slope sets the viscous mixing length at the CMB; a resume that changes it is
    refused. A record without the slope ran under 1.0."""
    config, path, out = _slope_case(tmp_path, stored, new)
    with pytest.raises(ResumeSeparationLawError, match='mlt_bottom_slope') as err:
        check_resume_separation_laws(config, path, out)
    assert f'written with {1.0 if stored is None else stored!r}' in str(err.value)


def test_the_allow_switch_lets_the_slope_change_and_reports_it(tmp_path, caplog):
    """With allow_resume_law_change the slope switches with one warning naming both values."""
    config, path, out = _slope_case(tmp_path, 1.0, 0.82, allow=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert check_resume_separation_laws(config, path, out) == [
            ('rheology.mlt_bottom_slope', 1.0, 0.82)
        ]
    warned = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(warned) == 1 and '1.0' in warned[0] and '0.82' in warned[0]


def test_the_slope_is_not_compared_with_the_rheology_off(tmp_path):
    """With the rheology off the slope is not read, so a change does not stop a resume."""
    out = _stored_run(tmp_path)
    text = (
        MINIMAL.read_text()
        + '\n[interior_energetics.aragog.rheology]\n    mlt_bottom_slope = 0.5\n'
    )
    path = tmp_path / 'user.toml'
    path.write_text(text)
    assert check_resume_separation_laws(read_config_object(path), path, out) == []
