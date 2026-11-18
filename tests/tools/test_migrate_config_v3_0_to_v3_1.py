"""Tests for the 3.0 -> 3.1 config migration tool.

Version 3.1 changed the escape scheme, ``atmos_clim.p_top`` and
``atmos_clim.agni.conduction`` defaults. The tool pins each to its 3.0 value
where a file leaves it unset, so a migrated file loads under 3.1 and runs as
it did under 3.0. The tests check that the pins reproduce the 3.0 settings in
use, that stated values are kept, and that a 3.0 file is refused before
migration with the command that fixes it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import tomlkit

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(_TOOLS))

# The loader imports proteus.config; stub the heavy pipeline submodule so the
# test never initialises the Julia-backed atmosphere code.
if 'proteus.proteus' not in sys.modules:
    _stub = types.ModuleType('proteus.proteus')
    _stub.Proteus = object
    sys.modules['proteus.proteus'] = _stub

import migrate_config_v3_0_to_v3_1 as mig  # noqa: E402

from proteus.config import read_config_object  # noqa: E402

_V3_0 = """config_version = "3.0"

[escape]
    module = "zephyrus"
    [escape.zephyrus]
        efficiency = 0.2  # kept
"""


def _write(tmp_path, text, name='old.toml'):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_3_0_file_is_refused_with_the_migration_command(tmp_path):
    """A 3.0 file is refused before any other check, naming the tool that translates it.

    The file also sets reservoir = "outgas", which the dispatcher default
    refuses; the version is reported instead, since that clash is a symptom of
    the version. The message names the three settings the tool pins.
    """
    path = _write(tmp_path, _V3_0.replace('[escape]\n', '[escape]\n    reservoir = "outgas"\n'))
    with pytest.raises(ValueError) as excinfo:
        read_config_object(path)
    message = str(excinfo.value)
    assert 'migrate_config_v3_0_to_v3_1.py' in message
    for setting in ('escape.zephyrus.scheme', 'atmos_clim.p_top', 'atmos_clim.agni.conduction'):
        assert setting in message, setting
    assert 'cannot be combined' not in message


def test_a_file_without_a_version_is_read_as_3_0(tmp_path):
    """A file with no config_version is refused as a 3.0 file, the version that allowed it.

    Without this, a 3.0 file that never stated its version would load under the
    3.1 defaults with no message. The same file with the 3.1 version loads, and
    the migration tool translates the version-less file.
    """
    body = '[escape]\n    module = "zephyrus"\n'
    bare = _write(tmp_path, body, name='bare.toml')
    with pytest.raises(ValueError) as excinfo:
        read_config_object(bare)
    assert 'no config_version, read as "3.0"' in str(excinfo.value)
    assert 'migrate_config_v3_0_to_v3_1.py' in str(excinfo.value)

    stated = _write(tmp_path, 'config_version = "3.1"\n' + body, name='stated.toml')
    assert read_config_object(stated).escape.uses_dispatcher is True

    out = tmp_path / 'bare.v3_1.toml'
    assert mig.main([str(bare), '-o', str(out)]) == 0
    assert read_config_object(out).escape.zephyrus.scheme == 'el'


def test_a_translation_that_would_change_the_file_writes_nothing(tmp_path):
    """Dotted keys and inline tables are refused, since a pin there would change other keys.

    tomlkit adds a new table after dotted keys, which moves them into it, and
    cannot add a sub-table to an inline table. The tool reads its own output
    back and refuses when anything but the version and the pins changed.
    """
    for name, text in (
        (
            'dotted',
            'config_version = "3.0"\nescape.module = "zephyrus"\natmos_clim.module = "janus"\n',
        ),
        ('inline', 'config_version = "3.0"\nescape = { module = "zephyrus" }\n'),
    ):
        src = _write(tmp_path, text, name=f'{name}.toml')
        out = tmp_path / f'{name}.out.toml'
        assert mig.main([str(src), '-o', str(out)]) == 1, name
        assert not out.exists(), name
    assert mig.check('a = 1\n', 'a = 2\n') != []


def test_migrated_file_runs_the_3_0_escape_and_atmosphere(tmp_path):
    """The migrated file loads under 3.1 with every 3.0 setting in use.

    3.0 ran energy-limited escape with the outgas reservoir, the Hill clip on,
    the tidal factor off, a 1e-6 bar top, and AGNI conduction off. A value the
    file stated (the efficiency) and its comment survive.
    """
    src = _write(tmp_path, _V3_0)
    out = tmp_path / 'new.toml'
    assert mig.main([str(src), '-o', str(out)]) == 0
    text = out.read_text()
    assert '# kept' in text

    cfg = read_config_object(out)
    assert cfg.config_version == '3.1'
    esc = cfg.escape
    assert esc.zephyrus.scheme == 'el' and esc.uses_dispatcher is False
    assert esc.effective_reservoir == 'outgas'
    assert esc.effective_hill_clamp is True
    assert esc.zephyrus.effective_tidal is False
    assert esc.zephyrus.efficiency == pytest.approx(0.2, rel=1e-12)
    assert cfg.atmos_clim.p_top == pytest.approx(1.0e-6, rel=1e-12)
    assert cfg.atmos_clim.agni.conduction is False


def test_stated_values_are_never_replaced():
    """A file that already states the scheme, the top, and conduction keeps them."""
    doc = tomlkit.parse(
        'config_version = "3.0"\n'
        '[escape.zephyrus]\nscheme = "dispatcher"\n'
        '[atmos_clim]\np_top = 2e-4\n[atmos_clim.agni]\nconduction = true\n'
    )
    assert mig.migrate(doc) == []
    assert doc['escape']['zephyrus']['scheme'] == 'dispatcher'
    assert doc['atmos_clim']['p_top'] == pytest.approx(2.0e-4, rel=1e-12)
    assert doc['atmos_clim']['agni']['conduction'] is True
    assert doc['config_version'] == '3.1'


def test_a_grid_file_only_changes_its_version():
    """A grid file takes no pins: they belong in its reference config."""
    doc = tomlkit.parse('config_version = "3.0"\nref_config = "base.toml"\n')
    assert mig.migrate(doc) == []
    assert doc['config_version'] == '3.1'
    assert 'escape' not in doc and 'atmos_clim' not in doc


def test_the_tool_refuses_other_versions_and_never_overwrites(tmp_path):
    """Only a 3.0 file is translated, and neither the input nor an existing file is replaced."""
    src = _write(tmp_path, _V3_0)
    before = src.read_text()
    assert mig.main([str(src), '-o', str(src)]) == 1
    taken = _write(tmp_path, 'x = 1\n', name='taken.toml')
    assert mig.main([str(src), '-o', str(taken)]) == 1
    assert src.read_text() == before and taken.read_text() == 'x = 1\n'

    newer = _write(tmp_path, 'config_version = "3.1"\n', name='newer.toml')
    assert mig.main([str(newer), '-o', str(tmp_path / 'out.toml')]) == 1
    assert not (tmp_path / 'out.toml').exists()
