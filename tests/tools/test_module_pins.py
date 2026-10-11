"""Unit tests for ``tools/_module_pins.py``.

The tool resolves how an external module is pinned in ``pyproject.toml``.
These tests cover the two distribution fields the installer reads:

* ``floor``: the ``>=`` version floor of a distribution, read from
  ``[project] dependencies`` and every optional-dependency list, and the
  refusal of a requirement without a single ``>=`` floor or of a name that
  is not required at all,
* ``floor-tag``: the one release tag whose number fields equal the floor,
  compared as integers (``26.10.6`` selects ``26.10.06``), with a leading
  ``v`` and trailing zero fields ignored, and the refusal of no match or of
  more than one,

that the ``<module> <field>`` lookup of ``[tool.proteus.modules]`` keeps
working and names the distribution fields for a distribution, and that a
``pyproject.toml`` that is not valid TOML is reported in one line. The command-line cases run a copy of the tool beside a
fixture ``pyproject.toml``, which it finds by walking up from its own file.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

TOOL = Path(__file__).resolve().parents[2] / 'tools' / '_module_pins.py'

# One requirement per case the floor rule distinguishes; pytest is written
# with spaces around the operator, as the real pyproject.toml does.
FIXTURE_PYPROJECT = """\
[project]
name = "fixture"
dependencies = [
    "fwl-mors>=26.10.6",
    "fwl-janus>=26.10.08,<27",
    "equinox==0.13.8",
    "compat~=1.4",
    "strict>2.0",
    "ruff",
    "pytest >= 8.1",
]

[project.optional-dependencies]
morrigan = ["fwl-morrigan>=26.07.27"]

[tool.proteus.modules.agni]
url = "https://example.invalid/AGNI.git"
ref = "0123abcd"
"""


def _load_tool():
    """Import the tool by path, since ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location('_module_pins', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pins = _load_tool()


@pytest.fixture
def tool_copy(tmp_path) -> Path:
    """A copy of the tool in ``tools/`` below a fixture ``pyproject.toml``."""
    (tmp_path / 'tools').mkdir()
    shutil.copy2(TOOL, tmp_path / 'tools' / TOOL.name)
    (tmp_path / 'pyproject.toml').write_text(FIXTURE_PYPROJECT)
    return tmp_path / 'tools' / TOOL.name


def _run(tool: Path, *args: str, stdin: str = '') -> subprocess.CompletedProcess:
    """Run the tool with ``args`` and ``stdin``; return the finished process."""
    return subprocess.run(
        [sys.executable, str(tool), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=20,
    )


class TestFloor:
    """``<distribution> floor`` prints the ``>=`` floor or exits 1."""

    @pytest.mark.parametrize(
        ('distribution', 'floor'),
        [
            ('fwl-mors', '26.10.6'),
            ('fwl-janus', '26.10.08'),
            ('fwl-morrigan', '26.07.27'),
            ('FWL_Mors', '26.10.6'),
            ('pytest', '8.1'),
        ],
        ids=[
            'plain lower bound',
            'lower bound of a range',
            'optional dependency list',
            'name written unnormalised',
            'spaces around the operator',
        ],
    )
    def test_prints_the_lower_bound_as_written(self, tool_copy, distribution, floor):
        """The floor is printed as written in pyproject.toml, not normalised.

        The range case discriminates against taking the last version of the
        requirement (27), and 26.10.6 against a zero-padded rewrite.
        """
        result = _run(tool_copy, distribution, 'floor')
        assert result.returncode == 0, result.stderr
        assert result.stdout == f'{floor}\n'
        assert result.stderr == ''

    @pytest.mark.parametrize(
        ('distribution', 'specifier'),
        [('equinox', '==0.13.8'), ('compat', '~=1.4'), ('strict', '>2.0'), ('ruff', 'none')],
        ids=['exact pin', 'compatible release', 'strict lower bound', 'no specifier'],
    )
    def test_requirement_without_lower_bound_exits_1(self, tool_copy, distribution, specifier):
        """Only ``>=`` is a floor; any other requirement names itself and fails.

        ``>2.0`` is the case a match on ``>`` alone would accept, and ``~=1.4``
        the one that carries an implied lower bound the installer must not guess.
        """
        result = _run(tool_copy, distribution, 'floor')
        assert result.returncode == 1
        assert result.stdout == ''
        assert distribution in result.stderr
        assert f'specifier: {specifier}' in result.stderr

    def test_unknown_distribution_exits_1(self, tool_copy):
        """A distribution that pyproject.toml does not require is named and refused."""
        result = _run(tool_copy, 'fwl-nothing', 'floor')
        assert result.returncode == 1
        assert result.stdout == ''
        assert "'fwl-nothing' is not a dependency in pyproject.toml" in result.stderr

    def test_two_different_floors_for_one_distribution_exit_1(self):
        """A distribution required twice with different floors has no single floor."""
        data = {
            'project': {
                'dependencies': ['fwl-mors>=26.10.6'],
                'optional-dependencies': {'old': ['fwl-mors>=26.7.1']},
            }
        }
        with pytest.raises(LookupError, match='different floors') as err:
            pins.find_floor(data, 'fwl-mors')
        assert 'fwl-mors>=26.10.6' in str(err.value)
        assert 'fwl-mors>=26.7.1' in str(err.value)

    def test_module_field_lookup_is_unchanged(self, tool_copy):
        """``<module> <field>`` still reads ``[tool.proteus.modules]``.

        A distribution field must not shadow it, and a missing field still
        exits 1 with the available fields listed.
        """
        ok = _run(tool_copy, 'agni', 'ref')
        assert ok.returncode == 0
        assert ok.stdout == '0123abcd\n'
        missing = _run(tool_copy, 'agni', 'sha256')
        assert missing.returncode == 1
        assert 'Available: ref, url' in missing.stderr

    def test_distribution_with_a_module_field_names_the_distribution_fields(self, tool_copy):
        """A distribution asked for a module field is told its fields, not called unknown.

        ``FWL_Mors`` is matched in its normalised form; a name that is
        neither a module nor a distribution still lists the known modules.
        """
        for name in ('fwl-mors', 'FWL_Mors'):
            result = _run(tool_copy, name, 'tag')
            assert result.returncode == 1
            assert result.stdout == ''
            assert (
                f"'{name}' is a distribution in pyproject.toml; "
                'the fields of a distribution are floor and floor-tag'
            ) in result.stderr
            assert 'Unknown module' not in result.stderr
        unknown = _run(tool_copy, 'fwl-nothing', 'tag')
        assert unknown.returncode == 1
        assert "Unknown module 'fwl-nothing'. Known: agni" in unknown.stderr

    def test_pyproject_syntax_error_exits_1_in_one_line(self, tool_copy):
        """A pyproject.toml that is no valid TOML is reported in one line naming the file,
        with exit 1 and no traceback, for a floor and for a module field."""
        pyproject = tool_copy.parents[1] / 'pyproject.toml'
        pyproject.write_text('[project\nname = "fixture"\n')
        for args in (('fwl-mors', 'floor'), ('agni', 'ref')):
            result = _run(tool_copy, *args)
            assert result.returncode == 1
            assert result.stdout == ''
            assert 'Traceback' not in result.stderr
            assert len(result.stderr.splitlines()) == 1
            assert result.stderr.startswith(f'cannot parse {pyproject}: ')
            assert 'line 1' in result.stderr

    def test_wrong_argument_count_prints_usage(self, tool_copy):
        """A wrong argument count exits 2 with usage naming both field kinds."""
        result = _run(tool_copy, 'fwl-mors')
        assert result.returncode == 2
        assert '<module> <field>' in result.stderr
        assert '<distribution> floor-tag' in result.stderr


class TestFloorTag:
    """``<distribution> floor-tag`` picks the one tag equal to the floor."""

    @pytest.mark.parametrize(
        ('floor', 'tags', 'expected'),
        [
            ('26.10.6', ['26.07.23', '26.09.23', '26.10.06'], '26.10.06'),
            ('26.7.24', ['26.07.10', '26.07.24', '26.10.10'], '26.07.24'),
            ('0.2.6', ['v0.2.5', 'v0.2.6', 'v0.2.6-alpha'], 'v0.2.6'),
            ('26.10', ['26.1', '26.10.0', '26.100'], '26.10.0'),
        ],
        ids=['MORS floor', 'ZEPHYRUS floor', 'v prefix', 'trailing zero field'],
    )
    def test_matches_number_fields_as_integers(self, floor, tags, expected):
        """Number fields compare as integers, so a zero-padded tag matches.

        In the two real cases the floor text is not among the tags, so a
        text match finds nothing. ``26.1`` and ``26.100`` beside ``26.10.0``
        catch a comparison that strips zeros from the text instead.
        """
        assert floor not in tags
        assert pins.match_floor_tag('fwl-x', floor, tags) == expected

    def test_tags_that_are_not_release_numbers_are_ignored(self):
        """Suffixed and prefixed tags are no candidates, so they cannot match twice."""
        tags = ['26.10.06rc1', 'v26.10.6-alpha', 'release-26.10.6', 'vv26.10.6', '', '26.10.06']
        assert pins.match_floor_tag('fwl-mors', '26.10.6', tags) == '26.10.06'
        with pytest.raises(LookupError, match='saw no release tags'):
            pins.match_floor_tag('fwl-mors', '26.10.6', tags[:-1])

    def test_no_match_names_distribution_floor_and_tags(self):
        """No match fails with the distribution, the floor and the 10 highest tags."""
        tags = [f'26.{month:02d}.01' for month in range(1, 13)] + ['v1.0']
        with pytest.raises(LookupError) as err:
            pins.match_floor_tag('fwl-mors', '26.10.6', tags)
        message = str(err.value)
        assert "'fwl-mors' floor 26.10.6: no tag matches it" in message
        assert 'release tags seen: 13' in message
        # Highest first: 26.12.01 down to 26.03.01, so the 3 lowest are cut.
        assert 'highest: 26.12.01, 26.11.01, 26.10.01' in message
        assert '26.03.01' in message
        assert '26.02.01' not in message and 'v1.0' not in message

    def test_two_matches_exit_1_naming_both(self):
        """Two tags with one version are ambiguous, so neither is checked out."""
        with pytest.raises(LookupError) as err:
            pins.match_floor_tag('fwl-mors', '26.10.6', ['26.10.6', '26.10.06', '26.10.07'])
        message = str(err.value)
        assert '2 tags match it: 26.10.06, 26.10.6' in message
        assert 'fwl-mors' in message

    def test_floor_that_is_not_a_release_number_exits_1(self):
        """A pre-release floor cannot be matched to a tag and is refused."""
        with pytest.raises(LookupError, match='not a plain release number'):
            pins.match_floor_tag('fwl-x', '1.0.2rc1', ['1.0.2rc1', '1.0.2'])
        assert pins._release_key('1.0.2rc1') is None
        assert pins._release_key('v26.10.0') == (26, 10)

    def test_reads_tags_from_stdin(self, tool_copy):
        """The command reads one tag per line, blank lines and spaces included."""
        result = _run(tool_copy, 'fwl-mors', 'floor-tag', stdin='26.07.23\n\n 26.10.06 \n')
        assert result.returncode == 0, result.stderr
        assert result.stdout == '26.10.06\n'

    def test_empty_stdin_exits_1(self, tool_copy):
        """No tags at all (a clone without tags) fails and names the floor."""
        result = _run(tool_copy, 'fwl-mors', 'floor-tag', stdin='')
        assert result.returncode == 1
        assert result.stdout == ''
        assert (
            "'fwl-mors' floor 26.10.6: no tag matches it; saw no release tags" in result.stderr
        )

    def test_requirement_without_floor_exits_1_before_reading_tags(self, tool_copy):
        """floor-tag applies the floor rule first, so an exact pin is refused."""
        result = _run(tool_copy, 'equinox', 'floor-tag', stdin='0.13.8\n')
        assert result.returncode == 1
        assert result.stdout == ''
        assert 'specifier: ==0.13.8' in result.stderr
