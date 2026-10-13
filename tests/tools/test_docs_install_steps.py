"""Unit tests for ``tools/docs_install_steps.py``.

The tool turns the install page into the commands a user types for one
platform tab and into a bash script that runs them in one shell session.
These tests check:

* tab selection: shared blocks run on every tab, tab blocks only on their tab,
  and an unknown tab exits 2,
* admonition blocks are skipped from the plan and listed on request,
* page line numbers of the commands, including backslash continuations,
* a substitution applies, and a stale substitution exits 2,
* the generated script runs in one shell session, writes the timing TSV and
  stops at the first non-zero exit code.

See ``docs/How-to/testing.md`` for the test framework.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / 'tools' / 'docs_install_steps.py'

_spec = importlib.util.spec_from_file_location('docs_install_steps', _TOOL)
steps = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules['docs_install_steps'] = steps
_spec.loader.exec_module(steps)

_PAGE = """\
# Install

```console
echo shared-one
```

=== "Alpha"

    ```console
    echo alpha-one
    echo alpha-two \\
      --long
    ```

=== "Beta"

    ```console
    echo beta-one   # a trailing comment
    ```

!!! note "Optional"
    ```console
    echo optional-shared
    ```

=== "Alpha"

    !!! tip "Optional for Alpha"
        ```console
        echo optional-alpha
        ```

```python
print("not a console block")
```

```console
$ echo shared-two
```
"""


@pytest.fixture
def page(tmp_path: Path) -> Path:
    """Write the fixture page and return its path."""
    path = tmp_path / 'page.md'
    path.write_text(_PAGE, encoding='utf-8')
    return path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the tool as a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, str(_TOOL), *args], capture_output=True, text=True, check=False
    )


def _write_subs(tmp_path: Path, match: str, replace: str = 'true') -> Path:
    """Write a one-entry substitution table and return its path."""
    path = tmp_path / 'subs.toml'
    path.write_text(
        f'[[substitution]]\nmatch = "{match}"\nreplace = "{replace}"\nreason = "test"\n',
        encoding='utf-8',
    )
    return path


def test_tab_selection_keeps_shared_blocks_and_selected_tab_only(page):
    """Alpha gets the shared blocks and its own, in page order, and no Beta command."""
    out = _run(str(page), '--tab', 'Alpha')
    assert out.returncode == 0
    assert out.stdout.splitlines() == [
        'echo shared-one',
        'echo alpha-one',
        'echo alpha-two --long',
        'echo shared-two',
    ]
    assert 'beta-one' not in out.stdout


def test_other_tab_does_not_see_first_tab_commands(page):
    """Beta gets its own command and the shared ones, none of Alpha's."""
    out = _run(str(page), '--tab', 'Beta')
    assert out.returncode == 0
    assert 'echo beta-one   # a trailing comment' in out.stdout.splitlines()
    assert 'alpha' not in out.stdout


def test_unknown_tab_exits_2_and_names_known_tabs(page):
    """A tab name that is not on the page is a structural error."""
    out = _run(str(page), '--tab', 'Gamma')
    assert out.returncode == 2
    assert 'unknown tab' in out.stderr
    assert 'Alpha' in out.stderr and 'Beta' in out.stderr
    assert out.stdout == ''


def test_page_without_console_block_exits_2(tmp_path):
    """A page with no console block cannot yield a plan."""
    path = tmp_path / 'empty.md'
    path.write_text('# Title\n\nNo commands here.\n\n```python\nx = 1\n```\n', encoding='utf-8')
    out = _run(str(path))
    assert out.returncode == 2
    assert 'no console block' in out.stderr


def test_admonition_blocks_are_skipped_and_listed(page):
    """Admonition blocks are optional: not in the plan, listed with --list-skipped."""
    plan = _run(str(page), '--tab', 'Alpha').stdout
    assert 'optional' not in plan
    listing = _run(str(page), '--tab', 'Alpha', '--list-skipped').stdout.splitlines()
    assert len(listing) == 2
    assert listing[0].endswith('(note) echo optional-shared')
    assert listing[1].endswith('(tip) echo optional-alpha')
    # An admonition under the other tab's header does not leak into Beta's list.
    beta = _run(str(page), '--tab', 'Beta', '--list-skipped').stdout
    assert 'optional-alpha' not in beta
    assert 'optional-shared' in beta


def test_line_numbers_are_page_lines_of_each_command(page):
    """Each command keeps the page line it starts on, also after a continuation."""
    required, _ = steps.select_steps(steps.parse_page(_PAGE), 'Alpha')
    lines = _PAGE.splitlines()
    for step in required:
        assert lines[step.line - 1].strip().lstrip('$ ').startswith(step.command.split(' ')[0])
    assert [s.line for s in required] == [4, 10, 11, 38]
    assert lines[9].strip() == 'echo alpha-one'
    assert lines[10].strip() == 'echo alpha-two \\'
    assert lines[37].strip() == '$ echo shared-two'


def test_substitution_replaces_command_and_table_shows_deviation(page, tmp_path):
    """A matching substitution changes the job command and shows its reason."""
    subs = _write_subs(tmp_path, 'echo alpha-one', 'echo replaced')
    out = _run(str(page), '--tab', 'Alpha', '--substitutions', str(subs), '--format', 'table')
    assert out.returncode == 0
    assert '| 10 | `echo alpha-one` | `echo replaced` | test |' in out.stdout
    assert '| 4 | `echo shared-one` | as written |  |' in out.stdout


def test_substitution_matches_command_with_its_page_comment(page, tmp_path):
    """The match may be the command text as on the page, trailing comment included."""
    subs = _write_subs(tmp_path, 'echo beta-one   # a trailing comment', 'echo replaced')
    out = _run(str(page), '--tab', 'Beta', '--substitutions', str(subs), '--format', 'table')
    assert out.returncode == 0
    assert '`echo replaced`' in out.stdout
    assert '(page comment: `# a trailing comment`)' in out.stdout


def test_stale_substitution_exits_2(page, tmp_path):
    """A match that is not on the page fails the run, so a page change is caught."""
    subs = _write_subs(tmp_path, 'echo alpha-one-renamed')
    out = _run(str(page), '--tab', 'Alpha', '--substitutions', str(subs), '--format', 'script')
    assert out.returncode == 2
    assert 'matches no command on the page' in out.stderr
    assert out.stdout == ''


def test_substitution_for_other_tab_command_is_stale(page, tmp_path):
    """A match that exists only on another tab is stale for the selected tab."""
    subs = _write_subs(tmp_path, 'echo beta-one   # a trailing comment')
    out = _run(str(page), '--tab', 'Alpha', '--substitutions', str(subs))
    assert out.returncode == 2


def test_substitution_missing_reason_exits_2(page, tmp_path):
    """An entry without a reason is rejected, so every deviation is explained."""
    path = tmp_path / 'subs.toml'
    path.write_text('[[substitution]]\nmatch = "echo alpha-one"\nreplace = "true"\n')
    out = _run(str(page), '--tab', 'Alpha', '--substitutions', str(path))
    assert out.returncode == 2
    assert "'reason'" in out.stderr


def _bash_script(page: Path, tmp_path: Path, tab: str | None, subs: Path | None = None) -> Path:
    """Generate the script for ``tab`` into ``tmp_path`` and return its path."""
    args = [str(page), '--format', 'script', '--timing', str(tmp_path / 't.tsv')]
    if tab is not None:
        args += ['--tab', tab]
    if subs is not None:
        args += ['--substitutions', str(subs)]
    out = _run(*args)
    assert out.returncode == 0, out.stderr
    script = tmp_path / 'run.sh'
    script.write_text(out.stdout, encoding='utf-8')
    return script


def test_generated_script_passes_bash_syntax_check(page, tmp_path):
    """The generated script parses with ``bash -n``."""
    script = _bash_script(page, tmp_path, 'Alpha')
    done = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_generated_script_runs_in_one_session_and_writes_timing(tmp_path):
    """State carries between commands, and the TSV has one row per command."""
    page = tmp_path / 'p.md'
    page.write_text(
        '```console\nexport STEP_VAR=carried\nmkdir sub\ncd sub\n'
        'echo "$STEP_VAR" > out.txt\ntest "$(basename "$PWD")" = sub\n```\n',
        encoding='utf-8',
    )
    script = _bash_script(page, tmp_path, None)
    done = subprocess.run(
        ['bash', '-e', str(script)], capture_output=True, text=True, cwd=tmp_path
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / 'sub' / 'out.txt').read_text().strip() == 'carried'
    rows = (tmp_path / 't.tsv').read_text().splitlines()
    assert rows[0] == 'page_line\twall_seconds\texit_code\tcommand'
    assert [r.split('\t')[0] for r in rows[1:]] == ['2', '3', '4', '5', '6']
    assert {r.split('\t')[2] for r in rows[1:]} == {'0'}
    assert done.stdout.count('::group::') == done.stdout.count('::endgroup::') == 5
    assert '::group::[page line 4] cd sub' in done.stdout


def test_generated_script_stops_at_first_failure(tmp_path):
    """The step after a non-zero exit code does not run; the exit code is passed on."""
    page = tmp_path / 'p.md'
    page.write_text(
        '```console\ntouch first.flag\nexit_code_seven() { return 7; }; exit_code_seven\n'
        'touch after.flag\n```\n',
        encoding='utf-8',
    )
    script = _bash_script(page, tmp_path, None)
    done = subprocess.run(
        ['bash', '-e', str(script)], capture_output=True, text=True, cwd=tmp_path
    )
    assert done.returncode == 7
    assert (tmp_path / 'first.flag').exists()
    assert not (tmp_path / 'after.flag').exists()
    rows = [r.split('\t') for r in (tmp_path / 't.tsv').read_text().splitlines()[1:]]
    assert [(r[0], r[2]) for r in rows] == [('2', '0'), ('3', '7')]
    assert '::error::[page line 3] exit code 7' in done.stdout


def test_generated_script_prints_deviation_line_for_substituted_command(tmp_path):
    """A substituted command prints the page text, the job text and the reason."""
    page = tmp_path / 'p.md'
    page.write_text('```console\necho on-the-page\n```\n', encoding='utf-8')
    subs = _write_subs(tmp_path, 'echo on-the-page', 'echo on-the-job')
    script = _bash_script(page, tmp_path, None, subs)
    done = subprocess.run(['bash', str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0
    assert (
        'DEVIATION: page says echo on-the-page; job runs echo on-the-job; reason: test'
        in done.stdout
    )
    assert 'on-the-job' in done.stdout.splitlines()
    assert '::group::[page line 2] echo on-the-page' in done.stdout


def test_committed_substitutions_match_the_install_page():
    """Every committed substitution matches a command on the Ubuntu tab of the page."""
    out = _run(
        str(_REPO_ROOT / 'docs' / 'How-to' / 'installation.md'),
        '--tab',
        'Debian / Ubuntu',
        '--substitutions',
        str(_REPO_ROOT / '.github' / 'docs-install-linux-substitutions.toml'),
        '--format',
        'table',
    )
    assert out.returncode == 0, out.stderr
    assert '`git clone "$GITHUB_WORKSPACE" PROTEUS`' in out.stdout
    assert '`bash install.sh` | as written' in out.stdout


@pytest.mark.parametrize(
    ('tabs', 'expected'),
    [
        (
            ['Alpha', 'Beta'],
            [
                'echo shared-one',
                'echo alpha-one',
                'echo alpha-two --long',
                'echo beta-one   # a trailing comment',
            ],
        ),
        (['Beta'], ['echo shared-one', 'echo beta-one   # a trailing comment']),
    ],
)
def test_several_tabs_are_selected_together_in_page_order(tabs, expected):
    """A page with two tab sets needs two names; each name adds its blocks, in page order."""
    required, _ = steps.select_steps(steps.parse_page(_PAGE), tabs)
    commands = [s.command for s in required]
    assert commands[: len(expected)] == expected
    assert ('echo alpha-one' in commands) == ('Alpha' in tabs)


def test_an_unknown_name_among_several_tabs_is_refused():
    """One wrong name stops the selection, so a renamed tab cannot drop its commands silently."""
    with pytest.raises(steps.InstallStepsError, match="unknown tab 'Gamma'"):
        steps.select_steps(steps.parse_page(_PAGE), ['Alpha', 'Gamma'])
