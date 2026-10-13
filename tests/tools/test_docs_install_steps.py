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
import re
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
    # a whole-line comment to skip
    echo alpha-two \\
      --long
    ```

=== "Beta"

    ```console
    echo beta-one   # a trailing comment
    echo beta-two \\
    ```

!!! note "Optional"
    ```console
    echo optional-shared
    ```

??? note "Collapsible"
    ```console
    echo optional-collapsible
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


def _run(capsys, *args: str) -> subprocess.CompletedProcess[str]:
    """Run steps.main in-process, capturing stdout/stderr and exit code."""
    capsys.readouterr()
    rc = steps.main(list(args))
    captured = capsys.readouterr()
    return subprocess.CompletedProcess(
        args=list(args), returncode=rc, stdout=captured.out, stderr=captured.err
    )


def _write_subs(tmp_path: Path, match: str, replace: str = 'true') -> Path:
    """Write a one-entry substitution table and return its path."""
    path = tmp_path / 'subs.toml'
    path.write_text(
        f'[[substitution]]\nmatch = "{match}"\nreplace = "{replace}"\nreason = "test"\n',
        encoding='utf-8',
    )
    return path


def test_cli_subprocess_invocation(page):
    """Running the tool as a subprocess executes __main__ and exits 0."""
    out = subprocess.run(
        [sys.executable, str(_TOOL), str(page), '--tab', 'Alpha'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0
    assert 'echo alpha-one' in out.stdout


def test_cli_subprocess_failure_exits_2(page):
    """Running the tool as a subprocess exits with code 2 on error."""
    out = subprocess.run(
        [sys.executable, str(_TOOL), str(page), '--tab', 'Gamma'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert 'error:' in out.stderr
    assert 'unknown tab' in out.stderr


def test_tab_selection_keeps_shared_blocks_and_selected_tab_only(page, capsys):
    """Alpha gets the shared blocks and its own, in page order, and no Beta command."""
    out = _run(capsys, str(page), '--tab', 'Alpha')
    assert out.returncode == 0
    assert out.stdout.splitlines() == [
        'echo shared-one',
        'echo alpha-one',
        'echo alpha-two --long',
        'echo shared-two',
    ]
    assert 'beta-one' not in out.stdout and 'beta-two' not in out.stdout
    assert '# a whole-line comment to skip' not in out.stdout


def test_other_tab_does_not_see_first_tab_commands(page, capsys):
    """Beta gets its own commands and the shared ones, none of Alpha's."""
    out = _run(capsys, str(page), '--tab', 'Beta')
    assert out.returncode == 0
    assert out.stdout.splitlines() == [
        'echo shared-one',
        'echo beta-one   # a trailing comment',
        'echo beta-two',
        'echo shared-two',
    ]
    assert 'alpha' not in out.stdout


def test_repeated_tab_cli_options_select_all_specified_tabs(page, capsys):
    """Passing multiple --tab options selects commands for all specified tabs in page order."""
    out = _run(capsys, str(page), '--tab', 'Alpha', '--tab', 'Beta')
    assert out.returncode == 0
    assert out.stdout.splitlines() == [
        'echo shared-one',
        'echo alpha-one',
        'echo alpha-two --long',
        'echo beta-one   # a trailing comment',
        'echo beta-two',
        'echo shared-two',
    ]
    assert 'echo alpha-one' in out.stdout and 'echo beta-one' in out.stdout


def test_unknown_tab_exits_2_and_names_known_tabs(page, capsys):
    """A tab name that is not on the page is a structural error."""
    out = _run(capsys, str(page), '--tab', 'Gamma')
    assert out.returncode == 2
    assert 'unknown tab' in out.stderr
    assert 'Alpha' in out.stderr and 'Beta' in out.stderr
    assert out.stdout == ''


def test_page_without_console_block_exits_2(tmp_path, capsys):
    """A page with no console block cannot yield a plan."""
    path = tmp_path / 'empty.md'
    path.write_text('# Title\n\nNo commands here.\n\n```python\nx = 1\n```\n', encoding='utf-8')
    out = _run(capsys, str(path))
    assert out.returncode == 2
    assert 'no console block' in out.stderr
    assert out.stdout == ''


def test_unreadable_page_file_exits_2_and_names_file(tmp_path, capsys):
    """An unreadable or non-existent page file causes an exit code of 2 naming the file."""
    missing = tmp_path / 'nonexistent_page.md'
    out = _run(capsys, str(missing))
    assert out.returncode == 2
    assert str(missing) in out.stderr
    assert 'cannot read' in out.stderr


def test_admonition_blocks_are_skipped_and_listed(page, capsys):
    """Admonition blocks are optional: not in the plan, listed with --list-skipped."""
    plan = _run(capsys, str(page), '--tab', 'Alpha').stdout
    assert 'optional' not in plan
    listing = _run(capsys, str(page), '--tab', 'Alpha', '--list-skipped').stdout.splitlines()
    assert len(listing) == 3
    assert listing[0].endswith('(note) echo optional-shared')
    assert listing[1].endswith('(note) echo optional-collapsible')
    assert listing[2].endswith('(tip) echo optional-alpha')
    beta = _run(capsys, str(page), '--tab', 'Beta', '--list-skipped').stdout
    assert 'optional-alpha' not in beta
    assert 'optional-shared' in beta and 'optional-collapsible' in beta


def test_list_skipped_with_no_optional_steps(tmp_path, capsys):
    """Listing skipped steps on a page without admonitions reports no optional steps."""
    p = tmp_path / 'no_opt.md'
    p.write_text('```console\necho standalone\n```\n', encoding='utf-8')
    out = _run(capsys, str(p), '--list-skipped')
    assert out.returncode == 0
    assert out.stdout == 'No optional steps.\n'


def test_line_numbers_are_page_lines_of_each_command(page):
    """Each command keeps the page line it starts on, also after a continuation."""
    required, _ = steps.select_steps(steps.parse_page(_PAGE), 'Alpha')
    lines = _PAGE.splitlines()
    for step in required:
        assert lines[step.line - 1].strip().lstrip('$ ').startswith(step.command.split(' ')[0])
    assert [s.line for s in required] == [4, 10, 12, 45]
    assert lines[9].strip() == 'echo alpha-one'
    assert lines[11].strip() == 'echo alpha-two \\'
    assert lines[44].strip() == '$ echo shared-two'


def test_substitution_replaces_command_and_table_shows_deviation(page, tmp_path, capsys):
    """A matching substitution changes the job command and shows its reason."""
    subs = _write_subs(tmp_path, 'echo alpha-one', 'echo replaced')
    out = _run(
        capsys, str(page), '--tab', 'Alpha', '--substitutions', str(subs), '--format', 'table'
    )
    assert out.returncode == 0
    assert '| 10 | `echo alpha-one` | `echo replaced` | test |' in out.stdout
    assert '| 4 | `echo shared-one` | as written |  |' in out.stdout


def test_substitution_matches_command_with_its_page_comment(page, tmp_path, capsys):
    """The match may be the command text as on the page, trailing comment included."""
    subs = _write_subs(tmp_path, 'echo beta-one   # a trailing comment', 'echo replaced')
    out = _run(
        capsys, str(page), '--tab', 'Beta', '--substitutions', str(subs), '--format', 'table'
    )
    assert out.returncode == 0
    assert '`echo replaced`' in out.stdout
    assert '(page comment: `# a trailing comment`)' in out.stdout


def test_substitution_matches_command_without_page_comment(page, tmp_path, capsys):
    """A substitution match without the trailing comment applies to the page command."""
    subs = _write_subs(tmp_path, 'echo beta-one', 'echo replaced-beta')
    out = _run(
        capsys,
        str(page),
        '--tab',
        'Beta',
        '--substitutions',
        str(subs),
        '--format',
        'commands',
    )
    assert out.returncode == 0
    assert 'echo replaced-beta' in out.stdout.splitlines()
    assert 'echo beta-one' not in out.stdout


def test_commands_format_prints_job_command_when_substituted(page, tmp_path, capsys):
    """The commands format prints the substituted job command rather than the page command."""
    subs = _write_subs(tmp_path, 'echo alpha-one', 'echo replaced-alpha')
    out = _run(
        capsys,
        str(page),
        '--tab',
        'Alpha',
        '--substitutions',
        str(subs),
        '--format',
        'commands',
    )
    assert out.returncode == 0
    assert 'echo replaced-alpha' in out.stdout.splitlines()
    assert 'echo alpha-one' not in out.stdout


def test_stale_substitution_exits_2(page, tmp_path, capsys):
    """A match that is not on the page fails the run and names all stale matches."""
    path = tmp_path / 'subs.toml'
    path.write_text(
        '[[substitution]]\nmatch = "echo alpha-one-renamed"\nreplace = "true"\nreason = "r1"\n\n'
        '[[substitution]]\nmatch = "echo alpha-two-renamed"\nreplace = "true"\nreason = "r2"\n',
        encoding='utf-8',
    )
    out = _run(
        capsys, str(page), '--tab', 'Alpha', '--substitutions', str(path), '--format', 'script'
    )
    assert out.returncode == 2
    assert 'matches no command on the page' in out.stderr
    assert "'echo alpha-one-renamed'" in out.stderr
    assert "'echo alpha-two-renamed'" in out.stderr
    assert out.stdout == ''


def test_substitution_for_other_tab_command_is_stale(page, tmp_path, capsys):
    """A match that exists only on another tab is stale for the selected tab."""
    subs = _write_subs(tmp_path, 'echo beta-one   # a trailing comment')
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(subs))
    assert out.returncode == 2
    assert 'matches no command on the page' in out.stderr
    assert 'echo beta-one' in out.stderr


def test_substitution_matching_prefix_only_is_stale(page, tmp_path, capsys):
    """A substitution whose match is only a prefix of a command is rejected as stale."""
    subs = _write_subs(tmp_path, 'echo alpha-two', 'echo replaced')
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(subs))
    assert out.returncode == 2
    assert 'matches no command on the page' in out.stderr
    assert 'echo alpha-two' in out.stderr


def test_substitution_missing_reason_exits_2(page, tmp_path, capsys):
    """An entry without a reason is rejected, so every deviation is explained."""
    path = tmp_path / 'subs.toml'
    path.write_text('[[substitution]]\nmatch = "echo alpha-one"\nreplace = "true"\n')
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(path))
    assert out.returncode == 2
    assert "'reason'" in out.stderr


def test_substitution_whitespace_only_reason_exits_2(page, tmp_path, capsys):
    """A substitution whose reason contains only whitespace is rejected with exit code 2."""
    path = tmp_path / 'subs.toml'
    path.write_text(
        '[[substitution]]\nmatch = "echo alpha-one"\nreplace = "true"\nreason = "   "\n',
        encoding='utf-8',
    )
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(path))
    assert out.returncode == 2
    assert "'reason'" in out.stderr


def test_duplicate_substitution_match_exits_2(page, tmp_path, capsys):
    """Duplicate substitution match entries in one table are rejected with exit code 2."""
    path = tmp_path / 'subs.toml'
    path.write_text(
        '[[substitution]]\nmatch = "echo alpha-one"\nreplace = "true"\nreason = "first"\n\n'
        '[[substitution]]\nmatch = "echo alpha-one"\nreplace = "false"\nreason = "second"\n',
        encoding='utf-8',
    )
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(path))
    assert out.returncode == 2
    assert 'repeats a match' in out.stderr


def test_missing_substitutions_file_exits_2_and_names_file(page, tmp_path, capsys):
    """A missing substitutions file causes an exit code of 2 naming the missing file."""
    missing = tmp_path / 'nonexistent_subs.toml'
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(missing))
    assert out.returncode == 2
    assert str(missing) in out.stderr
    assert 'cannot read substitutions' in out.stderr


def test_invalid_toml_substitutions_file_exits_2(page, tmp_path, capsys):
    """A malformed TOML substitutions file causes an exit code of 2 with an error message."""
    bad_toml = tmp_path / 'bad.toml'
    bad_toml.write_text('[[substitution\nmatch = broken', encoding='utf-8')
    out = _run(capsys, str(page), '--tab', 'Alpha', '--substitutions', str(bad_toml))
    assert out.returncode == 2
    assert 'cannot read substitutions' in out.stderr


def test_table_format_escapes_pipe_characters_in_commands(tmp_path, capsys):
    """Commands and reasons with pipe characters are escaped with a backslash in tables."""
    page_file = tmp_path / 'pipe.md'
    page_file.write_text('```console\necho hello | tr a-z A-Z\n```\n', encoding='utf-8')
    subs = tmp_path / 'subs.toml'
    subs.write_text(
        '[[substitution]]\nmatch = "echo hello | tr a-z A-Z"\n'
        'replace = "cat a | cat b"\nreason = "pipe | test"\n',
        encoding='utf-8',
    )
    out = _run(capsys, str(page_file), '--substitutions', str(subs), '--format', 'table')
    assert out.returncode == 0
    assert r'`echo hello \| tr a-z A-Z`' in out.stdout
    assert r'`cat a \| cat b`' in out.stdout
    assert r'pipe \| test' in out.stdout


def test_table_format_escapes_backticks_in_commands(tmp_path, capsys):
    """Commands with backticks are displayed with single quotes in table cells."""
    page_file = tmp_path / 'backtick.md'
    page_file.write_text('```console\necho `date`\n```\n', encoding='utf-8')
    out = _run(capsys, str(page_file), '--format', 'table')
    assert out.returncode == 0
    assert "`echo 'date'`" in out.stdout
    assert '`echo `date``' not in out.stdout


def _bash_script(
    page: Path,
    tmp_path: Path,
    tab: str | None,
    subs: Path | None = None,
    *,
    capsys,
    timing: Path | None = None,
) -> Path:
    """Generate the script for ``tab`` into ``tmp_path`` and return its path."""
    timing_path = str(timing or (tmp_path / 't.tsv'))
    args = [str(page), '--format', 'script', '--timing', timing_path]
    if tab is not None:
        args += ['--tab', tab]
    if subs is not None:
        args += ['--substitutions', str(subs)]
    out = _run(capsys, *args)
    assert out.returncode == 0, out.stderr
    script = tmp_path / 'run.sh'
    script.write_text(out.stdout, encoding='utf-8')
    return script


def test_generated_script_passes_bash_syntax_check(page, tmp_path, capsys):
    """The generated script parses with ``bash -n``."""
    script = _bash_script(page, tmp_path, 'Alpha', capsys=capsys)
    done = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert script.is_file()
    assert '#!/usr/bin/env bash' in script.read_text(encoding='utf-8')


def test_generated_script_runs_in_one_session_and_writes_timing(tmp_path, capsys):
    """State carries between commands, and the TSV has one row per command."""
    p = tmp_path / 'p.md'
    p.write_text(
        '```console\nexport STEP_VAR=carried\nmkdir sub\ncd sub\n'
        'echo "$STEP_VAR" > out.txt\ntest "$(basename "$PWD")" = sub\n'
        'sleep 1\n```\n',
        encoding='utf-8',
    )
    script = _bash_script(p, tmp_path, None, capsys=capsys)
    done = subprocess.run(
        ['bash', '-e', str(script)], capture_output=True, text=True, cwd=tmp_path
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / 'sub' / 'out.txt').read_text().strip() == 'carried'
    rows = (tmp_path / 't.tsv').read_text().splitlines()
    assert rows[0] == 'page_line\twall_seconds\texit_code\tcommand'
    assert [r.split('\t')[0] for r in rows[1:]] == ['2', '3', '4', '5', '6', '7']
    assert {r.split('\t')[2] for r in rows[1:]} == {'0'}
    for r in rows[1:]:
        assert re.match(r'^\d+\.\d$', r.split('\t')[1])
    sleep_row = rows[-1].split('\t')
    wall_sec = float(sleep_row[1])
    assert 0.9 <= wall_sec <= 30.0
    assert done.stdout.count('::group::') == done.stdout.count('::endgroup::') == 6
    assert '::group::[page line 4] cd sub' in done.stdout


def test_generated_script_handles_timing_path_with_spaces(tmp_path, capsys):
    """A timing file path containing spaces is quoted properly in the bash script."""
    timing_dir = tmp_path / 'my dir'
    timing_dir.mkdir()
    timing_file = timing_dir / 't.tsv'
    p = tmp_path / 'p.md'
    p.write_text('```console\necho hello\n```\n', encoding='utf-8')
    script = _bash_script(p, tmp_path, None, capsys=capsys, timing=timing_file)
    done = subprocess.run(
        ['bash', '-e', str(script)], capture_output=True, text=True, cwd=tmp_path
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert timing_file.is_file()
    rows = timing_file.read_text(encoding='utf-8').splitlines()
    assert len(rows) == 2
    assert rows[0] == 'page_line\twall_seconds\texit_code\tcommand'
    assert rows[1].startswith('2\t')


def test_timing_row_replaces_tab_characters_in_command(tmp_path, capsys):
    """A command containing tab characters produces a TSV row with exactly 4 columns."""
    p = tmp_path / 'tab_cmd.md'
    p.write_text('```console\necho\twith\ttabs\n```\n', encoding='utf-8')
    script = _bash_script(p, tmp_path, None, capsys=capsys)
    done = subprocess.run(['bash', str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    rows = (tmp_path / 't.tsv').read_text(encoding='utf-8').splitlines()
    assert len(rows) == 2
    cols = rows[1].split('\t')
    assert len(cols) == 4
    assert cols[3] == 'echo with tabs'


def test_generated_script_stops_at_first_failure(tmp_path, capsys):
    """The step after a non-zero exit code does not run; the exit code is passed on."""
    p = tmp_path / 'p.md'
    p.write_text(
        '```console\ntouch first.flag\nexit_code_seven() { return 7; }; exit_code_seven\n'
        'touch after.flag\n```\n',
        encoding='utf-8',
    )
    script = _bash_script(p, tmp_path, None, capsys=capsys)
    done = subprocess.run(
        ['bash', '-e', str(script)], capture_output=True, text=True, cwd=tmp_path
    )
    assert done.returncode == 7
    assert (tmp_path / 'first.flag').exists()
    assert not (tmp_path / 'after.flag').exists()
    rows = [r.split('\t') for r in (tmp_path / 't.tsv').read_text().splitlines()[1:]]
    assert [(r[0], r[2]) for r in rows] == [('2', '0'), ('3', '7')]
    assert '::error::[page line 3] exit code 7' in done.stdout


def test_generated_script_prints_deviation_line_for_substituted_command(tmp_path, capsys):
    """A substituted command prints the page text, the job text and the reason."""
    p = tmp_path / 'p.md'
    p.write_text('```console\necho on-the-page\n```\n', encoding='utf-8')
    subs = _write_subs(tmp_path, 'echo on-the-page', 'echo on-the-job')
    script = _bash_script(p, tmp_path, None, subs, capsys=capsys)
    done = subprocess.run(['bash', str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0
    assert (
        'DEVIATION: page says echo on-the-page; job runs echo on-the-job; reason: test'
        in done.stdout
    )
    assert 'on-the-job' in done.stdout.splitlines()
    assert '::group::[page line 2] echo on-the-page' in done.stdout


def test_committed_substitutions_match_the_install_page(capsys):
    """Every committed substitution matches a command on the Ubuntu tab of the page."""
    out = _run(
        capsys,
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
                'echo beta-two',
                'echo shared-two',
            ],
        ),
        (
            ['Beta'],
            [
                'echo shared-one',
                'echo beta-one   # a trailing comment',
                'echo beta-two',
                'echo shared-two',
            ],
        ),
    ],
)
def test_several_tabs_are_selected_together_in_page_order(tabs, expected):
    """Multiple tab names select the blocks of each named tab in page order."""
    required, _ = steps.select_steps(steps.parse_page(_PAGE), tabs)
    commands = [s.command for s in required]
    assert commands == expected
    assert ('echo alpha-one' in commands) == ('Alpha' in tabs)
    assert 'echo shared-one' in commands and 'echo shared-two' in commands


def test_an_unknown_name_among_several_tabs_is_refused():
    """One wrong name stops the selection, so a renamed tab cannot drop its commands silently."""
    with pytest.raises(steps.InstallStepsError, match="unknown tab 'Gamma'") as exc_info:
        steps.select_steps(steps.parse_page(_PAGE), ['Alpha', 'Gamma'])
    assert "unknown tab 'Gamma'" in str(exc_info.value)
    assert 'Alpha' in str(exc_info.value) and 'Beta' in str(exc_info.value)


def test_plan_entry_substituted_property():
    """PlanEntry.substituted is true only when the job command differs from the step command."""
    step = steps.Step(1, 'echo original', None, None)
    entry_same = steps.PlanEntry(step, 'echo original', '')
    entry_diff = steps.PlanEntry(step, 'echo replacement', 'reason')
    assert not entry_same.substituted
    assert entry_diff.substituted


def test_tab_with_no_required_commands_exits_2(tmp_path, capsys):
    """A tab containing only optional steps and no required command raises an error."""
    p = tmp_path / 'only_optional.md'
    p.write_text(
        '=== "Gamma"\n\n    !!! note "Only optional"\n        ```console\n        echo opt\n        ```\n',
        encoding='utf-8',
    )
    out = _run(capsys, str(p), '--tab', 'Gamma')
    assert out.returncode == 2
    assert 'no console block found for the selection' in out.stderr


def test_fence_like_line_does_not_close_console_block():
    """Three backticks followed by text do not close the console block; the line stays a command."""
    page_text = '```console\necho before\n```not-a-fence\necho after\n```\n'
    steps_list = steps.parse_page(page_text)
    assert [s.command for s in steps_list] == ['echo before', '```not-a-fence', 'echo after']
    assert [s.line for s in steps_list] == [2, 3, 4]
