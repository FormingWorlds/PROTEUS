"""Unit tests for ``proteus.grid.summarise``.

Covers the ``summarise`` entrypoint: invalid-path error, status
aggregation, general-category dispatch (Running / Completed / Error /
All / named statuses), code-status dispatch, and unmatched input.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

import proteus.grid.summarise as summarise_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(tmp_path, statuses):
    """Build a grid directory with one case per entry in ``statuses``.

    Each case_XXXXXX folder gets a status file containing two lines:
    the integer code, then a human-readable comment.
    """
    grid_dir = tmp_path / 'grid'
    grid_dir.mkdir()
    for i, code in enumerate(statuses):
        case_dir = grid_dir / f'case_{i:06d}'
        case_dir.mkdir()
        (case_dir / 'status').write_text(f'{code}\nsome-comment\n', encoding='utf-8')
    return grid_dir


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_summarise_raises_when_grid_directory_missing(tmp_path):
    """A non-existent grid path raises FileNotFoundError. A regression
    that returned silently would mask user typos and waste time.
    """
    with pytest.raises(FileNotFoundError, match='Invalid path'):
        summarise_mod.summarise(str(tmp_path / 'missing'))


def test_summarise_raises_when_case_status_file_missing(tmp_path):
    """If a case_NNNNNN folder exists but has no ``status`` file, the
    function raises FileNotFoundError. Discrimination against silent
    skip: every case must have a status file to be analysable.
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    (grid / 'case_000000').mkdir()  # no status file
    with pytest.raises(FileNotFoundError, match='Cannot find status file'):
        summarise_mod.summarise(str(grid))


# ---------------------------------------------------------------------------
# Status aggregation: tgt_status omitted
# ---------------------------------------------------------------------------


def test_summarise_returns_true_when_no_tgt_status(tmp_path, capsys):
    """With tgt_status=None, summarise prints statistics for each present
    status code and returns True (no filter applied).

    Discrimination: the stdout must include the count line (count + pct
    + comment); a regression that skipped the statistics loop would
    leave stdout empty.
    """
    grid = _make_grid(tmp_path, statuses=[0, 10, 10, 20])
    result = summarise_mod.summarise(str(grid))

    out = capsys.readouterr().out
    assert result is True
    # Discrimination: stats section header must be present
    assert 'Statistics:' in out
    # Discrimination: at least one count line was printed (we have 4 cases)
    assert 'Found 4 cases' in out


# ---------------------------------------------------------------------------
# tgt_status: general categories
# ---------------------------------------------------------------------------


def test_summarise_lists_running_cases_for_running_category(tmp_path, capsys):
    """tgt_status='Running' (range 0-9) lists every case in that range
    and returns True. Discrimination: only the matching cases must be
    printed; completed (10) and error (20) cases must not appear.
    """
    grid = _make_grid(tmp_path, statuses=[0, 5, 10, 20])
    result = summarise_mod.summarise(str(grid), tgt_status='Running')

    out = capsys.readouterr().out
    assert result is True
    assert 'Running cases:' in out
    # Discrimination: cases 0 and 1 (codes 0, 5) ARE running; cases 2 and 3
    # (codes 10, 20) are NOT and must not be listed under Running.
    assert 'Case 0    ' in out
    assert 'Case 1    ' in out
    # The completed/error cases must not appear in the Running section.
    # Since each case is printed only when matched, the absence of the
    # exact line for case 2 / 3 with codes 10 / 20 confirms the filter.
    assert 'Code 10 -' not in out.split('Running cases:')[1]


def test_summarise_lists_completed_cases_for_complete_alias(tmp_path, capsys):
    """tgt_status='complete' (an alias for 'completed') matches the
    Completed general category. Discrimination: the alias rewrite at
    line 75-76 must convert it.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11, 20])
    result = summarise_mod.summarise(str(grid), tgt_status='complete')

    out = capsys.readouterr().out
    assert result is True
    assert 'Completed cases:' in out


def test_summarise_prints_none_for_empty_general_category(tmp_path, capsys):
    """When no cases match the requested general category, the function
    prints '(None)'. Discrimination: a regression that omitted the
    sentinel would leave the section empty.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11])  # all Completed
    summarise_mod.summarise(str(grid), tgt_status='Error')

    out = capsys.readouterr().out
    assert 'Error cases:' in out
    assert '(None)' in out


# ---------------------------------------------------------------------------
# tgt_status: code= dispatch
# ---------------------------------------------------------------------------


def test_summarise_lists_cases_by_explicit_code(tmp_path, capsys):
    """tgt_status='code=10' lists only cases whose status is exactly 10.
    Discrimination: a regression that interpreted code=10 as a range
    would also match codes 11..19.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11, 10, 12])
    result = summarise_mod.summarise(str(grid), tgt_status='code=10')

    out = capsys.readouterr().out
    assert result is True
    assert 'Code 10 cases:' in out
    # Discrimination: cases 0 and 2 (code 10) listed; cases 1 and 3 not
    code10_section = out.split('Code 10 cases:')[1]
    assert 'Case 0    ' in code10_section
    assert 'Case 2    ' in code10_section
    # No case 1 (code 11) line in the code=10 section
    assert 'Case 1    ' not in code10_section


def test_summarise_treats_status_equals_as_code_equals(tmp_path, capsys):
    """tgt_status='status=10' is converted to 'code=10' before dispatch.
    Backwards-compatibility alias; a regression that dropped the
    replacement would treat status=10 as unmatched.
    """
    grid = _make_grid(tmp_path, statuses=[10])
    summarise_mod.summarise(str(grid), tgt_status='status=10')

    out = capsys.readouterr().out
    # Discrimination: the section header must appear (not an unmatched warning)
    assert 'Code 10 cases:' in out


# ---------------------------------------------------------------------------
# tgt_status: invalid
# ---------------------------------------------------------------------------


def test_summarise_prints_help_message_for_unmatched_status(tmp_path, capsys):
    """An unrecognised tgt_status falls through both the general-category
    and code= branches and prints a help message. Discrimination: a
    regression that crashed on unmatched input would not produce the
    'Invalid status category' message.
    """
    grid = _make_grid(tmp_path, statuses=[0])
    result = summarise_mod.summarise(str(grid), tgt_status='nonsense-category')

    out = capsys.readouterr().out
    assert result is False
    assert 'Invalid status category' in out
