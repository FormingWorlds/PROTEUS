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

import logging

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


def test_summarise_raises_when_grid_directory_missing(tmp_path, caplog):
    """A non-existent grid path raises FileNotFoundError. A regression
    that returned silently would mask user typos and waste time.
    Discrimination: nothing must be logged before the raise (no
    "Statistics" header from a partially-executed path).
    """
    with caplog.at_level(logging.INFO, logger='fwl'):
        with pytest.raises(FileNotFoundError, match='Invalid path'):
            summarise_mod.summarise(str(tmp_path / 'missing'))
    assert 'Statistics' not in caplog.text


def test_summarise_raises_when_case_status_file_missing(tmp_path):
    """If a case_NNNNNN folder exists but has no ``status`` file, the
    function raises FileNotFoundError. Discrimination against silent
    skip: every case must have a status file to be analysable; the
    error message must name the specific case directory so an operator
    can locate the broken case.
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    (grid / 'case_000000').mkdir()  # no status file
    with pytest.raises(FileNotFoundError, match='Cannot find status file') as exc:
        summarise_mod.summarise(str(grid))
    assert 'case_000000' in str(exc.value)


# ---------------------------------------------------------------------------
# Status aggregation: tgt_status omitted
# ---------------------------------------------------------------------------


def test_summarise_returns_true_when_no_tgt_status(tmp_path, caplog):
    """With tgt_status=None, summarise logs statistics for each present
    status code and returns True (no filter applied).

    Discrimination: the log must include the count line (count + pct
    + comment); a regression that skipped the statistics loop would
    leave the log empty.
    """
    grid = _make_grid(tmp_path, statuses=[0, 10, 10, 20])
    with caplog.at_level(logging.INFO, logger='fwl'):
        result = summarise_mod.summarise(str(grid))

    assert result is True
    assert 'Statistics:' in caplog.text
    assert 'Found 4 cases' in caplog.text


# ---------------------------------------------------------------------------
# tgt_status: general categories
# ---------------------------------------------------------------------------


def test_summarise_lists_running_cases_for_running_category(tmp_path, caplog):
    """tgt_status='Running' (range 0-9) lists every case in that range
    and returns True. Discrimination: only the matching cases must be
    logged; completed (10) and error (20) cases must not appear.
    """
    grid = _make_grid(tmp_path, statuses=[0, 5, 10, 20])
    with caplog.at_level(logging.INFO, logger='fwl'):
        result = summarise_mod.summarise(str(grid), tgt_status='Running')

    assert result is True
    assert 'Running cases:' in caplog.text
    assert 'Case 0    ' in caplog.text
    assert 'Case 1    ' in caplog.text
    running_section = caplog.text.split('Running cases:')[1]
    assert 'Code 10 -' not in running_section


def test_summarise_lists_completed_cases_for_complete_alias(tmp_path, caplog):
    """tgt_status='complete' (an alias for 'completed') matches the
    Completed general category. Discrimination: the alias rewrite at
    line 75-76 must convert it.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11, 20])
    with caplog.at_level(logging.INFO, logger='fwl'):
        result = summarise_mod.summarise(str(grid), tgt_status='complete')

    assert result is True
    assert 'Completed cases:' in caplog.text


def test_summarise_prints_none_for_empty_general_category(tmp_path, caplog):
    """When no cases match the requested general category, the function
    logs '(None)'. Discrimination: a regression that omitted the
    sentinel would leave the section empty.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11])  # all Completed
    with caplog.at_level(logging.INFO, logger='fwl'):
        summarise_mod.summarise(str(grid), tgt_status='Error')

    assert 'Error cases:' in caplog.text
    assert '(None)' in caplog.text


# ---------------------------------------------------------------------------
# tgt_status: code= dispatch
# ---------------------------------------------------------------------------


def test_summarise_lists_cases_by_explicit_code(tmp_path, caplog):
    """tgt_status='code=10' lists only cases whose status is exactly 10.
    Discrimination: a regression that interpreted code=10 as a range
    would also match codes 11..19.
    """
    grid = _make_grid(tmp_path, statuses=[10, 11, 10, 12])
    with caplog.at_level(logging.INFO, logger='fwl'):
        result = summarise_mod.summarise(str(grid), tgt_status='code=10')

    assert result is True
    assert 'Code 10 cases:' in caplog.text
    code10_section = caplog.text.split('Code 10 cases:')[1]
    assert 'Case 0    ' in code10_section
    assert 'Case 2    ' in code10_section
    assert 'Case 1    ' not in code10_section


def test_summarise_treats_status_equals_as_code_equals(tmp_path, caplog):
    """tgt_status='status=10' is converted to 'code=10' before dispatch.
    Backwards-compatibility alias; a regression that dropped the
    replacement would treat status=10 as unmatched. Discrimination:
    the unmatched-status help text must NOT appear (which would be
    emitted by the fall-through branch if the alias rewrite failed).
    """
    grid = _make_grid(tmp_path, statuses=[10])
    with caplog.at_level(logging.INFO, logger='fwl'):
        summarise_mod.summarise(str(grid), tgt_status='status=10')

    assert 'Code 10 cases:' in caplog.text
    assert 'Invalid status category' not in caplog.text


# ---------------------------------------------------------------------------
# tgt_status: invalid
# ---------------------------------------------------------------------------


def test_summarise_prints_help_message_for_unmatched_status(tmp_path, caplog):
    """An unrecognised tgt_status falls through both the general-category
    and code= branches and logs a warning. Discrimination: a regression
    that crashed on unmatched input would not produce the 'Invalid
    status category' message.
    """
    grid = _make_grid(tmp_path, statuses=[0])
    with caplog.at_level(logging.WARNING, logger='fwl'):
        result = summarise_mod.summarise(str(grid), tgt_status='nonsense-category')

    assert result is False
    assert 'Invalid status category' in caplog.text
