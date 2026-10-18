"""Unit tests for ``proteus.grid.summarise``.

Covers the ``summarise`` entrypoint: invalid-path error, status
aggregation, general-category dispatch (Running / Completed / Error /
All / named statuses), code-status dispatch, and unmatched input.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
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


@pytest.mark.parametrize('code', [30, 31, 32, 33, 34])
def test_summarise_groups_invariant_codes_30_to_34_under_error(tmp_path, caplog, code):
    """Invariant violation status codes 30 through 34 belong to the Error
    category and must not appear under Completed or Running.

    Discrimination: an upper bound of 30 on the Error range drops codes
    30..34 from Error summaries and prints '(None)' instead of Case 0.
    """
    grid = _make_grid(tmp_path, statuses=[code])

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Error') is True
    error_section = caplog.text.split('Error cases:')[1]
    assert f'Case 0     : Code {code}' in error_section
    assert '(None)' not in error_section

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Completed') is True
    completed_section = caplog.text.split('Completed cases:')[1]
    assert f'Code {code}' not in completed_section
    assert '(None)' in completed_section

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Running') is True
    running_section = caplog.text.split('Running cases:')[1]
    assert f'Code {code}' not in running_section
    assert '(None)' in running_section


def test_summarise_groups_evaporated_status_29_under_completed_not_error(tmp_path, caplog):
    """Status 29 ('Completed (planet evaporated)') is grouped under
    Completed (10-19 and 29) and excluded from Error (20-28 and 30-34)
    and Running (0-9).

    Discrimination: leaving 29 inside range(20, 30) misclassifies an
    evaporated planet as an Error case and omits it from Completed.
    """
    grid = _make_grid(tmp_path, statuses=[29])

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Completed') is True
    completed_section = caplog.text.split('Completed cases:')[1]
    assert 'Case 0     : Code 29 - Completed (planet evaporated)' in completed_section
    assert '(None)' not in completed_section

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Error') is True
    error_section = caplog.text.split('Error cases:')[1]
    assert 'Code 29' not in error_section
    assert '(None)' in error_section

    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        assert summarise_mod.summarise(str(grid), tgt_status='Running') is True
    running_section = caplog.text.split('Running cases:')[1]
    assert 'Code 29' not in running_section
    assert '(None)' in running_section


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


# ---------------------------------------------------------------------------
# Robustness: non-contiguous case folders and malformed status files
# ---------------------------------------------------------------------------


def test_summarise_handles_noncontiguous_case_folders(tmp_path, caplog):
    """Case folders need not be numbered 0..N-1: after a failed case is
    deleted the remaining folders have gaps (here 0, 2, 5). summarise must
    read the folders that exist and report cases by their real index.

    Discrimination: a regression that reconstructed paths as case_%06d for
    i in range(N) would raise FileNotFoundError on the absent case_000001,
    and would mislabel the error case (real index 5) as index 2.
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    for idx, code in [(0, 10), (2, 11), (5, 20)]:
        case = grid / f'case_{idx:06d}'
        case.mkdir()
        (case / 'status').write_text(f'{code}\nsome-comment\n', encoding='utf-8')

    with caplog.at_level(logging.INFO, logger='fwl'):
        result = summarise_mod.summarise(str(grid), tgt_status='code=20')

    assert result is True
    assert 'Found 3 cases' in caplog.text
    code20_section = caplog.text.split('Code 20 cases:')[1]
    # The error case is at real index 5, never at a contiguous index 2.
    assert 'Case 5    ' in code20_section
    assert 'Case 2    ' not in code20_section


def test_summarise_raises_on_empty_status_file(tmp_path):
    """A present-but-empty status file (a truncated or partial write) raises
    ValueError, distinct from the FileNotFoundError raised when the file is
    absent entirely. Discrimination: reading lines[0] of an empty file would
    otherwise raise an opaque IndexError rather than a named condition.
    """
    grid = tmp_path / 'grid'
    grid.mkdir()
    (grid / 'case_000000').mkdir()
    (grid / 'case_000000' / 'status').write_text('', encoding='utf-8')

    with pytest.raises(ValueError, match='Status file is empty') as exc:
        summarise_mod.summarise(str(grid))
    assert 'case_000000' in str(exc.value)
