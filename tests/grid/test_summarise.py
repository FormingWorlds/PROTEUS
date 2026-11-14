"""Unit tests for ``proteus.grid.summarise``.

Covers the ``summarise`` entrypoint: invalid-path error, status
aggregation, general-category dispatch (Running / Completed / Error /
Unknown / All / named statuses), code-status dispatch, unmatched input,
non-contiguous case folders, and cases without a readable status file,
which count as unknown.

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


# ---------------------------------------------------------------------------
# Robustness: non-contiguous case folders and unreadable status files
# ---------------------------------------------------------------------------


def test_summarise_handles_noncontiguous_case_folders(tmp_path, caplog):
    """Case folders need not be numbered 0..N-1: after a failed case is
    deleted the remaining folders have gaps (here 0, 2, 5). summarise must
    read the folders that exist and report cases by their real index.

    Discrimination: a regression that reconstructed paths as case_%06d for
    i in range(N) would count the absent case_000001 as unknown and
    mislabel the error case (real index 5) as index 2.
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


@pytest.mark.parametrize(
    'content',
    [None, '', 'not-a-code\n'],
    ids=['no-status-file', 'empty-status-file', 'unparseable-status-file'],
)
def test_summarise_counts_a_case_without_a_readable_status_as_unknown(
    tmp_path, caplog, content
):
    """A case folder whose status file is missing, empty or unparseable (the run
    stopped before PROTEUS wrote a status) counts as unknown, and the summary still
    gives the statistics of the other cases instead of stopping at that case. The
    case is listed by `code=-1`, `unknown` and `all`, by its own index; `unknown` also
    lists an unrecognised code (35), as grid-analyse's Unknown category does.
    """
    grid = _make_grid(tmp_path, statuses=[10, 10, 20, 35])
    status = grid / 'case_000001' / 'status'
    if content is None:
        status.unlink()
    else:
        status.write_text(content, encoding='utf-8')

    listed = {}
    for tgt, header in [
        ('code=-1', 'Code -1 cases:'),
        ('unknown', 'Unknown cases:'),
        ('all', 'All cases:'),
    ]:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='fwl'):
            assert summarise_mod.summarise(str(grid), tgt_status=tgt) is True
        listed[tgt] = caplog.text.split(header)[1]

    assert 'Found 4 cases' in caplog.text
    # One case of four in each row: 25 %.
    assert '1     (25%) Unknown (no readable status file)' in caplog.text
    assert '1     (25%) Completed (solidified)' in caplog.text
    assert '1     (25%) Error (generic case, or configuration issue)' in caplog.text
    assert '1     (25%) UNHANDLED STATUS (35)' in caplog.text
    assert 'UNHANDLED STATUS (-1)' not in caplog.text
    unknown_line = 'Case 1     : Code -1 - Unknown (no readable status file)'
    unrecognised_line = 'Case 3     : Code 35 - UNHANDLED STATUS (35)'
    assert unknown_line in listed['code=-1']
    assert 'Case 3    ' not in listed['code=-1']
    assert unknown_line in listed['unknown']
    assert unrecognised_line in listed['unknown']
    for tgt in ('code=-1', 'unknown'):
        assert 'Case 0    ' not in listed[tgt], tgt
        assert 'Case 2    ' not in listed[tgt], tgt
    # `all` lists every case, the unknown ones included.
    for idx in (0, 1, 2, 3):
        assert f'Case {idx}    ' in listed['all']
    assert unknown_line in listed['all']


# ---------------------------------------------------------------------------
# status_category
# ---------------------------------------------------------------------------


def test_status_category_bins_codes_by_documented_ranges():
    """status_category maps codes to Running (0-9), Completed (10-19 and 29) and
    Error (20-28).

    Uses boundary codes so an off-by-one range bound is caught. Code 29 is
    'Completed (planet evaporated)', so binning it by its decade as an Error would
    drop evaporated planets from the completed cases.
    """
    from proteus.grid.summarise import status_category
    from proteus.utils.helper import CommentFromStatus

    assert status_category(0) == 'Running'
    assert status_category(9) == 'Running'
    assert status_category(10) == 'Completed'
    assert status_category(19) == 'Completed'
    assert status_category(20) == 'Error'
    assert status_category(28) == 'Error'
    assert status_category(29) == 'Completed'
    assert CommentFromStatus(29).startswith('Completed')


def test_status_category_agrees_with_every_status_comment():
    """Every documented code is binned in the category its comment names, so the
    ranges cannot drift from CommentFromStatus again.
    """
    from proteus.grid.summarise import status_category
    from proteus.utils.helper import CommentFromStatus

    documented = [c for c in range(100) if not CommentFromStatus(c).startswith('UNHANDLED')]
    assert len(documented) > 20  # the helper documents the 0-29 codes
    for code in documented:
        comment = CommentFromStatus(code)
        expected = 'Running' if comment in ('Started', 'Running') else comment.split(' ')[0]
        assert status_category(code) == expected, (code, comment)


def test_summarise_completed_category_includes_evaporated_planets(tmp_path, caplog):
    """tgt_status='Completed' lists a code 29 case (planet evaporated) and the
    'Error' listing does not.
    """
    grid = _make_grid(tmp_path, statuses=[10, 29, 22])
    with caplog.at_level(logging.INFO, logger='fwl'):
        summarise_mod.summarise(str(grid), tgt_status='Completed')
    completed = caplog.text.split('Completed cases:')[1]
    assert 'Code 29 -' in completed
    assert 'Code 22 -' not in completed
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl'):
        summarise_mod.summarise(str(grid), tgt_status='Error')
    errors = caplog.text.split('Error cases:')[1]
    assert 'Code 22 -' in errors
    assert 'Code 29 -' not in errors


def test_status_category_unknown_for_out_of_range_code():
    """A code outside the documented ranges is 'Unknown', not silently a category."""
    from proteus.grid.summarise import status_category

    assert status_category(99) == 'Unknown'
    assert status_category(-1) == 'Unknown'
