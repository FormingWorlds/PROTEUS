# Test PROTEUS doctor module
from __future__ import annotations

import pytest
from packaging.version import Version


@pytest.mark.unit
def test_version_comparison_no_downgrade():
    """Test that version comparison correctly identifies updates vs downgrades.

    This is the fix for issue #646 where 25.11.19 -> 25.10.15 was incorrectly
    suggested as an "upgrade" when it's actually a downgrade.
    """
    # These should NOT suggest an update (current >= latest)
    test_cases_no_update = [
        ('25.11.19', '25.10.15'),  # The original bug case
        ('2.0.0', '1.9.9'),
        ('1.0.1', '1.0.0'),
        ('1.0.0', '1.0.0'),  # Same version
    ]

    for current, latest in test_cases_no_update:
        current_ver = Version(current)
        latest_ver = Version(latest)
        needs_update = current_ver < latest_ver
        assert not needs_update, f"Should NOT suggest update from {current} to {latest}"

    # These SHOULD suggest an update (current < latest)
    test_cases_needs_update = [
        ('25.10.15', '25.11.19'),
        ('1.0.0', '2.0.0'),
        ('1.0.0', '1.0.1'),
        ('1.9.9', '2.0.0'),
    ]

    for current, latest in test_cases_needs_update:
        current_ver = Version(current)
        latest_ver = Version(latest)
        needs_update = current_ver < latest_ver
        assert needs_update, f"SHOULD suggest update from {current} to {latest}"


@pytest.mark.unit
def test_version_comparison_with_v_prefix():
    """Test that version comparison handles 'v' prefix correctly."""
    # Git tags often have 'v' prefix
    current = 'v1.0.0'
    latest = 'v1.0.1'

    current_ver = Version(current.lstrip('v'))
    latest_ver = Version(latest.lstrip('v'))

    assert current_ver < latest_ver
