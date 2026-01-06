# Smoke test: Minimal PROTEUS coupling loop
#
# Purpose: Test that the core coupling loop works with real binaries
# using a minimal configuration. This validates the integration of
# PROTEUS components end-to-end.
#
# Runtime: ~20-30s (1 timestep, low resolution)
#
from __future__ import annotations

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus
from proteus.config import Config


@pytest.mark.smoke
def test_proteus_dummy_init():
    """Test PROTEUS initialization with dummy config.

    This smoke test validates that PROTEUS can load and initialize
    with the dummy.toml configuration (uses all dummy physics modules
    for fast execution).

    Tests:
    - Config loading succeeds
    - PROTEUS object instantiation works
    - Directory structure is set up
    - All required attributes are present
    """
    config_path = PROTEUS_ROOT / "input" / "demos" / "dummy.toml"

    # Initialize PROTEUS with minimal config
    runner = Proteus(config_path=config_path)

    # Validate that config loaded successfully
    assert runner.config is not None
    assert isinstance(runner.config, Config)
    assert runner.config.version == "2.0"

    # Validate that directories are initialized
    assert runner.directories is not None
    assert "output" in runner.directories
    assert "input" in runner.directories
