from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT

from proteus import Proteus

if os.getenv('CI'):
    # https://github.com/FormingWorlds/PROTEUS/pull/149
    pytest.skip(reason='No way of currently testing this on the CI.', allow_module_level=True)


def test_dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.config['log_level'] = 'WARNING'
    runner.config['plot_iterfreq'] = 0
    runner.config['iter_max'] = 0

    runner.start()

    output = Path(runner.directories['output'])

    assert 'Completed' in (output / 'status').read_text()
