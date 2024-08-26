from __future__ import annotations

from pathlib import Path

from helpers import PROTEUS_ROOT

from proteus import Proteus


def test_dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

    runner = Proteus(config_path=config_path)

    runner.config['log_level'] = 'WARNING'
    runner.config['plot_iterfreq'] = 0
    runner.config['iter_max'] = 0

    runner.start()

    output = Path(runner.directories['output'])

    assert 'Completed' in (output / 'status').read_text()
