from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import PROTEUS_ROOT
from pandas.testing import assert_frame_equal

from proteus import Proteus
from proteus.utils.coupler import ReadHelpfileFromCSV

if os.getenv('CI'):
    # https://github.com/FormingWorlds/PROTEUS/pull/149
    pytest.skip(reason='No way of currently testing this on the CI.', allow_module_level=True)


def test_dummy_run():
    config_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    ref_output = PROTEUS_ROOT / 'tests' / 'data' / 'integration' / 'dummy' #/ 'runtime_helpfile.csv'

    runner = Proteus(config_path=config_path)

    runner.start()

    output = Path(runner.directories['output'])

    hf_all = ReadHelpfileFromCSV(output)
    hf_all_ref = ReadHelpfileFromCSV(ref_output)

    assert_frame_equal(hf_all, hf_all_ref, rtol=1e-3)
    #assert 'Completed' in (output / 'status').read_text()
