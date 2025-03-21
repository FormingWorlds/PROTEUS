from __future__ import annotations

import tomllib
from pathlib import Path

this_script = Path(__file__)
root = this_script.parents[1]
pyproject = root / 'pyproject.toml'

with open(pyproject, 'rb') as f:
    metadata = tomllib.load(f)
    dependencies = metadata['project']['dependencies']

with open('requirements.txt', 'w') as f:
    this_script_rel = this_script.relative_to(root)
    f.write(f'# generated by {this_script_rel}\n')
    f.write('\n'.join(dependencies))
    f.write('\n')
