# Helper functions for running the tests
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

PROTEUS_ROOT = Path(__file__).parents[2]
NEGLECT = [
    'N2_mol_atm',
    'N2_mol_total',
    'N2_kg_atm',
    'N2_kg_total',
    'CH4_mol_atm',
    'CH4_mol_total',
    'CH4_kg_atm',
    'CH4_kg_total',
    'CH4_bar',
    'F_net',
    'runtime',
]


def resize_to_match(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Resize img2 to match img1's size if they don't match
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    return img1, img2


def df_intersect(df1, df2):
    """
    Filter two dataframes to only the columns which they have in common.
    """

    # Intersection of columns
    inter = set(df1.columns).intersection(set(df2.columns))
    inter = list(inter)

    # Filter
    out1 = df1.loc[:, df1.columns.isin(inter)]
    out2 = df2.loc[:, df2.columns.isin(inter)]

    # Return
    return out1, out2


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a fixed identity and no signing; return its stdout.

    Parameters
    ----------
    cwd : Path
        Working directory of the git call.
    *args : str
        The git arguments, for example ``'rev-parse', 'HEAD'``.

    Returns
    -------
    str
        The standard output, stripped.
    """
    result = subprocess.run(
        ['git', '-c', 'user.email=t@e.st', '-c', 'user.name=t', '-c', 'commit.gpgsign=false']
        + list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def scratch_proteus_root(root: Path, dependencies: list[str]) -> Path:
    """Build a PROTEUS root with the module clone tools below a ``pyproject.toml``.

    Parameters
    ----------
    root : Path
        Directory to create; it must not exist yet.
    dependencies : list of str
        The ``[project] dependencies`` of the fixture ``pyproject.toml``.

    Returns
    -------
    Path
        ``root``, holding ``tools/_get_common.sh``, ``tools/_module_pins.py``
        and ``tools/get_python_module.sh`` copied from this tree.
    """
    (root / 'tools').mkdir(parents=True)
    for name in ('_get_common.sh', '_module_pins.py', 'get_python_module.sh'):
        shutil.copy2(PROTEUS_ROOT / 'tools' / name, root / 'tools' / name)
    deps = ''.join(f'    "{d}",\n' for d in dependencies)
    (root / 'pyproject.toml').write_text(
        f'[project]\nname = "fixture"\ndependencies = [\n{deps}]\n'
    )
    return root
