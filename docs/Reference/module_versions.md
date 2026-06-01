# Module versions

PROTEUS pins the version of every submodule it depends on. The pins live in
[`pyproject.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/pyproject.toml)
and serve as the single source of truth for which versions are tested and
supported together. [`proteus doctor`](../How-to/doctor.md) checks your
installation against these pins, and
[`proteus update`](../How-to/doctor.md#proteus-update) aligns your
installation to them.

## Current pins

### Python packages (PyPI)

These modules are installed via `pip` and pinned with minimum version bounds
in `[project] dependencies`. Click a badge to view the pinned release.

<!-- BEGIN PYPI_TABLE -->
| Module | Role | Pin | Docs |
|--------|------|-----|------|
| fwl-janus | 1D convective atmosphere | [![fwl-janus](https://img.shields.io/badge/fwl--janus-%3E%3D24.11.05-blue)](https://pypi.org/project/fwl-janus/24.11.05/) | [Docs](https://proteus-framework.org/JANUS/) |
| fwl-mors | Stellar evolution | [![fwl-mors](https://img.shields.io/badge/fwl--mors-%3E%3D26.01.02-blue)](https://pypi.org/project/fwl-mors/26.01.02/) | [Docs](https://proteus-framework.org/MORS/) |
| fwl-calliope | Volatile outgassing | [![fwl-calliope](https://img.shields.io/badge/fwl--calliope-%3E%3D26.06.01-blue)](https://pypi.org/project/fwl-calliope/26.06.01/) | [Docs](https://proteus-framework.org/CALLIOPE/) |
| fwl-zephyrus | Atmospheric escape | [![fwl-zephyrus](https://img.shields.io/badge/fwl--zephyrus-%3E%3D25.03.11-blue)](https://pypi.org/project/fwl-zephyrus/25.03.11/) | [GitHub](https://github.com/FormingWorlds/ZEPHYRUS) |
| fwl-aragog | Interior thermal evolution | [![fwl-aragog](https://img.shields.io/badge/fwl--aragog-%3E%3D26.05.13-blue)](https://pypi.org/project/fwl-aragog/26.05.13/) | [Docs](https://proteus-framework.org/aragog/) |
| fwl-zalmoxis | Interior structure | [![fwl-zalmoxis](https://img.shields.io/badge/fwl--zalmoxis-%3E%3D26.05.13-blue)](https://pypi.org/project/fwl-zalmoxis/26.05.13/) | [Docs](https://proteus-framework.org/Zalmoxis/) |
| fwl-vulcan | Atmospheric chemistry | [![fwl-vulcan](https://img.shields.io/badge/fwl--vulcan-%3E%3D26.04.22-blue)](https://pypi.org/project/fwl-vulcan/26.04.22/) | [GitHub](https://github.com/FormingWorlds/VULCAN) |
<!-- END PYPI_TABLE -->

### Git-pinned modules (non-PyPI)

These modules are installed from source via dedicated setup scripts and pinned
to exact commit SHAs in `[tool.proteus.modules]`. Click a badge to view the
pinned commit.

<!-- BEGIN GIT_TABLE -->
| Module | Role | Pin | Docs |
|--------|------|-----|------|
| AGNI | Radiative-convective atmosphere (Julia) | [![AGNI](https://img.shields.io/badge/AGNI-b06a3fed-green)](https://github.com/nichollsh/AGNI/commit/b06a3fed51e0f1610556634d5b5a5e0425428f0e) | [Docs](https://www.h-nicholls.space/AGNI/) |
| SOCRATES | Spectral radiative transfer (Fortran) | [![SOCRATES](https://img.shields.io/badge/SOCRATES-e7133ff7-green)](https://github.com/FormingWorlds/SOCRATES/commit/e7133ff7388847c7939b38572c6e91cd05d5b755) | [Docs](https://proteus-framework.org/SOCRATES/) |
| SPIDER | Interior evolution (C, requires PETSc) | [![SPIDER](https://img.shields.io/badge/SPIDER-c9a3fd43-green)](https://github.com/FormingWorlds/SPIDER/commit/c9a3fd4301c7008291d4f4921506d36b6288f8ca) | [Docs](https://proteus-framework.org/SPIDER/) |
<!-- END GIT_TABLE -->

### Optional modules

<!-- BEGIN OPTIONAL_TABLE -->
| Module | Role | Pin | Docs |
|--------|------|-----|------|
| LovePy | Multi-phase tidal heating (Julia) | [![LovePy](https://img.shields.io/badge/LovePy-main-lightgrey)](https://github.com/nichollsh/LovePy) | [GitHub](https://github.com/nichollsh/LovePy) |
| atmodeller | Alternative outgassing backend | [![atmodeller](https://img.shields.io/badge/atmodeller-%3E%3D1.0.0-blue)](https://pypi.org/project/atmodeller/) | [GitHub](https://github.com/djbower/atmodeller) |
| BOREAS | Hydrodynamic escape | [![BOREAS](https://img.shields.io/badge/BOREAS-0174edb-green)](https://github.com/ExoInteriors/BOREAS/commit/0174edb) | [GitHub](https://github.com/ExoInteriors/BOREAS) |
| Obliqua | Orbital evolution and tides (Julia) | n/a | [GitHub](https://github.com/FormingWorlds/Obliqua) |
| PLATON | Synthetic observations | n/a | [Docs](https://platon.readthedocs.io/) |
<!-- END OPTIONAL_TABLE -->

---

## How version pinning works

PROTEUS uses two pinning mechanisms depending on the module type:

### PyPI packages (`[project] dependencies`)

Python submodules distributed on PyPI are pinned with minimum version bounds:

```toml
[project]
dependencies = [
    "fwl-aragog>=26.05.13",
    "fwl-janus>=24.11.05",
    ...
]
```

When you run `pip install -e ".[develop]"`, pip resolves the latest version
that satisfies these bounds. For editable (development) installs, the local
git checkout takes precedence over the PyPI version on `sys.path`.

### Git modules (`[tool.proteus.modules]`)

Non-PyPI modules (AGNI, SOCRATES, SPIDER) are pinned to exact commit SHAs:

```toml
[tool.proteus.modules.agni]
url = "https://github.com/nichollsh/AGNI.git"
ref = "b06a3fed51e0f1610556634d5b5a5e0425428f0e"
```

The `tools/get_*.sh` setup scripts read these pins via
`tools/_module_pins.py` and clone or checkout the pinned commit. CI uses the
same mechanism to guarantee reproducible builds.

### Bumping a version

To update a module pin:

1. **PyPI package**: edit the version bound in `[project] dependencies`
   (e.g. `fwl-aragog>=26.06.01`).
2. **Git module**: edit the `ref` field in `[tool.proteus.modules.<name>]`
   to the new commit SHA.

Both are single-line changes in `pyproject.toml`. After bumping, run
`python tools/generate_version_badges.py` to update the badge tables on
this page, then `proteus update` to apply the new pin locally.

---

**See also:** [Diagnose and update](../How-to/doctor.md) | [Installation](../How-to/installation.md) | [Model description](../Explanations/model.md)
