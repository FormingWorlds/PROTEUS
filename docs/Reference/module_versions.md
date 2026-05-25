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

{{ pypi_versions_table }}

### Git-pinned modules (non-PyPI)

These modules are installed from source via dedicated setup scripts and pinned
to exact commit SHAs in `[tool.proteus.modules]`. Click a badge to view the
pinned commit.

{{ git_versions_table }}

### Optional modules

{{ optional_versions_table }}

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
`proteus update` to apply the new pin locally, or let CI pick it up on
the next push. The badge tables on this page update automatically at the
next documentation build.

---

**See also:** [Diagnose and update](../How-to/doctor.md) | [Installation](../How-to/installation.md) | [Model description](../Explanations/model.md)
