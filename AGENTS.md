# PROTEUS agent instructions

PROTEUS couples interior, atmosphere, star, orbit and escape modules into one planet evolution model. Before a first edit:

- Tests for `src/proteus/<module>/<file>.py` go in `tests/<module>/test_<file>.py`; the test rules are in `tests/AGENTS.md`.
- Whole-planet element mass is conserved with oxygen included: `assert_mass_conservation` checks `M_atm <= M_planet` and the species sum every iteration. The only relaxation is `outgas.vapourise = true` (see Oxygen and mass accounting).
- Every site that sums element masses uses the same element set. A new `if e == 'O': continue` in one of them breaks the mass budget.
- Energy fluxes at the interior-atmosphere boundary agree between the two modules that compute them.
- These commands decide whether a change is ready (CI runs the same):

```bash
pytest -m "unit and not skip and not slow and not integration" --ignore=tests/examples
ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/
bash tools/validate_test_structure.sh
python tools/check_test_quality.py --check
python tools/agents/check_agents_md.py
```

<!-- fwl-core:begin sha256=63b17e6b5598d839 -->
## PROTEUS ecosystem rules

This repository is one module of PROTEUS, a coupled model of the evolution of rocky planets and their atmospheres. PROTEUS selects one module per process in its config:

- interior energetics: aragog, SPIDER; interior structure: Zalmoxis, SPIDER
- atmosphere energy balance: AGNI, JANUS, both with SOCRATES spectral radiative transfer
- volatile outgassing: CALLIOPE, atmodeller; atmospheric chemistry: VULCAN
- stellar evolution: MORS; atmospheric escape: ZEPHYRUS, BOREAS; tides: Obliqua, lovepy
- fwl-io downloads the reference data the modules use

A change to anything PROTEUS calls or reads (a function signature, a config key, an output column, a unit) can break the coupled model while this module's own tests pass. Search `src/proteus/` in PROTEUS for the name before you change it, and name the affected PROTEUS call sites in the pull request.

### Physics and numerics

- Units differ between modules: MORS and VULCAN use cgs, CALLIOPE, atmodeller and AGNI take pressures in bar, and the PROTEUS config reference pages state the unit of each key (for example stellar mass in M_sun, initial partial surface pressures in bar, stellar age in Gyr). State the unit of every physical quantity in its docstring and convert explicitly at the boundary: a unit mismatch between two modules passes the tests of both and shows only in the coupled run.
- A physics test must fail for the most plausible wrong formula: check a conservation law, a bound, a monotonicity, or a published or analytical value, and assert that the wrong result falls outside the tolerance.
- Take each physical constant from one source per repository (in Python `scipy.constants` or the module's constants file). Two retyped values of one constant differ at round-off and hide real differences between code paths.
- Do not loosen a solver tolerance, a conservation check or a clamp to make a run or a test pass. Find the cause; a loosened check also hides the next defect.

### Branches and pull requests

- Work on a feature branch `<initials>/<short-description>`; `main` changes only through a reviewed pull request.
- Fill every section of the repository's pull-request template, where it has one.
- Before you push, run the checks this file and `tests/AGENTS.md` name.

### Where knowledge goes

Rules for every contributor go in the `AGENTS.md` files, the reason for a change in its commit message and pull-request description, and the scientific validation of a module in its `docs/Validation/` pages, where the repository has them. Do not add memory, notes or decision-log files to the repository: nobody maintains them, and they go stale.
<!-- fwl-core:end -->

<!-- fwl-voice:begin sha256=a943ab6c93ddad24 -->
### Commit messages and public text

Commit messages, pull-request text, code comments, docstrings and test names describe the change and the current state of the code. They name no tool used to write the change and carry no tool-attribution trailer.
<!-- fwl-voice:end -->

## Environment

- `docs/How-to/installation.md` describes the installer (`proteus install-all`); `docs/How-to/manual_installation.md` gives the manual developer steps. Install every Python module editable (`pip install -e`), and PROTEUS with `pip install -e ".[develop,vulcan,atmodeller,inference]"`: with `[develop]` alone the optional modules are absent and their tests skip.
- Python 3.12: PETSc and SPIDER do not support a later version. Linux and macOS only.
- `FWL_DATA` and `RAD_DIR` must point at populated directories before a run. Reference data downloads on first use unless `--offline` is given.
- Use one conda env per git worktree. `conda create --clone` hardlinks the editable-install pointers, so a `pip install -e .` in one env can repoint `import proteus` in another. Before an A/B comparison run `python -c "import proteus; print(proteus.__file__)"`.
- The pre-commit hook runs `ruff check --fix` but not the formatter; run `ruff format` on the files you change, because CI checks `ruff format --check`.
- SOCRATES builds with `-Ofast -march=native`, so a built tree is tied to its CPU and not bit-reproducible; `SOCRATES_PORTABLE_FLAGS=1` switches to `-O2 -fno-fast-math`. Read `.github/agent-rules/socrates-build.md` before you change `tools/get_socrates.sh` or need bit-reproducible numbers.

## Running PROTEUS

- `proteus start -c <config.toml> --offline`. Detach long runs (`nohup ... &`, output redirected into the run directory); a foreground run dies with the shell.
- Resume a stopped run with `proteus start -r -c <config.toml>`. It needs more than `init_loops + 1` helpfile rows and the archived `<iter>_int.nc` snapshot under the run's `data/`; shorter runs refuse to resume.
- `--deterministic` pins `JAX_ENABLE_X64=1` and disables XLA fast math before JAX loads (single-threaded BLAS is always on). Use it for tight-tolerance Aragog runs that are sensitive to round-off.

## Physics and coupling contract

- Units: config values are in "human" units (M_sun, bar, Gyr, K); `hf_row` holds SI values with time in years; module APIs expect either. Check the unit at each of these boundaries.
- `Config` is not mutated at run time; use local variables. (Zalmoxis sets `config.orbit.module = 'dummy'` in `interior_energetics/wrapper.py`; do not copy it.)
- A temporary override of `hf_row` values for a module call is restored in a `finally` block; without it the helpfile records the override instead of the planet state.
- When two modules compute the same quantity (Zalmoxis core mass from the EOS, SPIDER's own `rho_core`), the second must not overwrite the first in `hf_row`.
- The main loop advances `Time` before the atmosphere step, so a comparison of `hf_row` with `hf_all.iloc[-1]` compares the new step with the previous one.
- A user value that a solver derives again (the oxygen budget against CALLIOPE's equilibrium) gets a one-time check at the initial condition that fails loudly on a large difference (`check_ic_oxygen_budget`, 50 %).

### Oxygen and mass accounting

`planet.elements.O_mode` defaults to `'ic_chemistry'` (the initial O budget comes from CALLIOPE's fO2-buffered equilibrium); `'ppmw'`, `'kg'` and `'FeO_mantle_wt_pct'` set it directly. Oxygen is buffered in the chemistry step and tracked in the PROTEUS mass accounting. The sites that sum element masses are listed in `.github/agent-rules/oxygen-accounting.md`, together with the `outgas.vapourise` relaxation: in that mode rock vapour enters `M_atm` without leaving the interior, the `M_atm <= M_planet` half of the check becomes a warning when the excess is larger than `M_vaps`, and the species-sum half and `atol_frac` stay unchanged. That non-conservation is intended.

## Review

Check each change against these points; `.github/agent-rules/code-review.md` has the detail and the reasons.

- Physics: T > 0 K; P > 0 and increasing with depth; mass fractions sum to 1; escape never exceeds the atmosphere; outgassing >= 0; radiative flux uses `sigma * T**4`.
- Units at the config, `hf_row` and module boundaries.
- No run-time mutation of `Config`; `hf_row` overrides restored; no echo-back overwrite between modules.
- The same element set at every aggregation site; `assert_mass_conservation` not weakened; `atol_frac` not widened.
- One-time initial-condition checks for user values that a solver derives again.
- Physical constants from one definition; PROTEUS, CALLIOPE and ZEPHYRUS each define their own.
- EOS tables: SPIDER needs complete P-S rectangles with uniform P spacing; Aragog needs full P-T grids, identical for solid and melt, never filtered by phase.
- `hf_row` against `hf_all.iloc[-1]` in the right order.
- Every attrs validator has a test with a valid and an invalid input; a validator that compares an instance with a primitive never fires.
- Tests follow `tests/AGENTS.md`.

## Code organisation

Many people edit PROTEUS in parallel, so keep changes local (full conventions: `docs/How-to/development_standards.md`):

- Files: aim for fewer than 500 lines; split past about 800 along concern boundaries. Functions: aim for fewer than 50 lines; extract helpers past about 80, and write long orchestration as named stage functions.
- A new module option gets its own `<option>.py` and a dispatch branch in `wrapper.py`, never a second option inside an existing file.
- Central registries (output-schema keys, config fields): one entry per line, trailing comma, grouped by module, alphabetical within a group.
- Line length: ruff allows 96; prefer fewer than 92. At most 3 levels of indentation.

## Plots and output

Verify new behaviour with plots. Write plots, data (`.txt`, `.csv`, `.npz`) and plotting scripts to `output_files/` (ignored by git) unless asked to commit a script; a committed script goes under `tests/`. Plots use the Wong colour-blind palette, a sans-serif font, inward ticks on all sides, dpi >= 150 and axis labels with units. Images committed to `docs/assets/` are AVIF (`magick in.png -quality 60 out.avif`). At the end of a plotting task, give the output folder, what each plot shows and anything unexpected.
