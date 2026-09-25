# PROTEUS agent instructions

PROTEUS couples interior, atmosphere, star, orbit and escape modules into one planet evolution model. Before a first edit:

- Tests for `src/proteus/<module>/<file>.py` go in `tests/<module>/test_<file>.py`; the test rules are in `tests/AGENTS.md`.
- Whole-planet element mass is conserved with oxygen included: after the outgassing step of every iteration, `assert_mass_conservation` checks `M_atm <= M_planet` and that `M_vol_atm` equals the summed mass of the volatile and noble-gas species (rock vapour excluded). The only relaxation is `outgas.vapourise = true` (see Oxygen and mass accounting).
- Every site that sums element masses includes oxygen; a new `if e == 'O': continue` in one of them breaks the mass budget. The sites differ on purpose in the rock-vapour elements: `M_ele` and `M_planet` sum `vol_element_list + noble_gases`, escape and structure sum `element_list`, and `check_desiccation` uses both (see `.github/agent-rules/code-review.md`).
- Energy fluxes at the interior-atmosphere boundary agree between the two modules that compute them.
- These commands decide whether a change is ready (CI runs the same; its test-quality step reports without blocking):

```bash
pytest -m "unit and not skip and not slow and not integration" --ignore=tests/examples
ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/
bash tools/validate_test_structure.sh
python tools/check_test_quality.py --check
python tools/agents/check_agents_md.py && python tools/agents/sync_core.py --check .
```

<!-- fwl-core:begin sha256=a03a2d49f7eaeaa1 -->
## PROTEUS ecosystem rules

PROTEUS couples separate module repositories into one model of the evolution of rocky planets and their atmospheres. Its config selects one module per process; the config values are the lowercase names:

- `interior_energetics.module`: `aragog`, `spider` (or the built-in `boundary`); `interior_struct.module`: `zalmoxis`, `spider`
- `atmos_clim.module`: `agni`, `janus`, both with SOCRATES spectral radiative transfer
- `outgas.module`: `calliope`, `atmodeller`; `atmos_chem.module`: `vulcan`
- `star.module`: `mors`; `escape.module`: `zephyrus`, `boreas`; `orbit.module` (tides): `obliqua`, `lovepy`
- fwl-io downloads the reference data the modules use

A module change to anything PROTEUS calls or reads (a function signature, a config key, an output column, a unit) can break the coupled model while the module's own tests pass. Search `src/proteus/` in PROTEUS for the name before you change it, and name the affected PROTEUS call sites in the pull request.

### Physics and numerics

- Units differ between and within modules: MORS and VULCAN use cgs; CALLIOPE and atmodeller work with pressures in bar; AGNI takes surface pressure in bar at setup and holds Pa inside; PROTEUS states the unit of every `hf_row` key in `GetHelpfileKeys` (`src/proteus/utils/coupler.py`: atmospheric pressures in bar, interior pressures in Pa, model time in years, orbital periods in s), and its config reference pages state each key's unit (for example stellar mass in M_sun, initial partial surface pressures in bar, stellar age in Gyr). State the unit of every physical quantity in its docstring and convert explicitly at the boundary: a unit mismatch between two modules passes the tests of both and shows only in the coupled run.
- A physics test must fail for the most plausible wrong formula: check a conservation law, a bound, a monotonicity, or a published or analytical value, and assert that the wrong result falls outside the tolerance.
- Take each physical constant from one source per repository (in Python `scipy.constants` or the module's constants file). Two retyped values of one constant differ at round-off and hide real differences between code paths.
- Do not loosen a solver tolerance, a conservation check or a clamp to make a run or a test pass. Find the cause; a loosened check also hides the next defect.

### Branches and pull requests

- Work on a feature branch `<initials>/<short-description>`; `main` changes only through a reviewed pull request.
- Fill every section of the repository's pull-request template, where it has one.
- Before you push, run the checks listed at the top of this `AGENTS.md` and in `tests/AGENTS.md`.

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
- Resume a stopped run with `proteus start -r -c <config.toml>`. It needs more than `init_loops + 1` helpfile rows and a complete pair of interior and atmosphere snapshots under the run's `data/`, named by simulation time in a form that depends on the module (`select_resumable_snapshot` in `src/proteus/utils/coupler.py`); shorter runs refuse to resume.
- `--deterministic` pins `JAX_ENABLE_X64=1` and disables XLA fast math before JAX loads (single-threaded BLAS is always on). Use it for tight-tolerance Aragog runs that are sensitive to round-off.

## Physics and coupling contract

- Units: `GetHelpfileKeys` in `src/proteus/utils/coupler.py` states the unit of every `hf_row` key (atmospheric pressures in bar, interior pressures such as `P_cmb` in Pa, model time in years, orbital periods in s); config keys state their unit in `docs/Reference/config/` (stellar mass in M_sun, planet mass in M_earth, stellar age in Gyr); module APIs expect their own. Check the unit at each of these boundaries.
- `Config` is not mutated at run time; use local variables. (Zalmoxis sets `config.orbit.module = 'dummy'` in `src/proteus/interior_energetics/wrapper.py`; do not copy it.)
- A temporary override of `hf_row` values for a module call is restored in a `finally` block; without it the helpfile records the override instead of the planet state.
- When two modules compute the same quantity (Zalmoxis core mass from the EOS, SPIDER's own `rho_core`), the second must not overwrite the first in `hf_row`.
- The main loop advances `Time` before the atmosphere step, so a comparison of `hf_row` with `hf_all.iloc[-1]` compares the new step with the previous one.
- A user value that a solver derives again gets a one-time check at the initial condition that fails loudly on a large difference. The example: `check_ic_oxygen_budget` compares the user oxygen mass with the chemistry result and fails above 50 %; it runs only with `planet.fO2_source = 'user_constant'` and an `O_mode` other than `'ic_chemistry'`.

### Oxygen and mass accounting

`planet.elements.O_mode` defaults to `'ic_chemistry'` (the initial O budget comes from the fO2-buffered chemistry); `'kg'` sets it from `O_budget` in kg, `'ppmw'` as a fraction of the volatile reservoir mass, `'FeO_mantle_wt_pct'` from the mantle FeO content. Oxygen is buffered in the chemistry step and tracked in the PROTEUS mass accounting. With `outgas.vapourise = true`, rock vapour enters `M_atm` without leaving the interior: the `M_atm <= M_planet` half of the check becomes a warning when the excess is larger than `M_vaps`, and the species-sum half and `atol_frac` stay unchanged. That non-conservation is intended. Detail: `.github/agent-rules/oxygen-accounting.md`; the aggregation sites: `.github/agent-rules/code-review.md`.

## Review

Check each change against these points; `.github/agent-rules/code-review.md` has the detail and the reasons.

- Physics: T > 0 K; P > 0 and increasing with depth; mass fractions sum to 1; the mass escaped in one step stays within its reservoir (`limit_escape_step` caps it at `ESCAPE_STEP_MAX_FRAC`); outgassing >= 0; radiative flux uses `sigma * T**4`.
- Units at the config, `hf_row` and module boundaries.
- No run-time mutation of `Config`; `hf_row` overrides restored; no echo-back overwrite between modules.
- Oxygen included at every aggregation site; rock-vapour elements kept out of `M_ele` and `M_planet`; `assert_mass_conservation` not weakened; `atol_frac` not widened.
- One-time initial-condition checks for user values that a solver derives again.
- Physical constants from one definition; PROTEUS, CALLIOPE and ZEPHYRUS each define their own.
- EOS tables: SPIDER and the Aragog entropy solver read P-S tables (complete rectangles, uniform P spacing); each Aragog P-T table is a full rectangular grid, because a grid filtered by phase makes scipy fall back to slow unstructured interpolation.
- `hf_row` against `hf_all.iloc[-1]` in the right order.
- Every attrs validator has a test with a valid and an invalid input; a validator that compares an instance with a primitive never fires.
- Tests follow `tests/AGENTS.md`.

## Code organisation

Many people edit PROTEUS in parallel, so keep changes local (full conventions: `docs/How-to/development_standards.md`):

- Files: aim for fewer than 500 lines; split past about 800 along concern boundaries. Functions: aim for fewer than 50 lines; extract helpers past about 80, and write long orchestration as named stage functions.
- A new module option gets its own `<option>.py` and a branch in that process's dispatch (its `wrapper.py`, where it has one), never a second option inside an existing file.
- Central registries (output-schema keys, config fields): one entry per line, trailing comma, grouped by module, alphabetical within a group.
- Line length: ruff allows 96; prefer fewer than 92. At most 3 levels of indentation.

## Plots and output

Verify new behaviour with plots. Write plots, data (`.txt`, `.csv`, `.npz`) and plotting scripts to `output_files/` (ignored by git) unless asked to commit a script; a committed script goes under `tests/`. Plots use the Wong colour-blind palette, a sans-serif font, inward ticks on all sides, dpi >= 150 and axis labels with units. Images committed to `docs/assets/` are AVIF (`magick in.png -quality 60 out.avif`). At the end of a plotting task, give the output folder, what each plot shows and anything unexpected.
