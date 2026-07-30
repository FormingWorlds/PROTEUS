# PROTEUS AI Agent Guidelines

**Trust these instructions.** Only search if information is incomplete or found to be in error.

**Identity & Mission**: You are an expert Scientific Software Engineer working on the PROTEUS ecosystem.

## Scope of These Guidelines

**These guidelines apply to ALL components in the PROTEUS ecosystem.** Whether you are working in:

- The main PROTEUS repository
- A standalone module (CALLIOPE, JANUS, MORS, etc.)
- Tests for any ecosystem component

Follow the same standards for testing, coverage, code quality, and infrastructure.

## High-Level Instructions

> ### Rule files you MUST read on every session
>
> PROTEUS keeps its Claude-Code rule files under `.github/.claude/rules/` (NOT the conventional repo-root `.claude/`, which is gitignored and so cannot be shared with collaborators). Claude Code does NOT auto-discover the rules at this unusual path. Read them explicitly at the start of every session and any time you open a related file:
>
> - [`.github/.claude/rules/proteus-tests.md`](.claude/rules/proteus-tests.md) -- test quality deep-dive: anti-happy-path patterns, discriminating-value guards, physics-invariant tiering, validation certification markers, adversarial-review trigger. **Required reading before editing any file under `tests/**` or `src/proteus/**`.**
> - [`.github/.claude/rules/proteus-code-review.md`](.claude/rules/proteus-code-review.md) -- review-pass gate, domain-aware physics review (Stefan-Boltzmann, hf_row save/restore, IC consistency, whole-element aggregation symmetry, etc.). **Required reading before any code review pass.**
>
> These two files plus this one are the canonical sources of truth for testing rigor and review criteria. Together they enforce PROTEUS's extreme-rigor stance on physics validity, anti-happy-path testing, and validation certification.

1. **Always** read the two rule files above plus the testing standards in this document and `docs/How-to/testing.md` before any code change.
2. **Always** inform the user that you are reading in this file by printing a message at the start of your response: "(Read in copilot-instructions.md...)"
3. When creating a PR, **always** follow the PR template and ensure all sections are filled out with relevant information.
4. **Claude-specific**: `CLAUDE.md` is a symlink to this file. Session learnings, plans, and memories live in `~/.claude/projects/<repo>/memory/` (per the global voice rule); they do NOT live in this repository.

## Ecosystem Structure

PROTEUS is a coupled atmosphere-interior framework with a modular architecture:

- **[PROTEUS](https://github.com/FormingWorlds/PROTEUS)** (main repository): Core coupling framework and orchestration
- **[AGNI](https://github.com/nichollsh/AGNI)**: Radiative-convective atmospheric energy module (Julia)
- **[SOCRATES](https://github.com/nichollsh/SOCRATES)**: Spectral radiative transfer code (Fortran)
- **[CALLIOPE](https://github.com/FormingWorlds/CALLIOPE)**: Volatile in-/outgassing and thermodynamics module (Python)
- **[JANUS](https://github.com/FormingWorlds/JANUS)**: 1D convective atmosphere module (Python)
- **[MORS](https://github.com/FormingWorlds/MORS)**: Stellar evolution module (Python)
- **[ARAGOG](https://github.com/FormingWorlds/aragog)**: Interior thermal evolution module based on T-P formalism (Python)
- **[SPIDER](https://github.com/FormingWorlds/SPIDER)**: Interior thermal evolution module based on T-S formalism (C)
- **[VULCAN](https://github.com/FormingWorlds/VULCAN)**: Atmospheric chemistry module (Python)
- **[ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS)**: Atmospheric escape module (Python)
- **[BOREAS](https://github.com/FormingWorlds/BOREAS)**: Hydrodynamic atmospheric escape module (Python)
- **[Obliqua](https://github.com/FormingWorlds/Obliqua)**: Tidal evolution module (Julia)

**Important:** Each module is maintained in its own GitHub repository but is typically cloned/installed within the PROTEUS directory structure for integrated development. When working on any module in the ecosystem, apply these guidelines consistently.

## Build & Validation

For installation instructions and dependency management across the ecosystem:

- **Main installation guide:** `docs/How-to/installation.md` - Standard user and developer installation procedures
- **Local machine setup:** `docs/How-to/local_machine_guide.md` - Platform-specific setup (macOS, Linux, Windows)
- **Cluster setup:** `docs/How-to/kapteyn_cluster_guide.md` - HPC cluster configuration (see also `docs/How-to/habrok_cluster_guide.md`, `docs/How-to/snellius_cluster_guide.md`)

When helping with installation or dependency issues, always reference these guides first. The `proteus install-all` command handles most submodule installations automatically. However, whenever possible, prefer the developer installation steps outlined in the installation guide for editable installs.

### Environment Setup

Follow the developer install in `docs/How-to/installation.md` step by step; it is
the single source of truth for the clone order, the PETSc pin, and the optional
modules. The traps that guide does not call out:

- **FWL_DATA** and **RAD_DIR** must be set before running PROTEUS
- **PETSc** is downloaded as a specific pre-compiled version from OSF (not built from source)
- **SPIDER** requires PETSc to be installed first
- All Python submodules should be installed as editable (`-e`) for development
- After each file change or edit, ruff format all changed files with `ruff check --fix ` and `ruff format --check`
- **Parallel tracks**: one conda env per git worktree. `conda create --clone` hardlinks pip-editable pointers, so a subsequent `pip install -e .` in one env can silently repoint another env's `import proteus`. Canary before each A/B run: `python -c "import proteus; print(proteus.__file__)"`.

### Build Commands

**No explicit build step** for Python code (installed via `pip install -e .`). Submodules require compilation:

- **SOCRATES** (Fortran): `cd socrates && ./build_code`
- **SPIDER** (C): `cd SPIDER && make`
- **AGNI** (Julia): `julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'`

**Always run** `pip install -e ".[develop]"` after code changes to update installation.

#### SOCRATES build flags

By default `tools/get_socrates.sh` keeps the configure flags `-Ofast
-march=native`, which give the best performance on the build host. Set
`SOCRATES_PORTABLE_FLAGS=1` to compile with `-O2 -fno-fast-math` instead: the
`-march=native` default bakes the build host's CPU extensions into the binary,
so a compiled tree reused on a different processor aborts with an
illegal-instruction fault, while the portable flags run on any CPU. CI sets the
switch because its caches are restored across runner machines with mixed CPU
generations. Dropping fast-math also removes compiler value reordering, the
build-to-build component of the ULP-level non-determinism that AGNI's Newton
solver amplifies into 1-2 % F_atm variance; run-to-run scatter from OpenMP
threading remains while OMPARG is set. In portable mode the build fails loudly
if a future SOCRATES release changes the flag string, so no manual edit is
needed.

For full bit-reproducibility (paper plots, CHILI, SPIDER-parity) install with
`SOCRATES_PORTABLE_FLAGS=1 bash tools/get_socrates.sh` and also clear
`OMPARG = -fopenmp` in `socrates/make/Mk_cmd`, then force a recompile with
`cd socrates/bin && make clean && cd .. && ./build_code` (no make rule depends
on `Mk_cmd`, so rebuilding without the clean reuses the OpenMP objects
unchanged); the install path keeps OpenMP enabled and does not clear it
automatically.

### Test Commands

Tier selection is by marker (`pytest -m unit`, `-m smoke`, `-m integration`). The
one invocation that is not guessable is the PR filter, which needs the extra
clauses to keep slow-marked files out:

```bash
pytest -m "unit and not skip and not slow and not integration" --ignore=tests/examples
```

Coverage thresholds live in `pyproject.toml` and are described under `Coverage
architecture` in the Testing Standards section below. Repo-specific checks:
`bash tools/validate_test_structure.sh` and `bash tools/coverage_analysis.sh`.

### Validation Pipeline

**CI runs on PRs** (`.github/workflows/ci-pr-checks.yml`):

1. **Unit tests**: `pytest -m "unit and not skip and not slow and not integration" --cov=src --cov-fail-under=$FAST_COV_FAIL_UNDER` (CI reads the fast threshold from `pyproject.toml` `[tool.proteus.coverage_fast]`; fixed at 80%, warn-only on draft PRs). The PR cycle runs the unit tier only; smoke, integration, and slow run nightly.
2. **Lint**: `ruff check src/ tests/` and `ruff format --check src/ tests/`
3. **Diff-cover**: 80% on changed lines, fast-suite coverage unioned with the latest nightly coverage (enforced)
4. **Estimated-total coverage**: PR unit coverage unioned with the latest nightly, measured against 90%
5. **Test structure**: `bash tools/validate_test_structure.sh`
6. **Editable install**: verifies the package installs correctly
7. **Test quality**: `python tools/check_test_quality.py --check` (reported, does not block)

**All must pass** before merge. Coverage gates are warn-only on draft PRs and block once the PR is ready for review. Coverage ceilings are fixed (fast 80%, full 90%) and never lowered.

## Testing Standards

PROTEUS is scientific simulation code, so the test suite is held to physics-grade rigor. The rules below are the contract; the deep-dive (anti-happy-path patterns, discriminating-value guards, certification markers, adversarial-review trigger) lives in [`.github/.claude/rules/proteus-tests.md`](.claude/rules/proteus-tests.md). Read that file before editing any test file or any source file under `src/proteus/**`. The two files must be kept in sync; if you change one, mirror the change in the other.

### Structure

- Tests MUST mirror source exactly: `src/proteus/<module>/<file>.py` -> `tests/<module>/test_<file>.py`. Validated by `bash tools/validate_test_structure.sh`.
- Framework: `pytest` exclusively in the `tests/` directory.
- Shared fixtures: `tests/conftest.py` (parameter classes `EarthLikeParams`, `UltraHotSuperEarthParams`, `IntermediateSuperEarthParams`; config paths `config_earth`, `config_minimal`, `config_dummy`, ...). Read `tests/conftest.py` before writing new tests.

### Markers and the module-level marker rule

Tier markers, with their CI surface and per-test wall-time budgets:

| Marker | What it tests | Speed budget | When CI runs it |
|---|---|---|---|
| `@pytest.mark.unit` | Python logic, heavy physics mocked | < 100 ms per test | Every PR (`unit and not skip and not slow and not integration`) |
| `@pytest.mark.smoke` | Real binaries, 1 timestep, low resolution | < 30 s per test | Nightly only (`smoke and not skip and not slow and not integration`) |
| `@pytest.mark.integration` | Multi-module coupling | Minutes per test | Nightly only |
| `@pytest.mark.slow` | Full physics validation | Up to hours per test | Nightly only (targeted file list) |
| `@pytest.mark.skip` | Placeholder, deliberately disabled | n/a | Never |

**Mandatory module-level marker** (no exceptions): every test file begins with

```python
pytestmark = [pytest.mark.<tier>, pytest.mark.timeout(<budget>)]
```

with timeouts: 30 s for unit, 60 s for smoke, 300 s for integration, 3600 s for slow. Per-function markers are additive but do not replace the module-level marker. CI runs `pytest -m "unit and not skip and not slow and not integration"`; tests without a tier marker are invisible to CI. The `pytest-timeout` ceiling is a defensive net against future regressions that introduce a hang; current budgets are well clear of it.

### Physics validity (tiered)

Every unit test on a **physics module** must assert at least one of the following invariants. Physics modules are: `interior_struct/*`, `interior_energetics/*`, `atmos_clim/*`, `atmos_chem/*`, `escape/*`, `outgas/*`, `orbit/*`, `star/*`, `observe/*`, `inference/objective.py`, `inference/BO.py`, `inference/async_BO.py`. Helpers under physics directories (e.g. `escape/common.py`, `outgas/common.py`) are physics-required when their outputs feed physics; the utility exemption applies only to pure structural plumbing (logging, path resolution, type coercion with no physical quantity). See `.github/.claude/rules/proteus-tests.md` section 3 for the source-file-purpose carve-out.

- **Conservation**: mass closure (sum of reservoirs = total), energy balance (LHS = RHS within tolerance), angular-momentum conservation.
- **Positivity / boundedness**: T > 0 K, P > 0 Pa, mass fractions in [0, 1], escape rate <= atmospheric mass, outgassing >= 0, melt fraction in [0, 1].
- **Monotonicity or symmetry**: e.g. P increasing with depth implies rho increasing; reversing time integration recovers the initial condition.
- **Pinned numeric value with a discrimination guard**: a closed-form value pinned via `pytest.approx`, accompanied by an explicit assertion that the most plausible wrong-formula result would differ from the correct one by more than the tolerance. The deep-dive file has examples.

Utility modules (`utils/*`, `config/*`, `plot/*`, `cli.py`, `inference/utils.py`, `tools/*`) are **exempt** from the physics-invariant requirement but still subject to the anti-happy-path rules (edge case, adversarial input handling where applicable, non-trivially-derivable assertion values).

Tag every test that asserts a physical invariant with `@pytest.mark.physics_invariant` so coverage of physics-invariant tests can be tracked separately from line coverage. The marker is per-function, NOT module-level: structural tests (ordering, autonomy, mutation-in-place, pass-through assignment) in a physics-module test file should NOT carry the marker. Reference-pinned tests carry both `@pytest.mark.reference_pinned` and `@pytest.mark.physics_invariant` (the published-value pin is itself the invariant). Granularity of the reference-pinned requirement is **per source file**, not per directory: `interior_energetics/aragog.py` and `interior_energetics/spider.py` each need their own pinned test, even though they live in the same directory. Per-source-file inventory is tracked in `docs/Validation/<module>/<file>.md`; the `tools/check_test_quality.py --reference-pinned-audit` command currently reports at directory granularity and may lag the per-file contract.

### Anti-happy-path rules (every new test)

Every new test function MUST include:

1. **At least one edge case** (boundary value, empty input, extreme physical parameter).
2. **At least one path that exercises the error contract**: a documented exception path, a guard return, or a graceful clamp. If the function under test has no validation logic, exercise the limit-input behavior (e.g. `e = 0` for an eccentricity-dependent routine) and assert the mathematical invariant.
3. **Assertion values that are NOT trivially derivable from the implementation**: discriminating numeric pins (not `T = 1` where every exponent gives 1), property-based assertions (monotonicity, conservation) preferred over point checks.

**Forbidden patterns** (these will be flagged by `tools/check_test_quality.py`):

- Single-assert test functions.
- Standalone weak assertions: `assert result is not None`, `assert result > 0`, `assert len(result) > 0`, `assert isinstance(result, dict)` as the only meaningful check.
- Tests with no function-level docstring.
- Tests using `==` adjacent to float literals.
- Tests asserting on a fixture's implicit default (e.g. `assert fixture is None` where the fixture returns `None` implicitly): a trivially-true test is worse than no test.

### Validation certification (marker policy)

Two markers track validation quality independently of line coverage:

- `@pytest.mark.physics_invariant` -- this test asserts at least one of the four invariants above. Every physics-module test should carry this marker if it qualifies.
- `@pytest.mark.reference_pinned` -- this test pins behavior against a **published benchmark** (cite the paper, figure, table), an **analytical limit** (e.g. the Stefan-Boltzmann black-body limit), or a **cross-implementation cross-check** (e.g. SPIDER vs Aragog at the same IC). Each physics module under `interior_*`, `interior_energetics/*`, `interior_struct/*`, `atmos_*`, `escape/*`, `outgas/*`, `orbit/*`, `star/*` must contain at least one `reference_pinned` test. Module-level inventory tracked in `docs/Validation/<module>.md`.

The new markers are registered in `pyproject.toml`. They do not gate CI by themselves; they are tracked via `tools/check_test_quality.py` for visibility.

### Float and numerical comparison

- NEVER use `==` for floats. Use `pytest.approx(val, rel=1e-5)` (or `abs=...`) or `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)`.
- State the tolerance rationale in a comment when the choice is non-obvious (e.g. "rtol=1e-3 because Cp lookup truncates to 4 sig fig").
- For pinned numeric values, include a **discrimination guard**: a follow-up `assert` showing the wrong-formula value would differ from the correct one by more than the tolerance. See `.github/.claude/rules/proteus-tests.md` Section 2 for the canonical pattern.

### Mocking discipline

- Default to `unittest.mock` for ALL external calls in unit tests: SOCRATES, AGNI, SPIDER, file I/O, network, Aragog/Zalmoxis solvers.
- Mock at the narrowest scope: a specific function, not a whole module.
- A mocked physics function must return **physically plausible** values; a mock that returns `0.0` or `1.0` for everything can mask real bugs.
- NEVER mock the function under test.
- Smoke tests use real binaries; integration tests use real submodules.

### Optional-dependency imports

Any test that imports an optional dependency (`hypothesis`, `boreas`, `atmodeller`, `lovepy`, `mors`, `vulcan`, the `inference` extra's `torch` / `botorch` / `gpytorch`, also `zalmoxis` when not installed via editable) MUST call `pytest.importorskip('<dep>')` at module top. The PR Docker image is built with `pip install --no-deps`; tests that import optional deps unconditionally will fail to collect on CI even though they run locally. This trap has recurred multiple times and is now lint-enforced.

### Module-level constants and `monkeypatch`

When the source under test reads an environment variable into a module-level constant at import time, e.g.

```python
FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', ...))
```

`monkeypatch.setenv` is **not sufficient**: the constant is frozen at the import that already happened. Patch the constant directly:

```python
monkeypatch.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path, raising=False)
```

Patch BOTH the env var (for downstream code that re-reads it) AND the constant (for code that reads only the constant).

### Voice rule for test artifacts

The repo-wide voice rule (zero AI-process disclosure in any public artifact, see top of this file) applies to test code with the same strictness as to source. The rule is scoped to artifacts other contributors and external readers see: test-skip reasons, test-file/function docstrings, test-function/class names, parametrize ids, log-capture assertions, **commit messages on test-touching commits, pull-request titles and bodies on test-touching PRs**, GitHub Actions job/step names, inline `src/proteus/**` comments, and shipped log strings. Out of scope: the rule documents themselves (this file, `.github/.claude/rules/proteus-tests.md`, `.github/.claude/rules/proteus-code-review.md`, `docs/How-to/test_*.md`) may legitimately name the procedures they define. Banned phrases inside in-scope artifacts: "audit", "review pass", "adversarial review", "Phase X" (AI-roadmap labels), "T1.x", "Group A/B/C/D" (AI work groups), `claude-config/...` paths, "Generated with Claude", em-dashes, en-dashes (except bibliographic page ranges). Write the OUTCOME, never the PROCESS.

### Speed and determinism

- Unit tests: < 100 ms wall-time each. The 30 s `timeout` is a defensive ceiling, not the target.
- Aggressively mock heavy simulations, file I/O, and external APIs in unit tests.
- Set seeds for any randomness: `np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`. All three must be seeded if all three are exercised; deterministic-only-on-one is a known regression vector.
- Use `tmp_path` (pytest fixture) for temporary files; do not produce large outputs in tests.

### Documentation per test

- File-level docstring: name the module under test, list the invariants and contract clauses the file exercises, and link to the test docs (`docs/How-to/testing.md`, `docs/Explanations/test_framework.md`).
- Function-level docstring: state the physical scenario or contract clause being verified, in plain language. Required (lint-enforced).
- Inline comments: explain **why** a specific input range was chosen ("T=300 K and T=1500 K so the T**3 vs T**4 difference is resolved well above tolerance").

### Independent review trigger

A pull request that adds or substantially modifies > 50 lines of test code across all its commits triggers an independent review pass before merge. The denominator is PR-level (`git diff origin/main...HEAD -- 'tests/**'`), not per-commit; splitting into many sub-50-line commits does not dodge the trigger. The reviewer cites the anti-happy-path rule, the discrimination-guard requirement, and the physics-invariant tier; flags single-assert tests, weak `is not None` patterns, missing module-level marker, missing `physics_invariant` tag on a physics-module test, and dead tests (tests that pass for the wrong reason).

### Tooling

- Validate test structure: `bash tools/validate_test_structure.sh`
- Test-quality lint (anti-happy-path, marker, weak assertions): `python tools/check_test_quality.py --check`
- Baseline (run after a deliberate sweep): `python tools/check_test_quality.py --baseline`
- Coverage analysis: `bash tools/coverage_analysis.sh`
- Format: `ruff format src/ tests/`
- Lint: `ruff check src/ tests/`

### Coverage architecture

PROTEUS runs three gates on every pull request. The full gate is not a fourth: it is the pyproject value the estimated total is measured against.

| Gate | Tests included | Target | Enforced |
|---|---|---|---|
| Fast gate (`tool.proteus.coverage_fast.fail_under`) | unit-only (PR) | Fixed **80%** | Every PR (warn-only on drafts) |
| Estimated total (PR unit coverage union with latest nightly artifact) | unit + smoke + integration + slow | **90%**, the `tool.coverage.report.fail_under` value | Every PR (warn-only on drafts) |
| Diff-cover | changed lines (fast + nightly union) | 80% (hard-coded; warn-only on drafts) | Every PR |

**What this means for contributors**: the coverage ceilings are fixed, not ratcheting: the fast (unit-only) gate is held at **80%** and the full gate at the **90%** PROTEUS-ecosystem target (`tools/update_coverage_threshold.py` enforces `CEILINGS = {'fast': 80.0, 'full': 90.0}` and the PR threshold guard fails if either is edited away from its fixed value). Unit tests alone are not expected to reach 90% because wrapper code that requires real binaries (SOCRATES, AGNI, SPIDER) is exercised only by the nightly tiers; that is why the fast gate sits at 80, not 90. The 90% target is reached via the estimated-total: the PR's unit coverage is unioned with the latest nightly artifact and compared against the full gate, and the diff-cover gate unions the fast and nightly reports the same way. Coverage gates run on draft PRs for visibility but are warn-only there; they block once the PR is marked ready for review.

Reports: `pytest --cov=src --cov-report=html` and open `htmlcov/index.html`. Module-level analysis: `bash tools/coverage_analysis.sh`. Diff-cover thresholds are documented in `docs/How-to/testing.md`.

## Safety & Determinism
- **Randomness:** Explicitly set seeds (e.g., `np.random.seed(42)`) in tests.
- **Files:** Do not generate tests that produce large output files (unless explicitly instructed); use `tempfile` or mocks.

## Verification and Diagnostic Plots

When testing new routines, reviewing behavior, or investigating edge cases across any PROTEUS ecosystem module:

- **Always produce plots** that verify the requested behavior. Plots are the primary verification artifact for scientific simulation code.
- **Store all generated plots and data in gitignored folders.** Use `output_files/` (already in `.gitignore`). Never commit generated plots or simulation output to the repository.
- **Store raw simulation data** alongside plots (same gitignored folder) when feasible (up to a few hundred MB). Formats: `.txt`, `.csv`, or `.npz`. This allows replotting without re-running.
- **Store plot-generating scripts in gitignored folders** unless the user explicitly asks to commit them. If committing, place in `src/tests/`.
- **At the end of a plotting task**, report: (1) output folder path, (2) what each plot shows, (3) notable findings or anomalies.
- **Plot standards**: matplotlib with Wong colorblind-friendly palette, sans-serif font (Helvetica/Arial), inward ticks on all sides, `dpi >= 150`, clear axis labels with units, legends, descriptive titles.
- **Documentation images**: use AVIF format (not PNG) for all plots committed to `docs/assets/`. AVIF is 3-5x smaller than PNG at equivalent quality. Convert with `magick input.png -quality 60 output.avif`. Reference in markdown as `![alt](path.avif)`.

## Code Quality

Ruff enforces the mechanical style rules; its config in `pyproject.toml` is the
source of truth. Two conventions ruff does NOT enforce:

- Line length: the config allows 96, but prefer < 92.
- Max indentation 3 levels.

### Code organization

PROTEUS is edited by many contributors in parallel; organise code so changes
stay local. Full conventions: `docs/How-to/development_standards.md`.

- Files: aim < 500 lines; split past ~800 along concern boundaries.
- Functions/methods: aim < 50 lines; extract helpers past ~80. Express long
  orchestration as named stage functions, not one inline body.
- New backend: add a new `<backend>.py` plus a dispatch branch in `wrapper.py`;
  never append a second backend into an existing backend file.
- Central registries (output-schema keys, config fields): one entry per line,
  trailing comma, grouped by module, alphabetical within group.
- Add to shared files narrowly: a stage function over an inline edit; a column
  in its module's group over the end of the global list.

## Important Notes

- **Data**: Large input files stored on Zenodo/OSF, downloaded automatically on first run (unless `--offline`).
- **CI caching**: ubuntu-latest + macos-latest runners with `actions/cache` for SOCRATES build, Julia depot, FWL_DATA, AGNI clone, pip wheels. Composite action `.github/actions/setup-proteus` handles platform-aware setup. Cache keys derive from `[tool.proteus.modules]` in pyproject.toml plus `.github/data-manifest.yaml`.
- **Test placeholders**: Some tests marked `@pytest.mark.skip` are placeholders. Excluded from CI.
- **Windows**: Not supported. Linux/macOS only.
- **Python version**: Must be 3.12 (PETSc/SPIDER require Python <= 3.12).

## Whole-planet oxygen accounting (issue #677)

Every config must declare an explicit `planet.elements.O_mode`. Four valid modes:

- `"ic_chemistry"`: defer the IC O budget to CALLIOPE's fO2-buffered equilibrium. Preserves pre-fix behaviour; backwards-compatible.
- `"ppmw"`, `"kg"`: parallel to the H/C/N/S modes; sets O_kg directly.
- `"FeO_mantle_wt_pct"`: alternative unit for petrologists. The number is interpreted as `O_kg = M_mantle * (wt% / 100) * (M_O / M_FeO)`. The mantle EOS density is NOT modified; PALEOS still assumes its built-in FeO content. The mode is a unit-of-convenience for setting the volatile-O budget in familiar terms.

Under D1A (the chosen design), CALLIOPE / atmodeller chemistry is unchanged. Oxygen is treated as a buffered element at the chemistry step but a tracked element in PROTEUS-side mass accounting. The asymmetry that previously let `M_atm > M_planet` at high H budgets is closed by including O in M_ele, in the Zalmoxis dry-mass subtraction, in the proportional escape distribution, and in the desiccation gate. Escape includes O in the unfractionated partitioning so `sum(esc_rate_e) == esc_rate_total` to within rounding. The runtime invariant `M_atm <= M_planet` is enforced via `assert_mass_conservation` in the main loop. An IC consistency check (`check_ic_oxygen_budget`, called once after the first outgas call) hard-fails on >50% divergence between user-supplied O_budget and CALLIOPE's equilibrium value.

## Documentation References

- **Testing**: `docs/How-to/testing.md`, `docs/Explanations/test_framework.md`
- **Installation**: `docs/How-to/installation.md`, `docs/How-to/local_machine_guide.md`
- **Usage**: `docs/How-to/usage.md`, `docs/How-to/config.md`
- **Docs development**: `docs/How-to/documentation.md` (build/serve with `zensical serve`)
- **Copilot guidelines**: `.github/copilot-instructions.md` (this file; applies to all ecosystem modules)

## Project memory and session learnings

Session-specific knowledge (debugging logs, design rationale, sprint focus, ADR drafts) lives outside this repository, in the Claude memory tree under `~/.claude/projects/<project>/memory/`. The previous in-repo `copilot-memory.md` file was retired in favor of that location because Claude's memory tree is per-user, sync-ready across machines, and not exposed in public commit history.

What still lives in this repository:

- Architectural decisions that affect every contributor: this file (`.github/copilot-instructions.md`).
- Test and review rules: `.github/.claude/rules/proteus-tests.md` and `.github/.claude/rules/proteus-code-review.md`.
- Per-PR rationale: PR descriptions.
- Per-commit rationale: commit messages.
- Module-level scientific validation: `docs/Validation/<module>.md` (created when the first `@pytest.mark.reference_pinned` test for that module is added).

Do not introduce a new in-repo "memory" or "decisions log" file. The four channels above are the contract.

---

## Quick Reference

```bash
# Setup ("[develop]" alone leaves the optional backends and the inference
# stack out, and their tests then skip; the extras below match CI)
pip install -e ".[develop,vulcan,atmodeller,inference]"

# Run simulation (detached; add -r / --resume to continue a killed run,
# add --deterministic for numerically fragile coupled runs)
nohup proteus start -c <cfg.toml> --offline > output/<run>/launch.log 2>&1 & disown
```

Resume requires `len(hf_all) > init_loops + 1` and the archived `<iter>_int.nc` snapshot under `data/`; see `src/proteus/proteus.py` ~395-430. Never foreground a multi-hour run; plain `&` alone dies on SIGHUP. `--deterministic` self-re-execs to pin `JAX_ENABLE_X64=1` + `XLA_FLAGS=--xla_cpu_enable_fast_math=false` before JAX import (on top of always-on `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=1`); use when Aragog hits T_core-jump-guard exhaustion on tight-tol runs.  **Remember**: Trust these instructions. Only search if information is incomplete or found to be in error.

---

> **⚠️ FILE SIZE LIMIT: This file must stay below 750 lines.** Enforced by pre-commit hook (`tools/check_file_sizes.sh`). File located at `.github/copilot-instructions.md`.
>
> This file is loaded into every session in full, so length is a running cost, not
> just a limit to stay under. Content a session could reconstruct by reading the
> repo (directory layouts, dependency lists, standard tool invocations, anything
> a lint config or CI check already enforces) belongs in the codebase or `docs/`,
> not here. What belongs here: gotchas, non-standard conventions, design
> rationale, and prohibitions.
