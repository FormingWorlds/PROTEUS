# Test Categorization for CI/CD

This document explains how PROTEUS tests are categorized and how they flow through the CI/CD pipeline.

> **Related documentation:** For complete testing infrastructure details including test structure, configuration, and ecosystem rollout, see [Testing Infrastructure](test_infrastructure.md).

Also see the consolidated CI status and planning in [CI/CD Status and Roadmap](test_infrastructure.md#cicd-status-and-roadmap-as-of-2026-01-27).

## Test Categories

All tests in PROTEUS are marked with pytest markers to enable targeted test selection:

### @pytest.mark.unit

**Purpose**: Fast validation of Python logic with mocked physics  
**Runtime**: <100ms per test (target)

**Count**: 480+ tests (as of 2026-01-27)  
**Runs In**: `ci-pr-checks.yml` — `pytest -m "unit and not skip"` (PR validation, ~2–5 min total)  
**Coverage**: Fast gate 32.03% in `pyproject.toml` `[tool.proteus.coverage_fast]`; Python interfaces, configuration, utilities, wrapper modules

**Implemented areas** (examples):

- `tests/config/` — Configuration system, defaults, validators, converters (many tests)
- `tests/atmos_clim/test_common.py`, `test_atmos_clim.py` — Atmosphere common and wrapper
- `tests/utils/` — Data, logs, helper, coupler, plot, terminate
- `tests/test_cli.py`, `tests/test_init.py` — CLI and package init
- `tests/plot/test_cpl_colours.py`, `test_cpl_helpers.py` — Color mapping and helpers
- `tests/star/test_star.py` — Stellar physics and instellation
- `tests/escape/`, `tests/outgas/`, `tests/observe/`, `tests/interior/`, `tests/orbit/`, `tests/atmos_chem/` — Module wrappers


### @pytest.mark.smoke

**Purpose**: Quick validation that binaries work with new Python code  
**Runtime**: <30s per test (target)

**Count**: Multiple tests; CI runs `pytest -m "smoke and not skip"`. Some are skipped (env/instability).  
**Runs In**: `ci-pr-checks.yml` after unit tests (~5–10 min total with unit)

**Implemented tests** (some skipped in CI):

- `tests/integration/test_smoke_minimal.py` — PROTEUS dummy init
- `tests/integration/test_smoke_atmos_interior.py` — Dummy atmosphere + dummy interior; JANUS/AGNI tests **skipped** (binaries)
- `tests/integration/test_smoke_modules.py` — Escape, star, orbit, outgas, full dummy chain
- `tests/integration/test_smoke_janus.py` — JANUS–interior coupling (**skipped**: runtime instability)
- `tests/integration/test_smoke_outgassing.py` — CALLIOPE outgassing (**skipped**: slow, reserved for nightly)


### @pytest.mark.integration

**Purpose**: Multi-module coupling and interaction tests  
**Runtime**: Minutes to hours

**Runs In**: `ci-nightly-science.yml` and `ci-nightly-science-v5.yml` (integration jobs and science-validation)

**Implemented tests** (examples):

- `tests/integration/test_integration_dummy.py` — PROTEUS with dummy modules (4 tests)
- `tests/integration/test_integration_std_config.py` — Standard config (ARAGOG+AGNI+CALLIOPE+ZEPHYRUS+MORS); some tests also `@pytest.mark.slow`
- `tests/integration/test_integration_aragog_agni.py` — ARAGOG–AGNI coupling
- `tests/integration/test_integration_aragog_janus.py` — ARAGOG–JANUS coupling
- `tests/integration/test_integration_dummy_agni.py` — Dummy + AGNI
- `tests/integration/test_integration_multi_timestep.py` — Multi-timestep
- `tests/integration/test_integration_calliope_multi_timestep.py` — CALLIOPE multi-timestep
- `tests/integration/test_albedo_lookup.py` — Albedo interpolation
- `tests/grid/test_grid.py`, `tests/inference/test_inference.py` — Grid and inference (integration)


### @pytest.mark.slow

**Purpose**: Full scientific validation with comprehensive simulations  
**When to use**: Tests that simulate Earth magma ocean, Venus greenhouse, Super-Earth evolution, or long multi-module runs.

**Runs In**: `ci-nightly-science.yml` (science-validation job) and `ci-nightly-science-v5.yml` (slow std_config run)

**Implemented slow tests**:

- `tests/integration/test_integration_std_config.py` — Standard-config evolution (ARAGOG+AGNI+CALLIOPE+ZEPHYRUS+MORS); marked `@pytest.mark.integration` and `@pytest.mark.slow`, run in nightly CI only.

**Future/planned scenarios** (budget ~3–4 h in nightly):

- Earth magma ocean solidification (1–4 h)
- Venus runaway greenhouse (30 min–2 h)
- Super-Earth interior evolution (2–6 h)

Module-level tests (`tests/escape/`, `tests/orbit/`, `tests/interior/`, etc.) are mostly unit-tested; add new `@pytest.mark.slow` tests in integration or in those modules when implementing full scenarios.


## CI/CD Pipeline

### Fast PR Checks (`ci-pr-checks.yml`)

**Trigger**: Pull requests to `main` or `dev`; pushes to `main`, `dev`, `tl/test_ecosystem_v5`, `tl/test_ecosystem_v5_fast`; `workflow_dispatch`  
**Duration**: ~5–10 minutes

**Strategy**:

1. Use pre-built Docker image (`latest` on main, branch-tagged on feature branches)
2. Overlay PR code onto `/opt/proteus/`; validate test structure
3. Run `pytest -m "unit and not skip"` with coverage (fast gate)
4. Run diff-cover (80% on changed lines), then `pytest -m "smoke and not skip"`
5. Lint runs in parallel: `ruff check src/ tests/`, `ruff format --check src/ tests/`

**Coverage gates**:

- **Fast gate**: 32.03% (`[tool.proteus.coverage_fast] fail_under` in `pyproject.toml`; ratcheted on push to main / `tl/test_ecosystem_v5`)
- **Diff-cover**: 80% on changed lines (enforced; `--diff-file` to avoid remote fetch in container)
- **Full gate**: 69% (`[tool.coverage.report]`; enforced in nightly, ratcheted on main)

**Command** (unit job):
```bash
pytest -m "unit and not skip" --ignore=tests/examples \
  --cov=src --cov-fail-under=32.03 --cov-report=term-missing --cov-report=xml --cov-report=html
```

**Artifacts**: HTML coverage, unit pytest log; optional download of last nightly integration coverage for “estimated total” in run summary.

### Nightly Science Validation (`ci-nightly-science.yml`)

**Trigger**: Schedule 03:00 UTC daily; `workflow_dispatch`  
**Duration**: ~4–6 hours total

**Jobs**:

1. **Quick integration** — `test_integration_dummy.py`, `test_integration_multi_timestep.py` (~15 min)
2. **Science validation** — `pytest -m "slow or integration"` with full coverage (69% gate), ratcheting on main
3. **Integration only** — `pytest -m "integration and not slow"` for separate tracking

**Artifacts**: HTML coverage, simulation outputs, test logs.

### Nightly Science Validation v5 (`ci-nightly-science-v5.yml`)

**Trigger**: Push to `tl/test_ecosystem_v5`; `workflow_dispatch`  
**Purpose**: Branch-specific nightly; runs integration + unit + slow (std_config), writes `coverage-integration-only.json`, uploads artifact `v5-branch-nightly-coverage`. Fast PR checks can use this artifact to show “estimated total” (unit + integration) in the run summary.

## Test Discovery & Organization

### Directory Structure

Tests mirror `src/proteus/`; `tools/validate_test_structure.sh` checks this (special dirs `data`, `helpers`, `integration` are skipped in module checks).

```text
tests/
├── examples/          # Excluded from CI
├── integration/       # Multi-module coupling (@pytest.mark.integration, some @pytest.mark.smoke)
│   ├── test_integration_dummy.py
│   ├── test_integration_dummy_agni.py
│   ├── test_integration_aragog_agni.py
│   ├── test_integration_aragog_janus.py
│   ├── test_integration_std_config.py   # also @pytest.mark.slow
│   ├── test_integration_multi_timestep.py
│   ├── test_integration_calliope_multi_timestep.py
│   ├── test_albedo_lookup.py
│   └── test_smoke_*.py                  # smoke tests (some skipped)
├── config/            # @pytest.mark.unit
├── grid/               # @pytest.mark.integration
├── inference/          # @pytest.mark.integration
├── plot/               # @pytest.mark.unit
├── atmos_chem/         # @pytest.mark.unit
├── atmos_clim/         # @pytest.mark.unit
├── escape/             # @pytest.mark.unit (some tests @pytest.mark.skip)
├── interior/           # @pytest.mark.unit (one test @pytest.mark.skip)
├── observe/            # @pytest.mark.unit
├── orbit/              # @pytest.mark.unit
├── outgas/             # @pytest.mark.unit
├── star/               # @pytest.mark.unit
├── utils/              # @pytest.mark.unit (some tests @pytest.mark.skip)
├── data/               # data/download tests (some @pytest.mark.skip)
├── test_cli.py         # @pytest.mark.unit
└── test_init.py       # @pytest.mark.unit
```

### Marker Counts (as of 2026-01-27)

| Marker | Approx. count | Runs In |
| --- | --- | --- |
| `@pytest.mark.unit` | 480+ | PR: `pytest -m "unit and not skip"` |
| `@pytest.mark.smoke` | Multiple (some skipped) | PR: `pytest -m "smoke and not skip"` |
| `@pytest.mark.integration` | 25+ | Nightly (`ci-nightly-science*.yml`) |
| `@pytest.mark.slow` | 2+ (std_config) | Nightly |
| `@pytest.mark.skip` | Individual tests across files | Excluded from CI |

**Targets**: 5–7 active smoke tests in PRs; more integration/slow scenarios in nightly. Fast gate 32.03%, full gate 69% (see `pyproject.toml`).

## Test Fixtures and conftest.py

All PROTEUS tests share common configuration and parameter sets through `tests/conftest.py`. This file provides:

### Parameter Classes for Test Scenarios

Three representative exoplanet scenarios are defined as parameter classes (all values in SI units):

1. **EarthLikeParams** — Modern Earth reference
   - Habitable zone, outgassed interior, thin CO₂-N₂ atmosphere
   - Use for habitability baseline tests

2. **UltraHotSuperEarthParams** — TOI-561 b
   - Ultra-short period (0.45 day), extreme irradiation (~100× Earth)
   - Ultra-low density (4.3 g/cm³) suggests thick volatile envelope
   - Use for atmospheric escape physics and magma ocean tests
   - Reference: Teske et al. (arXiv:2509.17231)

3. **IntermediateSuperEarthParams** — L 98-59 d
   - 3.7-day orbit, H₂-rich atmosphere (MMW ~9 u), permanent magma ocean
   - Bridges habitability and volatile loss regimes
   - Use for volatile retention and tidal heating tests
   - Reference: Nicholls et al. (arXiv:2507.02656)

### Session-Scoped Fixtures

All fixtures use `scope='session'` (cached once per test run for efficiency):

```python
def test_with_fixtures(earth_params, ultra_hot_params, config_minimal):
    """Example: use multiple fixtures in one test."""
    # earth_params: EarthLikeParams instance
    # ultra_hot_params: UltraHotSuperEarthParams instance
    # config_minimal: Path to input/minimal.toml
    pass
```

**Available Fixtures**:

- `earth_params` → EarthLikeParams instance
- `ultra_hot_params` → UltraHotSuperEarthParams instance
- `intermediate_params` → IntermediateSuperEarthParams instance
- `config_earth` → Path to `input/planets/earth.toml`
- `config_minimal` → Path to `input/minimal.toml`
- `config_dummy` → Path to `input/demos/dummy.toml`
- `proteus_root` → Absolute path to repository root

### Physical Constants in Parameters

All parameter classes use PROTEUS constants from `src/proteus/utils/constants.py`:

- Gravitational constant `const_G`
- Stefan-Boltzmann constant `const_sigma`
- Solar mass `M_sun`, Solar radius `R_sun`, Solar luminosity `L_sun`
- Earth mass `M_earth`, Earth radius `R_earth`
- Seconds per year `secs_per_year`

This ensures consistency across the entire test suite and with simulations.

## Running Tests Locally

### Unit tests (matches PR unit job)

```bash
pytest -m "unit and not skip"
```

### Smoke tests (matches PR smoke job)

```bash
pytest -m "smoke and not skip"
```

### Unit + Smoke (what PR runs)

```bash
pytest -m "(unit or smoke) and not skip"
```

### Integration tests (slower, ~minutes–hours)

```bash
pytest -m integration
```

### Slow tests (nightly scenarios)

```bash
pytest -m slow
```

### Everything except slow

```bash
pytest -m "not slow"
```

### With coverage

```bash
pytest --cov=src --cov-report=html
# or for unit-only coverage check vs fast gate:
pytest -m "unit and not skip" --cov=src --cov-fail-under=32.03 --cov-report=term-missing
```

## Implementation Checklist for New Tests

When implementing tests for a module, follow this checklist:

1. Identify the test type and marker:

- Unit test (Python logic, mocks physics) → `@pytest.mark.unit`
- Smoke test (binary init, 1 timestep) → `@pytest.mark.smoke`
- Integration test (multi-module workflow) → `@pytest.mark.integration`
- Slow test (comprehensive, hours-long) → `@pytest.mark.slow`

1. Create the test file (if not exists):

- Location: `tests/<module>/test_<filename>.py`
- Mirror `src/proteus/<module>/<filename>.py`; run `bash tools/validate_test_structure.sh` to verify

1. Add a test function with the correct marker:

```python
import pytest

@pytest.mark.unit
def test_my_feature():
   """Brief description of what this tests."""
   # Test code here
   assert result == expected
```

1. Replace @pytest.mark.skip placeholders with real tests:

```python
# Before:
@pytest.mark.skip(reason="Placeholder test")
def test_placeholder():
   pass

# After:
@pytest.mark.unit
def test_my_feature():
   # Real test implementation
```

1. Verify coverage improvements:

- Run `pytest --cov=src`
- Confirm coverage increased for your module

1. Run the appropriate test marker group:

- Unit: `pytest -m "unit and not skip"`
- Integration: `pytest -m integration`
- Ensure tests pass and, for unit, coverage meets fast gate (32.03%)

## Coverage Requirements

- **Fast gate**: 32.03% (`[tool.proteus.coverage_fast] fail_under` in `pyproject.toml`). Enforced in `ci-pr-checks.yml` for `pytest -m "unit and not skip"`. Ratcheted on push to main or `tl/test_ecosystem_v5`.
- **Full gate**: 69% (`[tool.coverage.report] fail_under`). Enforced in nightly runs; ratcheted on main.
- **Diff-cover**: 80% on changed lines in PRs (uses `--diff-file` in container).
- **Calculation**: Examples and skipped tests are excluded from PR runs (`unit and not skip`, `smoke and not skip`).
- **Tools**: `pytest --cov` or `coverage run -m pytest`; report at `htmlcov/index.html`.

Coverage is enforced in:

- `ci-pr-checks.yml` — unit tests (fast gate) and diff-cover
- `ci-nightly-science.yml` and `ci-nightly-science-v5.yml` — integration/slow (full gate on main)

**Last updated**: 2026-01-27

## References

- [Test Infrastructure Documentation](./test_infrastructure.md)
- [Docker CI Architecture](./docker_ci_architecture.md)
- [PROTEUS Copilot Instructions](../.github/copilot-instructions.md)
