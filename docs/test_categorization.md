# Test Categorization and CI/CD

## What This Document Is For

**New to PROTEUS testing?** This document explains how we organize tests into categories and how CI (Continuous Integration) automatically runs them when you submit code.

**Key concept:** Tests are labeled with *markers* that tell pytest what kind of test they are. Different markers run at different times—fast tests run on every pull request, slow tests run overnight.

For writing tests, see [Test Building](test_building.md). For coverage analysis and troubleshooting, see [Test Infrastructure](test_infrastructure.md).

---

## Test Categories (Markers)

Add one of these markers above each test function:

| Marker | What It Tests | Speed | When It Runs |
|--------|---------------|-------|--------------|
| `@pytest.mark.unit` | Python logic with mocked physics | <100 ms | Every PR |
| `@pytest.mark.smoke` | Real binaries, 1 timestep | <30 s | Every PR |
| `@pytest.mark.integration` | Multiple modules working together | Minutes | Nightly only |
| `@pytest.mark.slow` | Full physics simulations | Hours | Nightly only |
| `@pytest.mark.skip` | Temporarily disabled | — | Never |

### Which marker should I use?

- **Most tests → `unit`**: Testing a single function? Mock external dependencies, use `unit`.
- **Testing real binaries → `smoke`**: Need SOCRATES/AGNI/SPIDER actually running? Use `smoke`. Module-level smoke tests (e.g. in `tests/atmos_clim/`) validate a single binary with 1 timestep. Integration-level smoke tests (in `tests/integration/`) validate the coupling framework end-to-end with dummy modules.
- **Testing module coupling → `integration`**: ARAGOG + AGNI working together? Use `integration`.
- **Full science runs → `slow`**: Multi-hour simulations? Use `slow`.

---

## CI/CD Pipeline

### What Happens When You Open a PR

1. **Structure check**: Validates `tests/` mirrors `src/proteus/`
2. **Unit tests (Linux)**: Runs `pytest -m "unit and not skip"` with coverage
3. **Diff-cover**: Checks 80% coverage on your changed lines
4. **Smoke tests (Linux)**: Runs `pytest -m "smoke and not skip"`
5. **Unit tests (macOS)**: Runs unit tests on macOS (no compiled binaries)
6. **Lint**: Checks code style with ruff
7. **Summary**: Aggregates results from all platforms into a unified report

**Runtime**: ~5-10 minutes

### What Happens Nightly

The nightly workflow (`ci-nightly.yml`) is primarily triggered by `docker-build.yml` after the 2am UTC image rebuild. A 3am UTC cron acts as a fallback if the docker build didn't run. A deduplication check prevents running twice.

- Runs ALL tests (unit → smoke → integration → slow)
- Updates coverage thresholds (ratcheting)
- Uploads aggregate coverage (unit + smoke + integration) to Codecov
- Sets `PROTEUS_CI_NIGHTLY=1` to enable additional smoke tests

### Coverage Rules

| Rule | Value | What It Means |
|------|-------|---------------|
| Grace period | 0.3% | Small coverage drops allowed (warning posted) |
| Diff-cover | 80% | Your changed lines need 80% coverage |
| Staleness | 48h | PR fails if nightly data is too old |

---

## Test Layout

Tests mirror `src/proteus/`. Validation: `bash tools/validate_test_structure.sh`. Special dirs `data`, `helpers`, `integration` are handled in validation.

```text
tests/
├── integration/     # test_smoke_*.py, test_integration_*.py, test_aragog_*, test_std_config, etc.
├── config/, utils/, plot/, star/, orbit/, interior/, escape/, outgas/, observe/, atmos_clim/, atmos_chem/
├── grid/, inference/, data/
├── test_cli.py, test_init.py
└── conftest.py      # Shared fixtures (see Test Infrastructure)
```

---

## Fixtures (`tests/conftest.py`)

- **Parameter classes**: `EarthLikeParams`, `UltraHotSuperEarthParams`, `IntermediateSuperEarthParams` (session-scoped).
- **Config paths**: `config_earth`, `config_minimal`, `config_dummy`, `proteus_root`.
- **Fixtures**: `earth_params`, `ultra_hot_params`, `intermediate_params` (instances of the above).

Integration-specific fixtures (e.g. multi-timestep runs, conservation checks) are in `tests/integration/conftest.py`. See [Test Infrastructure](test_infrastructure.md) for details.

---

## Running Tests Locally

```bash
pytest -m "unit and not skip"           # Unit only (matches PR)
pytest -m "smoke and not skip"         # Smoke only
pytest -m "(unit or smoke) and not skip"  # What PR runs
pytest -m integration                   # Integration
pytest -m slow                         # Slow
pytest -m "not slow"                   # All except slow
pytest --cov=src --cov-report=html     # With coverage
```

For fast gate check: `pytest -m "unit and not skip" --cov=src --cov-fail-under=<value>` (value from `pyproject.toml`).

---

## Adding New Tests

1. Choose marker: `unit` / `smoke` / `integration` / `slow`.
2. Create `tests/<module>/test_<filename>.py` if needed (mirror source).
3. Use `@pytest.mark.<marker>` and docstrings; use `pytest.approx` for floats.
4. Run `bash tools/validate_test_structure.sh`; run the relevant marker group; ensure coverage meets the fast gate for unit changes.

---

## Coverage Requirements

### Coverage Gates

| Gate | Tests Included | When Checked | Threshold Source |
|------|---------------|--------------|------------------|
| Fast gate | unit + smoke | Every PR | `[tool.proteus.coverage_fast] fail_under` |
| Estimated total | union of PR (unit+smoke) + nightly | Every PR | `[tool.coverage.report] fail_under` |
| Full gate | unit + smoke + integration + slow | Nightly | `[tool.coverage.report] fail_under` |
| Diff-cover | changed lines only | Every PR | Hard-coded 80% |

### What Each Test Tier Contributes

- **`unit`** — Bulk of Python logic coverage (functions, branches, error paths). Fastest feedback loop.
- **`smoke`** — Covers binary wrapper code and real I/O paths that unit tests mock away.
- **`integration`** — Covers cross-module coupling paths (e.g., ARAGOG + JANUS handoff).
- **`slow`** — Full scientific validation. Contributes to coverage but primarily validates physics, not code paths.

All thresholds auto-increase ("ratchet") and never decrease. Check coverage locally with `pytest --cov=src --cov-report=html`.

For details on how coverage is collected across workflows and how the estimated total is computed, see [Coverage Collection & Reporting](test_infrastructure.md#coverage-collection-reporting).

---

## References

- [Test Infrastructure](test_infrastructure.md) — Coverage workflows, reusable quality gate, troubleshooting
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Docker CI Architecture](docker_ci_architecture.md) — Docker image, CI pipelines
- [.github/copilot-instructions.md](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/copilot-instructions.md) — Commands and thresholds
