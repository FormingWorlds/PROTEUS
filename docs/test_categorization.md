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

**Which marker should I use?**
- **Most tests → `unit`**: Testing a single function? Mock external dependencies, use `unit`.
- **Testing real binaries → `smoke`**: Need SOCRATES/AGNI/SPIDER actually running? Use `smoke`.
- **Testing module coupling → `integration`**: ARAGOG + AGNI working together? Use `integration`.
- **Full science runs → `slow`**: Multi-hour simulations? Use `slow`.

---

## CI/CD Pipeline

### What Happens When You Open a PR

1. **Structure check**: Validates `tests/` mirrors `src/proteus/`
2. **Unit tests**: Runs `pytest -m "unit and not skip"` with coverage
3. **Diff-cover**: Checks 80% coverage on your changed lines
4. **Smoke tests**: Runs `pytest -m "smoke and not skip"`
5. **Lint**: Checks code style with ruff

**Runtime**: ~5-10 minutes

### What Happens Nightly

The nightly workflow (`ci-nightly.yml`) runs at 3am UTC and:
- Runs ALL tests (unit → smoke → integration → slow)
- Updates coverage thresholds (ratcheting)
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

**Current thresholds** (from `pyproject.toml`):
- **Fast gate**: 44.45% — checked on every PR (unit + smoke)
- **Full gate**: 59% — checked nightly (all tests)
- **Diff-cover**: 80% — required on changed lines

These thresholds auto-increase ("ratchet") and never decrease. Check coverage locally with `pytest --cov=src --cov-report=html`.

---

## References

- [Test Infrastructure](test_infrastructure.md) — Coverage workflows, reusable quality gate, troubleshooting
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Docker CI Architecture](docker_ci_architecture.md) — Docker image, CI pipelines
- [AGENTS.md](../AGENTS.md) — Commands and thresholds
