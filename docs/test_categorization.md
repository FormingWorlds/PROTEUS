# Test Categorization and CI/CD

This document defines how PROTEUS tests are categorized and how they run in CI. For infrastructure (layout, coverage workflow, troubleshooting) see [Test Infrastructure](test_infrastructure.md).

---

## Test Categories (Markers)

All tests use pytest markers. Current thresholds are in `pyproject.toml` (`[tool.proteus.coverage_fast]` and `[tool.coverage.report]`).

| Marker | Purpose | Runtime | Runs in |
|--------|---------|---------|---------|
| `@pytest.mark.unit` | Python logic, mocked physics | &lt;100 ms/test | PR: `pytest -m "unit and not skip"` |
| `@pytest.mark.smoke` | Real binaries, 1 timestep, low res | &lt;30 s/test | PR: `pytest -m "smoke and not skip"`; some run only in nightly when `PROTEUS_CI_NIGHTLY=1` |
| `@pytest.mark.integration` | Multi-module coupling | Minutes | Nightly |
| `@pytest.mark.slow` | Full physics / long runs | Hours | Nightly |
| `@pytest.mark.skip` | Placeholder or env-limited | — | Excluded from CI |

**Unit** (480+ tests): config, utils, wrappers (atmos_clim, atmos_chem, escape, interior, observe, orbit, outgas, star), plot, CLI, init.  
**Smoke**: `tests/integration/test_smoke_minimal.py`, `test_smoke_atmos_interior.py`, `test_smoke_modules.py`, `test_smoke_janus.py`, `test_smoke_outgassing.py`. Some smoke tests are skipped in PR and run in nightly (JANUS/AGNI/CALLIOPE when binaries available).  
**Integration**: `tests/integration/test_integration_*.py`, `test_aragog_agni`, `test_aragog_janus`, `test_std_config`, `test_albedo_lookup`, plus grid/inference.  
**Slow**: e.g. `test_integration_std_config.py` extended run; future scenarios (Earth magma ocean, Venus greenhouse) as needed.

---

## CI/CD Pipeline

### Fast PR Checks (`ci-pr-checks.yml`)

- **Trigger**: Pull requests and pushes to `main`, `dev`, `tl/test_ecosystem_v5`, `tl/test_ecosystem_v5_fast`; `workflow_dispatch`.
- **Steps**: Validate test structure → `pytest -m "unit and not skip"` with coverage (fast gate) → diff-cover 80% on changed lines → `pytest -m "smoke and not skip"` → lint in parallel.
- **Coverage**: Fast gate from `[tool.proteus.coverage_fast] fail_under`; ratcheted on push to main / `tl/test_ecosystem_v5`. Diff-cover 80% on changed lines.

### Nightly Science Validation (`ci-nightly-science-v5.yml`)

- **Trigger**: Push to `tl/test_ecosystem_v5`; `workflow_dispatch`. Sets `PROTEUS_CI_NIGHTLY=1`.
- **Steps**: Unit → smoke (including those gated by `PROTEUS_CI_NIGHTLY`) → integration (dummy + `integration and not slow`) → slow (e.g. std_config). Combined coverage; full gate from `[tool.coverage.report] fail_under`.

### Nightly (`ci-nightly-science.yml`)

- **Trigger**: Schedule (e.g. 03:00 UTC); `workflow_dispatch`. Uses main + latest image.
- **Jobs**: Quick integration; science validation (`slow or integration`); integration-only. Full coverage ratcheting on main.

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

- **Fast gate**: `[tool.proteus.coverage_fast] fail_under` in `pyproject.toml`. Enforced in PR for unit run. Ratcheted on main / `tl/test_ecosystem_v5`.
- **Full gate**: `[tool.coverage.report] fail_under`. Enforced in nightly. Ratcheted on main.
- **Diff-cover**: 80% on changed lines in PRs.
- **Tools**: `pytest --cov=src` or `coverage run -m pytest`; reports in `htmlcov/` or `coverage report`.

---

## References

- [Test Infrastructure](test_infrastructure.md)
- [Test Building](test_building.md) — Prompts for unit/integration tests
- [Test Building Strategy](test_building_strategy.md) — Status and principles
- [AGENTS.md](../AGENTS.md) — Commands and thresholds
