# Validation: `src/proteus/interior_energetics/common.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
shared interior-energetics helpers against a published source, an analytical
limit, or a cross-implementation comparison. The marker is registered in
`pyproject.toml`.

These helpers are the implementation of the linear crystallisation front that
the `boundary` backend uses, so the pins here cover that backend. Radiogenic
heating is shared through `interior_energetics/radiogenic.py`.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_boundary.py::test_solidification_front_matches_values_frozen_before_the_shared_helper` | Cross-implementation: the boundary backend's previous inline front expressions | Confirms that the solidification radius and its temperature derivative survive delegation exactly. | 2026-11-10 |
