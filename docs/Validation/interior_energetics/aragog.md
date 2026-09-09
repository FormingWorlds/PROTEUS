# Validation: `src/proteus/interior_energetics/aragog.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.interior_energetics.aragog` against a published source,
an analytical limit, or a cross-implementation cross-check. The marker is
registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_energetics/test_aragog.py::test_numpy_entropy_state_negative_pin_matches_jax_thermal_and_chemical` | Cross-implementation cross-check: aragog's EOS-backed numpy `EntropyState.update` against aragog's `jax.phase.compute_mlt` | n/a (dependency-internal comparison) | Constructs `EntropyState` and `PhaseParams` directly with matched thermal/chemical pin values (bypassing the `interior_energetics` config) to check that the two code paths `aragog.py` can dispatch to agree on the negative-pin convention: both clamp a negative multiplier to a spatially uniform `abs(value)` profile on the thermal and chemical eddy-diffusivity channels, even though the numpy side runs the real EOS-backed physics and the jax side runs a synthetic mesh/phase fixture. Config-to-solver threading of both fields is covered separately by the mock-based `test_setup_solver_threads_eddy_diffusivity_thermal`/`_chemical` tests. |
