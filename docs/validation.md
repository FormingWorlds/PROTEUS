# Validation Suite

## What This Is For

The `tests/validation/` directory contains scripts for running extended physics validation of PROTEUS beyond what automated CI covers.
Use it to validate feature branches, compare interior models (e.g. SPIDER with Zalmoxis-derived vs. Anderson-Williamson meshes), and verify that coupled evolution produces physically plausible results.

Unlike unit and smoke tests, validation runs execute real multi-module simulations (atmosphere + interior + outgassing + escape) over extended timescales.
They are too expensive for CI but essential for verifying that coupling changes do not introduce subtle physics regressions.

!!! note "Living toolbox"
    The scripts in this directory are working tools, not frozen references.
    Update them as the validation needs of the project evolve — new test blocks, additional diagnostics, different parameter sweeps.

---

## Directory Layout

```
tests/validation/
├── .gitignore              # Only .py and .toml files are tracked
├── runners/                # Launch simulations
│   ├── base_validation.toml
│   ├── run_validation_matrix.py
│   └── generate_habrok_grid.py
├── analysis/               # Post-process results and generate reports
│   ├── analyze_habrok_results.py
│   ├── generate_validation_report.py
│   ├── plot_validation_results.py
│   ├── plot_radial_profiles.py
│   ├── analyze_grid.py
│   └── plot_profiles.py
└── tools/                  # Standalone utilities (phase boundaries, etc.)
    ├── compare_melting_curves.py
    └── generate_monteux_phase_boundaries.py
```

All output artifacts (plots, HTML reports, result directories, CSVs) are gitignored.
Only `.py` and `.toml` source files are tracked.

---

## Subfolder Reference

### `runners/` — Launch Simulations

| Script | Purpose |
|--------|---------|
| `base_validation.toml` | Base TOML configuration shared by all validation cases. Sets common physics options, output settings, and convergence criteria. Individual cases override mass, CMF, and structure mode. |
| `run_validation_matrix.py` | Run the full validation matrix locally. Supports filtering by `--mass`, `--cmf`, `--struct`, `--phase2`, parallel execution with `--workers`, and `--resume` to skip completed cases. |
| `generate_habrok_grid.py` | Generate TOML configs and a SLURM dispatch script for running the validation matrix on the Habrok HPC cluster. |

### `analysis/` — Post-process and Report

| Script | Purpose |
|--------|---------|
| `analyze_habrok_results.py` | Parse Habrok output directories. Produces a summary CSV, completion statistics, AW-vs-Zalmoxis comparison tables, and a stability frontier map. |
| `generate_validation_report.py` | Generate a self-contained HTML validation report with base64-embedded plots and pass/fail checks. Also writes a Markdown version. |
| `plot_validation_results.py` | Comparison plots (thermal evolution, melt fraction, solidification time) across the test matrix, plus physical plausibility validation. |
| `plot_radial_profiles.py` | Radial cross-sections: $T(r)$ and $\phi(r)$ at multiple timesteps, overlaid on solidus/liquidus curves. |
| `analyze_grid.py` | Quick summary analysis of a completed Habrok grid (metrics extraction). |
| `plot_profiles.py` | Plot interior and atmospheric profiles from Habrok JSON snapshots. |

### `tools/` — Standalone Utilities

| Script | Purpose |
|--------|---------|
| `compare_melting_curves.py` | Convert SPIDER (entropy-based) and Zalmoxis (Monteux) melting curves to $T(P)$ space and plot side-by-side for comparison. |
| `generate_monteux_phase_boundaries.py` | Invert SPIDER's $T(P,S)$ lookup tables to produce $S(P)$ phase boundary files compatible with SPIDER's input format, using Monteux et al. (2016) $T(P)$ curves. |

---

## Typical Workflow

### Local validation of a feature branch

```bash
# Run a subset of the matrix (e.g. 1 Earth mass, default CMFs)
cd tests/validation
python runners/run_validation_matrix.py --mass 1.0 --workers 4

# Generate comparison plots
python analysis/plot_validation_results.py

# Check radial profiles for a specific case
python analysis/plot_radial_profiles.py --cases A_M1.0_CMF0.325_AW B_M1.0_CMF0.325_ZAL
```

### Full matrix on Habrok

```bash
# Generate configs and SLURM script
python runners/generate_habrok_grid.py --outdir /scratch/$USER/habrok_validation

# Submit
sbatch /scratch/$USER/habrok_validation/slurm_dispatch.sh

# After jobs complete, fetch results and analyze
python analysis/analyze_habrok_results.py --outdir /scratch/$USER/habrok_validation
python analysis/generate_validation_report.py --outdir /scratch/$USER/habrok_validation
```

### Phase boundary comparison

```bash
# Compare SPIDER vs Zalmoxis melting curves
python tools/compare_melting_curves.py

# Generate Monteux-derived phase boundaries for SPIDER
python tools/generate_monteux_phase_boundaries.py
```

---

## What Gets Validated

The test matrix covers combinations of:

- **Planet mass**: 1, 2, 3 $M_\oplus$
- **Core mass fraction (CMF)**: 0.1 – 0.5
- **Interior structure mode**: Anderson-Williamson (baseline) vs. Zalmoxis (self-consistent)
- **Phase 2 structural feedback**: with and without re-computation of structure during solidification

Key physical checks include:

- Monotonic cooling (surface temperature and melt fraction decrease over time)
- Crystallisation timescale within expected range for each mass
- Energy conservation (radiative flux consistent with internal energy change)
- AW and Zalmoxis results agree to within expected tolerances
- Phase 2 feedback does not destabilise the evolution

---

## Adding New Validation Cases

1. Define the new parameter combination or test block in `runners/run_validation_matrix.py` (local) or `runners/generate_habrok_grid.py` (cluster).
2. Add corresponding analysis or plotting logic in the `analysis/` scripts.
3. Update the validation checks in `analysis/plot_validation_results.py` or `analysis/generate_validation_report.py` as needed.
4. Commit the updated `.py` / `.toml` files. Output artifacts remain gitignored.
