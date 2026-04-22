"""
Unified redox module for PROTEUS (#57).

This subpackage implements the Evans/Hirschmann redox-budget formalism
that replaces the pre-#57 convention of treating fO2 as a user-set
scalar pinned to the Iron-Wustite buffer. The module provides:

- Reference-state registry and redox-budget computations
  (:mod:`proteus.redox.budget`).
- Oxybarometers and reference buffers (IW / QFM / NNO),
  Schaefer+2024 Eq 13, Hirschmann+2022, Sossi+2020,
  Stagno+2013 peridotite fallback (:mod:`proteus.redox.buffers`).
- Partitioning hooks: Mariana's depth-resolved Fe³⁺/Fe²⁺ evolution
  stub (#653), metal–silicate K_D stubs (#526), EOS correction
  stub (#432) (:mod:`proteus.redox.partitioning`).
- Transactional Brent solver over ΔIW
  (:mod:`proteus.redox.solver`, Commit C).
- Species-resolved escape debit and conservation assertion
  (:mod:`proteus.redox.conservation`).
- Main-loop orchestration helper
  (:mod:`proteus.redox.coupling`).

Status registry for the three deferred physics hooks; `proteus doctor`
and the t=0 log line surface these to stdout so every run records
which hooks were active.

Plan: claude-config/plans/let-s-brainstorm-about-this-calm-canyon.md
v6.
"""
from __future__ import annotations

IMPLEMENTATION_STATUS = {
    'fe3_evolution': 'stub-bulk',       # #653 flips to 'mariana-schaefer2024'
    'mineral_melt_KD': 'stub-bulk',     # paired with fe3_evolution (#653)
    'liquidus_phase': 'stub-bg',        # single-mineral fallback (#653)
    'metal_silicate': 'zeros',          # #526 flips to e.g. 'young2023'
    'eos_correction': 'identity',       # #432 flips to 'dorfman24+hirose19'
    'schaefer_eq13': 'active',          # Commit B
    'stagno2013_peridotite': 'active',  # Commit B
    'h2o_escape_rb': 'stub',            # Commit C
    'bridgmanite': 'off',               # out of scope
}


def print_status_line() -> str:
    """Emit the one-line status summary consumed by `proteus doctor`."""
    return 'redox hooks: ' + ', '.join(
        f'{k}={v}' for k, v in sorted(IMPLEMENTATION_STATUS.items())
    )
