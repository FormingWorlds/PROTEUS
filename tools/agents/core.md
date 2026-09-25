## PROTEUS ecosystem rules

This repository is one module of PROTEUS, a coupled model of the evolution of rocky planets and their atmospheres. PROTEUS selects one module per process in its config:

- interior energetics: aragog, SPIDER; interior structure: Zalmoxis, SPIDER
- atmosphere energy balance: AGNI, JANUS, both with SOCRATES spectral radiative transfer
- volatile outgassing: CALLIOPE, atmodeller; atmospheric chemistry: VULCAN
- stellar evolution: MORS; atmospheric escape: ZEPHYRUS, BOREAS; tides: Obliqua, lovepy
- fwl-io downloads the reference data the modules use

A change to anything PROTEUS calls or reads (a function signature, a config key, an output column, a unit) can break the coupled model while this module's own tests pass. Search `src/proteus/` in PROTEUS for the name before you change it, and name the affected PROTEUS call sites in the pull request.

### Physics and numerics

- Units differ between modules: MORS and VULCAN use cgs, CALLIOPE, atmodeller and AGNI take pressures in bar, and the PROTEUS config reference pages state the unit of each key (for example stellar mass in M_sun, initial partial surface pressures in bar, stellar age in Gyr). State the unit of every physical quantity in its docstring and convert explicitly at the boundary: a unit mismatch between two modules passes the tests of both and shows only in the coupled run.
- A physics test must fail for the most plausible wrong formula: check a conservation law, a bound, a monotonicity, or a published or analytical value, and assert that the wrong result falls outside the tolerance.
- Take each physical constant from one source per repository (in Python `scipy.constants` or the module's constants file). Two retyped values of one constant differ at round-off and hide real differences between code paths.
- Do not loosen a solver tolerance, a conservation check or a clamp to make a run or a test pass. Find the cause; a loosened check also hides the next defect.

### Branches and pull requests

- Work on a feature branch `<initials>/<short-description>`; `main` changes only through a reviewed pull request.
- Fill every section of the repository's pull-request template, where it has one.
- Before you push, run the checks this file and `tests/AGENTS.md` name.

### Where knowledge goes

Rules for every contributor go in the `AGENTS.md` files, the reason for a change in its commit message and pull-request description, and the scientific validation of a module in its `docs/Validation/` pages, where the repository has them. Do not add memory, notes or decision-log files to the repository: nobody maintains them, and they go stale.
