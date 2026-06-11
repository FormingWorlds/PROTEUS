# `proteus.interior_energetics.wrapper`

The orchestration wrapper for the interior energetics submodule (Aragog or SPIDER).
The most relevant function for Zalmoxis coupling is `equilibrate_initial_state()`, which iterates CALLIOPE + Zalmoxis to a converged state before the main coupling loop starts.

For the conceptual overview, see the [Zalmoxis coupling explainer](https://proteus-framework.org/Zalmoxis/Explanations/proteus_coupling.html#pre-main-loop-equilibration).

::: proteus.interior_energetics.wrapper.equilibrate_initial_state
    options:
      show_source: true
      show_root_heading: true

::: proteus.interior_energetics.wrapper.solve_structure
    options:
      show_source: true
      show_root_heading: true
