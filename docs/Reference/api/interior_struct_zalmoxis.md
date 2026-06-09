# `proteus.interior_struct.zalmoxis`

The PROTEUS-side wrapper around the [Zalmoxis](https://proteus-framework.org/Zalmoxis) interior structure module.
This module builds the call-time configuration dict, calls `zalmoxis.solver.main()`, writes the Aragog mesh file, and validates the schema contract on `zalmoxis_output.dat`.

For the conceptual overview of how Zalmoxis is integrated into PROTEUS, see the [Zalmoxis coupling explainer](https://proteus-framework.org/Zalmoxis/Explanations/proteus_coupling.html).
For the practical TOML recipe, see the [Zalmoxis coupling how-to](https://proteus-framework.org/Zalmoxis/How-to/proteus_coupling.html).

::: proteus.interior_struct.zalmoxis
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source
      filters: ["!^_"]
