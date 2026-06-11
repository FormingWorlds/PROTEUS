# Configuration file

PROTEUS uses [TOML](https://toml.io/en/) to structure its configuration files.
This page lists all parameters with their types, defaults, and descriptions.
For topic-specific parameter guides, see the **configuration reference** pages:

- [Execution and output](../Reference/config/params.md)
- [Planet and volatiles](../Reference/config/planet.md)
- [Star and orbit](../Reference/config/star_orbit.md)
- [Interior structure and energetics](../Reference/config/interior.md)
- [Atmosphere and chemistry](../Reference/config/atmosphere.md)
- [Escape and outgassing](../Reference/config/escape_outgas.md)
- [Synthetic observations](../Reference/config/observe.md)

For worked examples, see the [Tutorials](../Tutorials/quick_start_dummy.md).

## Defaults and required parameters

**Every parameter has a built-in default.** The defaults are defined in the
configuration schema in
[`src/proteus/config/`](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config),
so a configuration file only needs to set the parameters whose defaults you
want to change. Empty sections can be omitted entirely. The
[`minimal.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/input/minimal.toml)
example shows how little a working file needs: a few science-critical choices
(planet mass, orbit, volatiles, redox), with everything else left at its
default.

**Where to find the default for each parameter.** The configuration reference
pages listed above give the default value of every parameter in their
`Default` column, alongside its type and description. The auto-generated
listing further down this page is built from the same schema and shows each
parameter's source definition, including the coded default value.

**To see exactly which defaults applied to a run**, open the
`init_coupler.toml` file in that run's output folder. It is a completed copy of
the configuration with every parameter resolved, including all the defaults
that were filled in for the options you did not set. This is the
fully-expanded configuration the simulation actually used.

**Some parameters are conditionally required.** A parameter that is only
meaningful for one module is required when that module is selected. For
example, the `mors` stellar evolution module (`star.module = 'mors'`) requires
`star.mors.age_now`, whereas the `dummy` module does not. The configuration
loader reports an error at startup if a required parameter is missing for the
chosen modules.

See [`all_options.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/input/all_options.toml) for a comprehensive example. Have a look at the other [input configs](https://github.com/FormingWorlds/PROTEUS/tree/main/input) for ideas of how to set up your config in practice.

## Root parameters

::: proteus.config._config
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## General parameters

::: proteus.config._params
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Stellar evolution

::: proteus.config._star
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Orbital evolution and tides

::: proteus.config._orbit
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Interior structure

::: proteus.config._struct
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Magma ocean and planetary interior

::: proteus.config._interior
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Atmosphere climate

::: proteus.config._atmos_clim
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Atmospheric escape

::: proteus.config._escape
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Atmospheric chemistry

::: proteus.config._atmos_chem
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Volatile outgassing

::: proteus.config._outgas
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Elemental delivery and accretion

::: proteus.config._accretion
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

## Synthetic observations

::: proteus.config._observe
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source

---

??? info "For developers: adding a new parameter"

    So, you are developing a new model and want to add some parameters?
    Follow these steps:

    1. Decide on a good parameter name (*e.g.* `my_star_var`), and under which section to place it (*e.g.* `star`).
       Add the new variable to the [config submodule](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config/_star.py).
    2. Add the type for your variable, *e.g.* [`float`](https://docs.python.org/3/library/functions.html#float), [`int`](https://docs.python.org/3/library/functions.html#int), [`str`](https://docs.python.org/3/library/stdtypes.html#str).
       You can also add complex types, please check the [code](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config) for inspiration.
    3. Add a [validator](https://www.attrs.org/en/stable/api.html#module-attrs.validators)!
       If your variable has a maximum value (*e.g.* 10), you can add a validator to make sure
       that any values above 10 are rejected: `my_star_var: float = field(validator=attrs.validators.le(10))`
    4. Add a description for your new variable under `Attributes` in the docstring.
       The documentation uses the description to generate this documentation.
    5. Update the example [input configs](https://github.com/FormingWorlds/PROTEUS/tree/main/input).
       Proteus checks tests all input configs in this directory are valid.
    6. Use your parameter in your code, *i.e.*: `config.star.my_star_var`

    ```python title="src/proteus/config/_star.py"
    class Star:
        """Stellar parameters.

        Attributes
        ----------
        my_star_var: float
            Star variable, must be 10 or lower!
        """
        my_star_var: float = field(validator=attrs.validators.le(10))
    ```

    Proteus uses [attrs](https://www.attrs.org) for its
    parameter handling. Please see the [examples](https://www.attrs.org/en/stable/examples.html)
    for more information how to work with attrs.
