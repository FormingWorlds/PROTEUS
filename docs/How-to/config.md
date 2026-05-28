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

Many parameters have default values, so you do not have to provide them in
the file. Some parameters are conditionally required. For example, if you use
the `mors` stellar evolution module (`star.module = 'mors'`), then you must
also set `star.mors.age_now`. If you use the `dummy` module instead, that
parameter is not required.

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
