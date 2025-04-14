# Configuration file

PROTEUS uses [TOML](https://toml.io/en/) to structure its configuration files.

All of the parameters required to run the model are
listed below with short explanations of their purpose and the values
they accept. Configuration files can contain blank lines. Comments are
indicated with a `#` symbol. Whitespace indentation is purely stylistic.

Many of the parameters have default values, meaning that you do not have to provide them in
the file. Some parameters are conditionally required. For example, if you use the `mors`
stellar evolution module (i.e. `star.module == 'mors'`), then you are required to also set
the variable `star.mors.age_now`. However, if you instead decided to use the `dummy`
stellar evolution module then the `age_now` parameter is not required.

See the `default.toml` configuration for a comprehensive example of all possible parameters.

### Examples

Have a look at the [input configs](https://github.com/FormingWorlds/PROTEUS/tree/main/input)
for ideas of how to set up your config in practice.

## Developers: adding a new parameter

So, you are developing a new model and want to add some parameters?
Follow these steps:

1. Decide on a good parameter name (*e.g.* `my_star_var`), and under which section to place it (*e.g.* `star`).
   Add the new variable to the [config submodule](https://github.com/FormingWorlds/PROTEUS/tree/main/src/proteus/config/_star.py).
2. Add the type for your variable, *e.g.* [float][], [int][], [str][].
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

::: proteus.config._delivery
    options:
      heading_level: 3
      show_root_heading: False
      show_root_toc_entry: False
      members_order: source
