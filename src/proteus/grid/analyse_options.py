# Options of the [post_processing] table of a grid file, checked by `proteus grid` at
# launch and by `proteus grid-analyse` before it reads any case.
from __future__ import annotations

# Outputs drawn in the ECDF plot when [post_processing] sets no output_variables.
DEFAULT_OUTPUT_VARIABLES = (
    'solidification_time',
    'Phi_global',
    'T_surf',
    'P_surf',
    'atm_kg_per_mol',
)

# Keys accepted in the [post_processing] table, with their expected types.
POST_PROCESSING_OPTIONS = {
    'update_csv': bool,
    'plot_status': bool,
    'plot_ecdf': bool,
    'colormap': str,
    'colourmap': str,
    'output_variables': list,
    'plot_format': str,
    'style': str,
}

# Image formats the grid plots may be saved as.
SUPPORTED_PLOT_FORMATS = ('png', 'pdf')

# Variants of the proteus-mpl style that [post_processing].style may load.
PLOT_STYLES = ('light', 'dark')


def _check_values(table: dict) -> None:
    """Raise ValueError for a [post_processing] value outside its accepted set."""
    outputs = table.get('output_variables', [])
    if not all(isinstance(v, str) for v in outputs):
        raise ValueError('[post_processing].output_variables must be a list of strings')
    if {'colormap', 'colourmap'} <= set(table) and table['colormap'] != table['colourmap']:
        raise ValueError('[post_processing] sets both colormap and colourmap differently')
    fmt = table.get('plot_format')
    if fmt is not None and fmt.lower() not in SUPPORTED_PLOT_FORMATS:
        raise ValueError(
            f"Unsupported [post_processing].plot_format '{fmt}'; choose one of "
            f'{SUPPORTED_PLOT_FORMATS}'
        )
    style = table.get('style')
    if style is not None and style not in PLOT_STYLES:
        raise ValueError(
            f"Unknown [post_processing].style '{style}'; choose one of {PLOT_STYLES}"
        )
    name = table.get('colormap', table.get('colourmap'))
    if name is not None and not colourmap_exists(name):
        raise ValueError(
            f"Unknown [post_processing] colour map '{name}': not a matplotlib or "
            "cmcrameri colour map (for example 'batlowK_r' or 'viridis')"
        )


def colourmap_exists(name: str) -> bool:
    """Return whether ``name`` is a matplotlib or cmcrameri colour map name."""
    import cmcrameri.cm  # noqa: F401  (registers the 'cmc.<name>' colour maps)
    import matplotlib as mpl

    return name in mpl.colormaps or f'cmc.{name}' in mpl.colormaps


def check_post_processing_table(table) -> None:
    """Raise ValueError for a [post_processing] table with unknown keys or bad values.

    Parameters
    ----------
    table : object
        The ``post_processing`` entry of a parsed grid file.
    """
    if not isinstance(table, dict):
        raise ValueError(f'[post_processing] must be a table, got {type(table).__name__}')
    unknown = sorted(set(table) - set(POST_PROCESSING_OPTIONS))
    if unknown:
        raise ValueError(
            f'Unknown key(s) {unknown} in [post_processing]; accepted keys are '
            f'{sorted(POST_PROCESSING_OPTIONS)}'
        )
    for key, value in table.items():
        expected = POST_PROCESSING_OPTIONS[key]
        if not isinstance(value, expected):
            raise ValueError(
                f'[post_processing].{key} must be a {expected.__name__}, '
                f'got {value!r} ({type(value).__name__})'
            )
    _check_values(table)


def read_post_processing_options(config: dict) -> dict:
    """Return the validated ``[post_processing]`` options with defaults filled in."""
    table = config.get('post_processing', {})
    check_post_processing_table(table)
    opts = dict(table)
    opts.setdefault('update_csv', True)
    opts.setdefault('plot_status', True)
    opts.setdefault('plot_ecdf', True)
    # Accept both spellings; 'colormap' matches matplotlib, 'colourmap' the UK style.
    opts['colormap'] = opts.get('colormap', opts.pop('colourmap', None))
    opts.setdefault('output_variables', list(DEFAULT_OUTPUT_VARIABLES))
    opts.setdefault('plot_format', None)
    opts.setdefault('style', None)
    return opts
