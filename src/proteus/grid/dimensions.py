# Value construction for the dimensions (swept axes) of a PROTEUS parameter grid.
from __future__ import annotations

from numbers import Real

import numpy as np

# Keys each dimension method reads from its table, in the order of its Grid setter.
DIMENSION_METHODS = {
    'direct': ('values',),
    'linspace': ('start', 'stop', 'count'),
    'logspace': ('start', 'stop', 'count'),
    'arange': ('start', 'stop', 'step'),
}

# Fraction of an arange step within which two values count as the same grid point.
ARANGE_STEP_RTOL = 1e-6


def _check_count(count: int) -> None:
    """Raise ValueError unless ``count`` is a positive number of points."""
    if count < 1:
        raise ValueError(f'Grid dimension count must be at least 1, got {count}')


def dimension_linspace(start: float, stop: float, count: int) -> list[float]:
    """Return ``count`` values evenly spaced on a linear scale over [start, stop]."""
    _check_count(count)
    return [float(v) for v in np.linspace(start, stop, count)]


def dimension_logspace(start: float, stop: float, count: int) -> list[float]:
    """Return ``count`` values evenly spaced on a log scale over [start, stop].

    Raises
    ------
    ValueError
        If ``count`` < 1, or ``start`` or ``stop`` is not positive.
    """
    _check_count(count)
    if start <= 0 or stop <= 0:
        raise ValueError(
            f'Grid logspace bounds must be positive, got start={start}, stop={stop}'
        )
    return [float(v) for v in np.logspace(np.log10(start), np.log10(stop), count)]


def dimension_arange(start: float, stop: float, step: float) -> list[float]:
    """Return values from start to stop in steps of ``step``, including the endpoint.

    The endpoint is appended when ``np.arange`` omits it. ``start == stop`` gives
    that single value. Values count as equal within ``ARANGE_STEP_RTOL * |step|``,
    so the axis does not depend on the magnitude of its values.

    Raises
    ------
    ValueError
        If ``step`` is zero, or its sign cannot reach ``stop`` from ``start``.
    """
    if start == stop:
        return [float(stop)]
    if step == 0 or (stop - start) * step < 0:
        raise ValueError(f'Grid arange step {step} cannot reach stop={stop} from start={start}')
    tol = ARANGE_STEP_RTOL * abs(step)
    values = [float(v) for v in np.arange(start, stop, step)]
    # np.arange can end within rounding of stop (0.1 to 0.4 by 0.1); keep one endpoint.
    if abs(values[-1] - stop) <= tol:
        values[-1] = float(stop)
    else:
        values.append(float(stop))
    return values


def direct_values_error(values) -> str | None:
    """Return why a direct grid axis cannot be built from ``values``, or None if it can.

    The axis needs a non-empty list, tuple or 1-D array of numbers and booleans, or of
    strings: duplicates are removed and numbers sorted, which has no defined result for a
    NaN (it equals nothing, itself included) or for strings beside other values.
    """
    if not (
        isinstance(values, (list, tuple))
        or (isinstance(values, np.ndarray) and values.ndim == 1)
    ):
        return f'is not a list of values: {values!r}'
    values = list(values)
    if not values:
        return 'has no values'
    for v in values:
        if not isinstance(v, (Real, np.bool_, str)):
            return f'holds {v!r}, which is not a real number, boolean or string'
    if any(v != v for v in values):
        return 'holds NaN, which cannot be deduplicated or ordered'
    first = isinstance(values[0], str)
    other = next((v for v in values if isinstance(v, str) != first), None)
    if other is not None:
        return f'mixes strings with other values ({values[0]!r} and {other!r})'
    return None


def dimension_direct(values: list, sort: bool = True) -> list:
    """Return the given values with duplicates removed (sorted when numeric).

    String values (or any values with ``sort=False``) keep their written order,
    so the case numbering of a string axis is the same in every process (a ``set``
    of strings is ordered by the per-process string hash). NumPy scalars become
    Python values.

    Raises
    ------
    ValueError
        If ``values`` cannot make an axis (see ``direct_values_error``).
    """
    why = direct_values_error(values)
    if why:
        raise ValueError(f'Grid direct dimension {why}')
    values = [v.item() if isinstance(v, np.generic) else v for v in values]  # array, tuple
    the_list = list(dict.fromkeys(values))  # written order, duplicates removed
    if (not isinstance(values[0], str)) and sort:  # sort if numeric
        the_list = sorted(the_list)
    return the_list


_DIMENSION_BUILDERS = {
    'direct': dimension_direct,
    'linspace': dimension_linspace,
    'logspace': dimension_logspace,
    'arange': dimension_arange,
}


def dimension_values(table: dict) -> list:
    """Return the swept values for one grid dimension table.

    Shared by the ``Grid`` setters (proteus.grid.manage) and by the grid analysis
    (proteus.grid.analyse), so both read identical values for every axis.

    Parameters
    ----------
    table : dict
        A dimension table from a grid config: a ``method`` plus the keys that
        method reads (see ``DIMENSION_METHODS``).

    Returns
    -------
    list
        The values that axis takes across the grid.

    Raises
    ------
    ValueError
        If ``method`` is missing or not one of ``DIMENSION_METHODS``, or the
        method's arguments describe an empty axis.
    """
    method = table.get('method')
    if method not in _DIMENSION_BUILDERS:
        raise ValueError(
            f'Unknown grid dimension method {method!r}; '
            f'expected one of {", ".join(sorted(_DIMENSION_BUILDERS))}'
        )
    args = [table[arg] for arg in DIMENSION_METHODS[method]]
    return _DIMENSION_BUILDERS[method](*args)
