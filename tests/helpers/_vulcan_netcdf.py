"""Write small NetCDF files in the schema of VULCAN's `write_netcdf`.

The variables PROTEUS reads, with VULCAN's names, dimensions, CGS units and
fill value: `p` and `tmp` at the nz layer centres, `pl` and `zl` at the nz+1
interfaces, `Kzz` on the interfaces with the bottom and top ones filled,
`species` as strings and `x_gas` [nz, species]. Arrays go from the bottom up.
"""

from __future__ import annotations

import numpy as np

# The netCDF default fill value for doubles, which VULCAN writes
VULCAN_FILL = 9.96921e36


def vulcan_layers(P_b: float = 1.0e8, P_t: float = 1.0e2, nz: int = 6) -> dict:
    """A bottom-up column of nz layers [dyn cm-2, K, cm, cm2 s-1], log-spaced in pressure."""
    p = np.logspace(np.log10(P_b), np.log10(P_t), nz)
    pl = np.empty(nz + 1)
    pl[1:-1] = np.sqrt(p[1:] * p[:-1])
    pl[0] = p[0] ** 1.5 / p[1] ** 0.5
    pl[-1] = p[-1] ** 1.5 / p[-2] ** 0.5
    return {
        'p': p,
        'pl': pl,
        'tmp': np.linspace(1500.0, 400.0, nz),
        'zl': np.linspace(0.0, 5.0e7, nz + 1),
        'Kzz': np.linspace(1.0e5, 1.0e8, nz - 1),
    }


def write_vulcan_netcdf(fpath: str, layers: dict, species: list, x_gas: np.ndarray) -> None:
    """Write `layers` (from `vulcan_layers`) and mixing ratios as a VULCAN NetCDF file."""
    import netCDF4 as nc

    nz = len(layers['p'])
    with nc.Dataset(fpath, 'w', format='NETCDF4') as ds:
        ds.createDimension('layer', nz)
        ds.createDimension('level', nz + 1)
        ds.createDimension('species', len(species))

        def put(name, dims, data):
            var = ds.createVariable(name, 'f8', dims, fill_value=VULCAN_FILL)
            var[:] = data

        put('p', ('layer',), layers['p'])
        put('pl', ('level',), layers['pl'])
        put('tmp', ('layer',), layers['tmp'])
        put('zl', ('level',), layers['zl'])
        kzz = np.full(nz + 1, VULCAN_FILL)
        kzz[1:-1] = layers['Kzz']
        put('Kzz', ('level',), kzz)
        put('x_gas', ('layer', 'species'), x_gas)
        var = ds.createVariable('species', str, ('species',))
        var[:] = np.array(species, dtype=object)
