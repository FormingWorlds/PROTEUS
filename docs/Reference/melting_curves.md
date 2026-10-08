# Melting curves

PROTEUS uses precomputed solidus and liquidus curves from laboratory experiments and theoretical parametrizations of silicate melting. These curves define the temperatures at which a silicate material begins to melt (solidus) and becomes fully molten (liquidus) as a function of pressure.

The melting-curve exporter generates lookup tables in both pressure-temperature (P-T) and pressure-entropy (P-S) space for several literature parametrizations of peridotite / silicate melting.

## What the exporter does

The script `tools/solidus_func.py` works with the EOS lookup tables:

- `temperature_solid.dat`
- `temperature_melt.dat`

These tables provide temperature as a function of entropy and pressure on structured grids. The exporter performs the following steps:

1. Build solidus and liquidus curves in P-T space from literature fits.
2. Convert those curves into P-S space by inverting the EOS relation \(T(S, P)\).
3. Resample the solidus and liquidus entropy curves onto a common pressure grid.
4. Save both the P-T and P-S versions to disk for later use by PROTEUS.

## Available parametrizations

The following directory names are supported and should be used exactly as written in the TOML configuration in the `melting_dir` parameter:

| Directory name      | Reference                     | DOI |
|--------------------|------------------------------|-----|
| `andrault_2011`    | Andrault et al. (2011)       | [10.1016/j.epsl.2011.02.006](https://doi.org/10.1016/j.epsl.2011.02.006) |
| `monteux_2016`     | Monteux et al. (2016)        | [10.1016/j.epsl.2016.05.010](https://doi.org/10.1016/j.epsl.2016.05.010) |
| `wolf_bower_2018`  | Wolf & Bower (2018)          | [10.1016/j.pepi.2017.11.004](https://doi.org/10.1016/j.pepi.2017.11.004) <br> [10.1051/0004-6361/201935710](https://doi.org/10.1051/0004-6361/201935710) |
| `katz_2003`        | Katz et al. (2003)           | [10.1029/2002GC000433](https://doi.org/10.1029/2002GC000433) |
| `fei_2021`         | Fei et al. (2021)            | [10.1038/s41467-021-21170-y](https://doi.org/10.1038/s41467-021-21170-y) |
| `belonoshko_2005`  | Belonoshko et al. (2005)     | [10.1103/PhysRevLett.94.195701](https://doi.org/10.1103/PhysRevLett.94.195701) |
| `fiquet_2010`      | Fiquet et al. (2010)         | [10.1126/science.1192448](https://doi.org/10.1126/science.1192448) |
| `hirschmann_2000`  | Hirschmann (2000)            | [10.1029/2000GC000070](https://doi.org/10.1029/2000GC000070) |
| `stixrude_2014`    | Stixrude (2014)              | [10.1098/rsta.2013.0076](https://doi.org/10.1098/rsta.2013.0076) |
| `lin_2024`         | Lin et al. (2024)            | [10.1038/s41561-024-01495-1](https://doi.org/10.1038/s41561-024-01495-1) |

## Generate melting curves

Before running PROTEUS, generate the lookup tables:

```console
python tools/solidus_func.py --all
```

Alternatively, generate a single parametrization using a specific flag (for example `--katz2003`, `--lin2024`).

This computes all parametrizations, converts them to P-T and P-S space, and stores them in:

```console
$FWL_DATA/interior_lookup_tables/Melting_curves/
```

---

**See also:** [Interior structure and energetics reference](config/interior.md) | [Configuration file](../How-to/config.md) | [Reference data](data.md)
