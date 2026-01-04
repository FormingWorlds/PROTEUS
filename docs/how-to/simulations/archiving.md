# Archiving Output Files

## Why Archive?

Running PROTEUS, especially large grids of simulations, generates many files. Archiving reduces storage footprint by packing output files into compressed `.tar` archives.

## Automatic Archiving

Enable archiving via the configuration option `params.out.archive_mod`. PROTEUS will automatically archive output files according to your settings.

## Extracting Archives

To extract archived output files for analysis or visualization:

```console
proteus extract-archives -c [cfgfile]
```

## Re-archiving Files

To pack data files back into `.tar` archives after extraction:

```console
proteus create-archives -c [cfgfile]
```

This process is fully reversible—you can extract and re-archive as needed.

## Archive Contents

Archived `.tar` files typically contain:

- Atmosphere data files (`.nc` NetCDF files)
- Interior data (`.json` files)
- Other simulation outputs

The archive makes files inaccessible until extracted, reducing I/O overhead while maintaining data integrity.
