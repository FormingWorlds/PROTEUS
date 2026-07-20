This folder holds example PROTEUS configurations, one per subdirectory.

Run one with:

    proteus start -c examples/<name>/init_coupler.toml

The run writes its outputs (logs, data, and plots) under the top-level
output/ directory, in a subdirectory named by the config's params.out.path.
Those outputs are not committed, and the examples are not test cases; they
may not match the current version of the code.
