# juliacall and the Julia system image

Read this when `import proteus`, `import juliacall` or an AGNI call fails with
`could not load library ".../.juliaup/bin/../lib/julia/sys.so"`, or when pytest
exits with code 1 and prints nothing.

This happens on some machines and shells, not all. juliapkg recorded the juliaup
launcher (`~/.juliaup/bin/julia`) as the Julia executable, and juliacall looks for
the Julia system image next to it. Julia then ends the Python process, so there is
no Python traceback.

Point juliacall at the real Julia binary and at its project, both together
(juliacall refuses one without the other), in the shell before any PROTEUS, AGNI
or test command:

```bash
export PYTHON_JULIACALL_EXE="$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')"
export PYTHON_JULIACALL_PROJECT="$(python -c 'import juliapkg; print(juliapkg.project())')"
```

The first line asks Julia, through the launcher, for the path of the binary it
runs. The second gives the Julia project that juliacall uses for this Python
environment (`$CONDA_PREFIX/julia_env` in a conda env). Where juliacall starts
Julia without them, leave both unset.
