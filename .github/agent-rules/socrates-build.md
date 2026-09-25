# SOCRATES build flags

Read this before changing `tools/get_socrates.sh`, before debugging an
illegal-instruction crash from a restored SOCRATES tree, and before producing
numbers that have to be bit-reproducible.


## Default flags and the portable switch

By default `tools/get_socrates.sh` keeps the configure flags `-Ofast
-march=native`, which give the best performance on the build host. Set
`SOCRATES_PORTABLE_FLAGS=1` to compile with `-O2 -fno-fast-math` instead.

The `-march=native` default bakes the build host's CPU extensions into the
binary, so a compiled tree reused on a different processor aborts with an
illegal-instruction fault, while the portable flags run on any CPU. CI sets the
switch because its caches are restored across runner machines with mixed CPU
generations.

Dropping fast-math also removes compiler value reordering, the build-to-build
component of the ULP-level non-determinism that AGNI's Newton solver amplifies
into 1-2 % F_atm variance; run-to-run scatter from OpenMP threading remains
while OMPARG is set.

In portable mode the build fails loudly if a future SOCRATES release changes the
flag string, so no manual edit is needed.

## Full bit-reproducibility

For paper plots, CHILI, and SPIDER-parity work, install with

```bash
SOCRATES_PORTABLE_FLAGS=1 bash tools/get_socrates.sh
```

and also clear `OMPARG = -fopenmp` in `socrates/make/Mk_cmd`, then force a
recompile:

```bash
cd socrates/bin && make clean && cd .. && ./build_code
```

The clean is required: no make rule depends on `Mk_cmd`, so rebuilding without
it reuses the OpenMP objects unchanged. The install path keeps OpenMP enabled
and does not clear it automatically.
