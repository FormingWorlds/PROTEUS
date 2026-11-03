# Validation: `src/proteus/orbit/obliqua.py`

## What this page checks

This page checks how `proteus.orbit.obliqua` looks up the tidal Love number of a body. The Love number $k(\sigma)$ measures how much a body deforms under a tide of forcing frequency $\sigma$ (in rad s$^{-1}$), and its imaginary part sets the tidal dissipation. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the tidal response at negative forcing frequencies has the wrong sign of dissipation. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case. The `reference_pinned` marker is registered in `pyproject.toml`.

| Test id | Reference | Scope |
|---|---|---|
| `tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry` | Reality condition for the frequency response of a real-valued (causal) linear physical system: $k(-\sigma) = k^*(\sigma)$ | Pins that `LN_from_lookup` returns the complex conjugate of a lookup-table node's Love number when queried at the negative of that node's forcing frequency, rather than the same value or its negation. |

## Love number at a negative forcing frequency

**Why this check exists.** The lookup table holds Love numbers for $\sigma \geq 0$ only, so `LN_from_lookup` must apply a symmetry itself for a negative frequency. Returning the table value unchanged, or its negative, flips the sign of the imaginary part or of the whole response without an error message.

**What it tests.** The test seeds a table node at $\sigma = 2\times 10^{-6}$ with $k = 0.02 - 0.03i$ and queries the mode at $\sigma = -2\times 10^{-6}$ (`m=-2, k=0`). The result must be $\mathrm{conj}(0.02 - 0.03i) = 0.02 + 0.03i$, with a positive imaginary part.

**How we know the right answer.** For any causal, real-valued linear system the transfer function obeys $k(-\sigma) = k^*(\sigma)$, the condition that also underlies the Kramers-Kronig relations (see the derivation below).

**Known limit.** The test queries the mirror of 1 table node, so it does not check the interpolation between nodes at negative frequencies.

**Why it matters.** With `orbit.planet_satellite_model` set to `ps1d` or `ps1d_evec`, the orbit wrapper calls `LN_from_lookup` to get the Love numbers of the satellite from its lookup table. In these models, their imaginary part sets the tidal dissipation.

![Love number of the table node and of the query at the negative frequency in the complex plane](../../assets/validation/orbit_obliqua_love_number_symmetry.svg#only-light){ width="80%" }
![Love number of the table node and of the query at the negative frequency in the complex plane](../../assets/validation/orbit_obliqua_love_number_symmetry_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the real part and the y-axis the imaginary part of the Love number $k$, which describes the tidal response. The cross is the table node $0.02 - 0.03i$ at the tidal forcing frequency $\sigma = +2\times10^{-6}$ rad s$^{-1}$ (context). The hollow square is its complex conjugate, the asserted expected value. The filled circle is the asserted code value at $\sigma = -2\times10^{-6}$, and the dashed line is the real axis. A pass means that the circle sits in the square, above the axis.

??? info "Test details"
    - Reference: reality condition of a causal linear system, $k(-\sigma) = k^*(\sigma)$ (analytical limit).
    - Tolerance: relative 1e-9 on the complex value; imaginary part above 0.
    - Test id: `tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry`
    - Command: `pytest -m reference_pinned tests/orbit/test_obliqua.py --validation-plots`

## Re-derivation note

A tidal Love number $k(\sigma)$ is the transfer function from a real-valued input (the tide-raising potential) to a real-valued output (the deformation of the body). This is why the symmetry $k(-\sigma) = k^*(\sigma)$ holds. A lookup that returned the node value unchanged, or flipped only its real part, gives an imaginary part of $-0.03$ and fails both the value pin and the guard `imag > 0`.
