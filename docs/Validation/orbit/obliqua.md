# Validation: `src/proteus/orbit/obliqua.py`

## What this page checks

This page checks how `proteus.orbit.obliqua` looks up the tidal Love number of a body. The Love number $k(\sigma)$ measures how much a body deforms under a tide of forcing frequency $\sigma$; its imaginary part sets the tidal dissipation. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the tidal response at negative forcing frequencies has the wrong sign of dissipation. The case below says why the check exists, what it tests, how we know the right answer and why it matters; a box "Test details" ends the case. The `reference_pinned` marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry` | Reality condition for the frequency response of a real-valued (causal) linear physical system: $k(-\sigma) = k^*(\sigma)$ | n/a (analytical limit) | Pins that `LN_from_lookup` returns the complex conjugate of a lookup-table node's Love number when queried at the negative of that node's forcing frequency, rather than the same value or its negation. |

## Love number at a negative forcing frequency

**Why this check exists.** The lookup table holds Love numbers for $\sigma \geq 0$ only, so `LN_from_lookup` must apply a symmetry itself for a negative frequency. Returning the table value unchanged, or its negative, flips the sign of the imaginary part or of the whole response without an error message.

**What it tests.** The test seeds a table node at $\sigma = 2\times 10^{-6}$ with $k = 0.02 - 0.03i$ and queries the mode at $\sigma = -2\times 10^{-6}$ (`m=-2, k=0`). The result must be $\mathrm{conj}(0.02 - 0.03i) = 0.02 + 0.03i$, with a positive imaginary part.

**How we know the right answer.** For any causal, real-valued linear system the transfer function obeys $k(-\sigma) = k^*(\sigma)$, the condition that also underlies the Kramers-Kronig relations (see the derivation below).

**Why it matters.** With `orbit.planet_satellite_model` set to `ps1d` or `ps1d_evec`, the orbit wrapper calls `LN_from_lookup` to get the Love numbers of the satellite from its lookup table.

??? info "Test details"
    - Reference: reality condition of a causal linear system, $k(-\sigma) = k^*(\sigma)$ (analytical limit).
    - Tolerance: relative 1e-9 on the complex value; imaginary part above 0.
    - Test id: `tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry`
    - Command: `pytest tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry`

## Re-derivation note

A tidal Love number $k(\sigma)$ is the transfer function of a real-valued
input (the tide-raising potential) to a real-valued output (the body's
deformation), evaluated at forcing frequency $\sigma$. For any causal,
real-valued linear system, the transfer function obeys the Hermitian
symmetry $k(-\sigma) = k^*(\sigma)$, the same condition that underlies the
Kramers-Kronig relations. `LN_from_lookup` looks up Love numbers on a
one-sided ($\sigma \geq 0$) table and must apply this symmetry itself when a
negative-frequency mode is requested.

The test seeds a lookup-table node at $\sigma = 2\times 10^{-6}$ with
$k = 0.02 - 0.03i$ and queries the mode at $\sigma = -2\times 10^{-6}$
(`m=-2, k=0`). The expected result is $\mathrm{conj}(0.02 - 0.03i) =
0.02 + 0.03i$. A regression that instead returned the un-conjugated node
value, or one that only flipped the real part, would fail both the
`pytest.approx` pin and the sign discrimination guard (`imag > 0`).
