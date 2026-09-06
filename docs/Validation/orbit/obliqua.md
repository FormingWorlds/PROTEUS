# Validation: `src/proteus/orbit/obliqua.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.obliqua` against a published source or analytical
limit. The marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_obliqua.py::test_ln_from_lookup_enforces_love_number_reality_symmetry` | Reality condition for the frequency response of a real-valued (causal) linear physical system: $k(-\sigma) = k^*(\sigma)$ | n/a (analytical limit) | Pins that `LN_from_lookup` returns the complex conjugate of a lookup-table node's Love number when queried at the negative of that node's forcing frequency, rather than the same value or its negation. |

## Re-derivation note

A tidal Love number $k(\sigma)$ is the transfer function of a real-valued
input (the tide-raising potential) to a real-valued output (the body's
deformation), evaluated at forcing frequency $\sigma$. For any causal,
real-valued linear system, the transfer function obeys the Hermitian
symmetry $k(-\sigma) = k^*(\sigma)$ -- the same condition that underlies the
Kramers-Kronig relations. `LN_from_lookup` looks up Love numbers on a
one-sided ($\sigma \geq 0$) table and must apply this symmetry itself when a
negative-frequency mode is requested.

The test seeds a lookup-table node at $\sigma = 2\times 10^{-6}$ with
$k = 0.02 - 0.03i$ and queries the mode at $\sigma = -2\times 10^{-6}$
(`m=-2, k=0`). The expected result is $\mathrm{conj}(0.02 - 0.03i) =
0.02 + 0.03i$. A regression that instead returned the un-conjugated node
value, or one that only flipped the real part, would fail both the
`pytest.approx` pin and the sign discrimination guard (`imag > 0`).
