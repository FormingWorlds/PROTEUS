"""Unit tests for the Aragog/SPIDER config-parity refactor (Tier 1-4, 2026-04-08).

These tests cover the new / refactored pieces introduced by the
parity commits `52138bc8` through `47711120` on `tl/interior-refactor`:

- `interior_energetics.wrapper._is_spider_ps_format` - P-S format sniff
- `interior_energetics.wrapper._rectangularize_spider_ps_file` -
  normalization of SPIDER's quasi-regular P-S tables
- `config._interior.Interior.__attrs_post_init__` - Tier 4 deprecation
  alias resolution (num_tolerance, spider.tolerance_rel,
  spider.matprop_smooth_width)
- `config._interior` - Tier 3 physics-constant fields default values

All tests are pure-Python: no Julia, no SOCRATES, no SPIDER binary.
They use tmp_path fixtures for file-system checks and mock the
config surface where needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# =====================================================================
# _is_spider_ps_format sniffer
# =====================================================================


def test_is_spider_ps_format_accepts_canonical_header(tmp_path):
    """A file whose first line is `# 5 <nS> <nP>` is accepted."""
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    p = tmp_path / 'density_melt.dat'
    p.write_text('# 5 2020 95\n# Pressure, Entropy, Quantity\ndata\n')
    assert _is_spider_ps_format(str(p)) is True


def test_is_spider_ps_format_rejects_pt_header(tmp_path):
    """A file whose first line is a P-T header is rejected."""
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    p = tmp_path / 'density_melt.dat'
    p.write_text('#pressure temperature density\n0.0 1000.0 3000.0\n')
    assert _is_spider_ps_format(str(p)) is False


def test_is_spider_ps_format_rejects_missing_file(tmp_path):
    """A non-existent path is rejected (returns False, does not raise)."""
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    assert _is_spider_ps_format(str(tmp_path / 'nope.dat')) is False


def test_is_spider_ps_format_rejects_wrong_head_count(tmp_path):
    """A `# 6 ...` header (6 header lines, not SPIDER's canonical 5)
    is rejected because we only accept the canonical format."""
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    p = tmp_path / 'density_melt.dat'
    p.write_text('# 6 2020 95\n# different format\n')
    assert _is_spider_ps_format(str(p)) is False


# =====================================================================
# _rectangularize_spider_ps_file
# =====================================================================


def _write_spider_ps_file(path, n_P, n_S, drift=0.0):
    """Write a minimal SPIDER-format P-S file for testing.

    The data is intentionally trivial: Q(i,j) = i + 10*j so we can
    verify the reshape semantics. When drift > 0, P values drift
    across S slices by `drift` on the relative scale, mimicking
    SPIDER's quasi-regular layout.
    """
    rng = np.random.default_rng(0)
    lines = [
        f'# 5 {n_P} {n_S}\n',
        '# Pressure, Entropy, Quantity\n',
        '# column * scaling factor should be SI units\n',
        '# scaling factors (constant) for each column given on line below\n',
        '# 1.0e9 1.0 1.0\n',
    ]
    for j in range(n_S):
        S_j = float(j) * 100.0
        # P varies fastest (inner loop)
        for i in range(n_P):
            P_base = float(i) * 10.0
            # Apply drift: each slice has slightly different P values
            jitter = rng.uniform(-drift, drift) * max(P_base, 1.0)
            P_val = P_base + jitter
            Q_val = float(i + 10 * j)
            lines.append(f'{P_val:.12e} {S_j:.12e} {Q_val:.12e}\n')
    path.write_text(''.join(lines))


def test_rectangularize_clean_file_round_trip(tmp_path):
    """A file that is already strictly rectangular is preserved
    byte-for-byte in meaning (data values unchanged)."""
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    src = tmp_path / 'src.dat'
    dst = tmp_path / 'dst.dat'
    _write_spider_ps_file(src, n_P=4, n_S=3, drift=0.0)
    _rectangularize_spider_ps_file(str(src), str(dst))

    out = np.genfromtxt(dst, skip_header=5)
    assert out.shape == (12, 3)

    # Canonical ordering: P varies fastest, S slowest, Q = i + 10*j
    for row_idx in range(12):
        j = row_idx // 4  # S slice
        i = row_idx % 4   # P index
        assert out[row_idx, 0] == pytest.approx(float(i) * 10.0)
        assert out[row_idx, 1] == pytest.approx(float(j) * 100.0)
        assert out[row_idx, 2] == pytest.approx(float(i + 10 * j))


def test_rectangularize_quasi_regular_drift_snapped(tmp_path):
    """A file with small P drift (SPIDER's actual layout) gets
    snapped to the first slice's P values. All output rows have
    the same P_canonical[i] for a given i modulo n_P."""
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    src = tmp_path / 'src.dat'
    dst = tmp_path / 'dst.dat'
    _write_spider_ps_file(src, n_P=5, n_S=4, drift=1e-9)
    _rectangularize_spider_ps_file(str(src), str(dst))

    out = np.genfromtxt(dst, skip_header=5)
    n_P, n_S = 5, 4
    assert out.shape == (n_P * n_S, 3)

    # Canonical P grid: column 0 of first 5 rows
    P_canonical = out[:n_P, 0]

    # Every subsequent S slice must have EXACTLY the same P values
    for j in range(1, n_S):
        P_slice = out[j * n_P:(j + 1) * n_P, 0]
        np.testing.assert_array_equal(P_slice, P_canonical)


def test_rectangularize_rejects_large_drift(tmp_path):
    """A file with >1e-4 relative P drift is NOT quasi-regular and
    must be rejected with a clear error. This guards against silently
    turning a genuine irregular grid into nonsense."""
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    src = tmp_path / 'src.dat'
    dst = tmp_path / 'dst.dat'
    _write_spider_ps_file(src, n_P=4, n_S=3, drift=0.1)  # 10% drift
    with pytest.raises(ValueError, match='not quasi-rectangular'):
        _rectangularize_spider_ps_file(str(src), str(dst))


def test_rectangularize_rejects_header_row_count_mismatch(tmp_path):
    """A file whose header NX*NY doesn't match the actual row count
    is rejected."""
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    src = tmp_path / 'src.dat'
    dst = tmp_path / 'dst.dat'
    # Header says 5x3 = 15 but file has only 10 data rows
    lines = [
        '# 5 5 3\n',
        '# Pressure, Entropy, Quantity\n',
        '# column * scaling factor should be SI units\n',
        '# scaling factors (constant) for each column given on line below\n',
        '# 1.0e9 1.0 1.0\n',
    ]
    for i in range(10):
        lines.append(f'{i:.1f} 0.0 {i:.1f}\n')
    src.write_text(''.join(lines))

    with pytest.raises(ValueError, match='NX\\*NY'):
        _rectangularize_spider_ps_file(str(src), str(dst))


# =====================================================================
# Tier 4 deprecation alias resolution
# =====================================================================


def test_tier4_num_tolerance_alias_copies_to_rtol():
    """Setting the deprecated num_tolerance alias copies its value to
    rtol and emits a DeprecationWarning."""
    from proteus.config._interior import Interior

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ie = Interior(num_tolerance=3.14e-7)
    assert ie.rtol == pytest.approx(3.14e-7)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and 'num_tolerance' in str(w.message)
        for w in caught
    )


def test_tier4_num_tolerance_and_rtol_conflict_raises():
    """If BOTH num_tolerance and rtol are set to distinct non-default
    values, loading must raise ValueError — we can't guess."""
    from proteus.config._interior import Interior

    with pytest.raises(ValueError, match='num_tolerance'):
        Interior(num_tolerance=1e-6, rtol=1e-9)


def test_tier4_num_tolerance_and_rtol_same_value_silent():
    """If both are set to the SAME value, no warning fires."""
    from proteus.config._interior import Interior

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ie = Interior(num_tolerance=1e-8, rtol=1e-8)
    assert ie.rtol == pytest.approx(1e-8)
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_tier4_spider_tolerance_rel_alias_copies_to_rtol():
    """Setting the deprecated Spider.tolerance_rel alias copies its
    value to top-level Interior.rtol and warns."""
    from proteus.config._interior import Interior, Spider

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ie = Interior(spider=Spider(tolerance_rel=5.5e-9))
    assert ie.rtol == pytest.approx(5.5e-9)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and 'tolerance_rel' in str(w.message)
        for w in caught
    )


def test_tier4_spider_matprop_smooth_width_alias():
    """Setting the deprecated Spider.matprop_smooth_width alias copies
    its value to the top-level field and warns."""
    from proteus.config._interior import Interior, Spider

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ie = Interior(spider=Spider(matprop_smooth_width=0.055))
    assert ie.matprop_smooth_width == pytest.approx(0.055)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and 'matprop_smooth_width' in str(w.message)
        for w in caught
    )


def test_tier4_spider_tolerance_rel_conflict_raises():
    """Setting Spider.tolerance_rel to a value different from rtol
    at the top level must raise."""
    from proteus.config._interior import Interior, Spider

    with pytest.raises(ValueError, match='spider\\.tolerance_rel'):
        Interior(rtol=1e-9, spider=Spider(tolerance_rel=1e-6))


# =====================================================================
# Tier 3 physics-constant defaults match SPIDER exactly
# =====================================================================


def test_tier3_solid_log10visc_default_matches_spider():
    """Regression guard: Aragog previously hardcoded solid viscosity
    1e21 (log10=21), SPIDER uses 22.0. The parity refactor fixes
    Aragog's value to match SPIDER. The config default must be 22.0,
    NOT 21.0, or the 10x rheology undervalue returns."""
    from proteus.config._interior import Interior

    ie = Interior()
    assert ie.solid_log10visc == 22.0, (
        'Aragog must use the same solid viscosity as SPIDER. Previous '
        'hardcoded value was 1e21 (log10=21); SPIDER uses 22.0.'
    )
    assert 10.0 ** ie.solid_log10visc == 1e22


def test_tier3_adams_williamson_rhos_default_matches_spider():
    """Aragog previously hardcoded 4090 (0.27% off from SPIDER's
    4078.95095544). The refactor unifies both on SPIDER's value."""
    from proteus.config._interior import Interior

    ie = Interior()
    assert ie.adams_williamson_rhos == pytest.approx(4078.95095544)
    assert ie.adams_williamson_rhos != 4090.0


def test_tier3_physics_constants_all_set_to_spider_defaults():
    """Every Tier 3 field resolves to the SPIDER-matching default
    the refactor promises."""
    from proteus.config._interior import Interior

    ie = Interior()
    assert ie.adams_williamson_rhos == pytest.approx(4078.95095544)
    assert ie.adams_williamson_beta == pytest.approx(1.1115348931000002e-07)
    assert ie.adiabatic_bulk_modulus == 260e9
    assert ie.melt_log10visc == 2.0
    assert ie.solid_log10visc == 22.0
    assert ie.melt_cond == 4.0
    assert ie.solid_cond == 4.0
    assert ie.eddy_diffusivity_thermal == 1.0
    assert ie.eddy_diffusivity_chemical == 1.0
    assert ie.latent_heat_of_fusion == 4e6
    assert ie.phase_transition_width == 0.1
    assert ie.core_tfac_avg == 1.147
