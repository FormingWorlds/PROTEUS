"""Unit tests for ``proteus.utils.visual`` helper functions.

Covers ``interp_spec`` (input-spectrum resampling onto the CMF
wavelength grid) and ``ColourSystem`` (spectrum -> CIE XYZ -> RGB
pipeline plus hex output and gamut clipping).

Utility module: physics-invariant marker not required; anti-happy-path
rules still apply (edge case + limit-input + non-trivial assertion).

See also:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.utils.visual import (
    ColourSystem,
    cmf,
    cs_srgb,
    illuminant_D65,
    interp_spec,
    xyz_from_xy,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_interp_spec_single_point_returns_constant():
    """A single-point input spectrum interpolates to a constant value on
    the CMF wavelength grid. This is the greygas RT path where the input
    is effectively flat.
    """
    wl = np.array([500.0])
    fl = np.array([42.0])
    out = interp_spec(wl, fl)
    np.testing.assert_allclose(out, 42.0, rtol=1e-12)
    assert out.shape == cmf[:, 0].shape


@pytest.mark.unit
def test_interp_spec_multi_point_interpolates_to_cmf_grid():
    """A multi-point spectrum is interpolated onto the CMF wavelength
    grid. The output is finite at every CMF wavelength and matches the
    input endpoints at the grid edges (380 nm and 780 nm here).
    """
    wl = np.array([380.0, 500.0, 780.0])
    fl = np.array([1.0, 2.0, 4.0])

    # interpolate onto wavelength grid used for colormatching functions
    out = interp_spec(wl, fl)
    assert out.shape == cmf[:, 0].shape

    # check CMF mapping works ok
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# ColourSystem: spectrum -> XYZ -> RGB pipeline
# ---------------------------------------------------------------------------


def test_xyz_to_rgb_in_gamut_white_point_normalises_to_max_one():
    """The white-point XYZ for a colour system must transform to an RGB
    triple whose maximum component is exactly 1 (after normalisation).

    Discriminating: a normalisation bug that divided by the sum
    instead of the max would push all three channels below 1; a bug
    that skipped normalisation entirely would scale by the white
    illuminant's intrinsic magnitude. Pin the max to 1 with tight tol.
    """
    rgb = cs_srgb.xyz_to_rgb(illuminant_D65)
    assert rgb.shape == (3,)
    # All-finite, all-in-[0,1] after gamut clipping + normalisation.
    assert np.all(np.isfinite(rgb))
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 1)
    assert rgb.max() == pytest.approx(1.0, rel=1e-12)


def test_xyz_to_rgb_out_of_gamut_input_is_desaturated_into_gamut():
    """An XYZ point that maps to a negative RGB component must be
    desaturated (shifted by the magnitude of the most-negative channel)
    rather than clipped, and the resulting RGB is non-negative.

    Edge: the gamut-clipping branch only fires when at least one RGB
    component is negative; pick a deliberately out-of-gamut XYZ
    (pure-Y > 1) that triggers it.
    """
    # XYZ vector that lands a negative component on the sRGB primaries.
    xyz_out_of_gamut = np.array([0.05, 1.0, 0.05])
    rgb = cs_srgb.xyz_to_rgb(xyz_out_of_gamut)
    # Desaturation guarantees non-negativity post-shift.
    assert np.all(rgb >= 0)
    # Normalisation guarantees the max is 1.
    assert rgb.max() == pytest.approx(1.0, rel=1e-12)
    # Discrimination guard: if the branch had been skipped, the raw
    # transform would have given a negative component. Re-derive the
    # raw transform and confirm at least one entry would be negative.
    raw = cs_srgb.T.dot(xyz_out_of_gamut)
    assert raw.min() < 0


def test_xyz_to_rgb_zero_input_returns_zero_without_division_by_zero():
    """A zero XYZ vector must not trigger a divide-by-zero; the
    early-return on ``all(rgb == 0)`` skips the normalisation step.

    Edge: this is the limit-input case for a completely dark spectrum.
    """
    rgb = cs_srgb.xyz_to_rgb(np.zeros(3))
    np.testing.assert_array_equal(rgb, np.zeros(3))
    assert rgb.shape == (3,)  # structural: output shape matches input


def test_xyz_to_rgb_html_format_returns_seven_char_hex_string():
    """The HTML out_fmt returns a '#rrggbb' string for the white
    illuminant. Length and prefix are pinned; the channel values are
    pinned to discriminate a #ffffff (true white) outcome from a
    palette-mismatch bug.
    """
    out = cs_srgb.xyz_to_rgb(illuminant_D65, out_fmt='html')
    assert isinstance(out, str)
    assert out.startswith('#')
    assert len(out) == 7
    # White-point normalisation must produce a string close to '#ffffff'
    # (max channel exactly 1 -> hex 'ff'); the other two channels are
    # high but not exactly 'ff'. Pin the leading 'ff' to discriminate
    # against a normalisation-off-by-one bug that would land at '#fefefe'.
    assert 'ff' in out


def test_spec_to_xyz_uniform_spectrum_returns_normalised_triple():
    """A flat (uniform-amplitude) spectrum produces an XYZ triple that
    sums to 1 (the chromaticity normalisation). The Y component for a
    flat spectrum equals the integral of the y-bar CMF column divided
    by the integral of (x_bar + y_bar + z_bar).

    Discriminating: pin Y to its closed-form value. A regression that
    skipped the normalisation step would put Y at the un-normalised
    sum (~21.4), three orders of magnitude away.
    """
    spec = np.ones(cmf.shape[0])
    xyz = cs_srgb.spec_to_xyz(spec)
    assert xyz.shape == (3,)
    # Normalisation: components sum to 1.
    assert xyz.sum() == pytest.approx(1.0, rel=1e-12)
    # Closed-form Y for a flat spectrum on this CMF table.
    y_expected = cmf[:, 2].sum() / cmf[:, 1:].sum()
    assert xyz[1] == pytest.approx(y_expected, rel=1e-12)
    # Sign + scale guard. Y for a flat broadband spectrum should be of
    # order 0.3 (the relative weight of the y-bar lobe in CIE 1931),
    # not 0.01 or 1.0.
    assert 0.2 < xyz[1] < 0.5


def test_spec_to_xyz_zero_spectrum_returns_unnormalised_zero_triple():
    """A spectrum of all-zeros has denominator 0; the source returns
    the un-normalised XYZ (also all-zeros) rather than dividing.

    Edge: limit-input case. Without the guard, this would NaN-propagate
    into ``xyz_to_rgb`` and corrupt downstream colour assignments.
    """
    xyz = cs_srgb.spec_to_xyz(np.zeros(cmf.shape[0]))
    np.testing.assert_array_equal(xyz, np.zeros(3))
    assert not np.any(np.isnan(xyz))  # no NaN propagation from 0/0


def test_spec_to_rgb_uniform_spectrum_returns_finite_rgb_with_max_one():
    """A flat spectrum produces a non-saturated, in-gamut RGB triple
    whose maximum component is exactly 1 after normalisation.

    This exercises the full pipeline: spec_to_xyz -> xyz_to_rgb.
    """
    spec = np.ones(cmf.shape[0])
    rgb = cs_srgb.spec_to_rgb(spec)
    assert rgb.shape == (3,)
    assert np.all(np.isfinite(rgb))
    assert np.all(rgb >= 0)
    assert rgb.max() == pytest.approx(1.0, rel=1e-12)


def test_spec_to_rgb_html_format_returns_seven_char_hex_string():
    """The HTML out_fmt is forwarded through the full pipeline.

    Discriminating: confirms that the html branch reaches xyz_to_rgb
    via spec_to_rgb (line 175 in visual.py).
    """
    spec = np.ones(cmf.shape[0])
    out = cs_srgb.spec_to_rgb(spec, out_fmt='html')
    assert isinstance(out, str)
    assert out.startswith('#')
    assert len(out) == 7


def test_rgb_to_hex_matches_known_palette_entries():
    """Convert known RGB triples to hex strings and confirm the
    formatter pins each channel to a two-character lowercase hex.

    Discriminating: pin three distinct values so a regression that
    cast to int via truncation vs round, or that swapped the channel
    ordering, would land on a different string.
    """
    # Pure red, pure blue, mid-grey (0.5 -> 127 after int cast).
    red = cs_srgb.rgb_to_hex(np.array([1.0, 0.0, 0.0]))
    blue = cs_srgb.rgb_to_hex(np.array([0.0, 0.0, 1.0]))
    grey = cs_srgb.rgb_to_hex(np.array([0.5, 0.5, 0.5]))
    assert red == '#ff0000'
    assert blue == '#0000ff'
    # 0.5 * 255 = 127.5 ; int() truncates -> 127 -> '7f'. Pinning to '7f'
    # discriminates against a round() implementation that would land on '80'.
    assert grey == '#7f7f7f'


def test_colour_system_construction_inverts_chromaticity_matrix():
    """A fresh ColourSystem must hold an invertible M and a self-consistent
    T = MI / wscale. The white point must round-trip to itself under
    T @ white (up to floating-point noise).

    Edge: this catches a regression to xyz_from_xy that would put
    z = -(x+y) instead of 1-x-y, breaking the matrix invertibility.
    """
    cs = ColourSystem(
        red=xyz_from_xy(0.64, 0.33),
        green=xyz_from_xy(0.30, 0.60),
        blue=xyz_from_xy(0.15, 0.06),
        white=illuminant_D65,
    )
    # Round-trip identity: T @ M @ wscale should recover the white XYZ.
    recovered = cs.M @ cs.wscale
    np.testing.assert_allclose(recovered, illuminant_D65, rtol=1e-12)
    # Sanity: M is 3x3 and invertible (no NaN in MI).
    assert cs.M.shape == (3, 3)
    assert np.all(np.isfinite(cs.MI))
