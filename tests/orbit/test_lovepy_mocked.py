"""Unit tests for proteus.orbit.lovepy: LovePy tidal heating wrapper.

LovePy is the Julia-backed tidal heating module. The Python wrapper
in ``src/proteus/orbit/lovepy.py`` packages PROTEUS interior arrays
into Julia types via ``juliacall``, calls
``LovePy.calc_lovepy_tides``, and writes the per-cell tidal power
density back into ``Interior_t.tides``.

These unit tests mock the Julia side (``juliacall`` types and
``jl.calc_lovepy_tides``) so the wrapper paths execute without a
real Julia + LovePy install. ``juliacall`` is importable in the
standard PROTEUS environment; ``LovePy.jl`` is an opt-in package
gated on ``./tools/get_lovepy.sh``.

Module scope:

- ``import_lovepy`` calls ``jl.seval('using LovePy')`` exactly once.
- ``_jlarr`` and ``_jlsca`` invoke ``juliacall.convert`` with the
  documented Julia destination types. A regression that swapped
  ``Array[prec, 1]`` for ``Array[prec, 2]`` or for a different
  precision would fail the type-argument assertion.
- ``run_lovepy`` dispatch by ``interior_energetics.module``:
  - dummy / boundary: fully-liquid early return when ``visc[0] <
    visc_thresh`` returns 0.0 and writes no tides.
  - dummy / boundary: heated branch builds the two-cell three-edge
    arrays, calls ``calc_lovepy_tides``, writes the per-cell power
    into ``interior_o.tides[0]``, and returns Imk2.
  - spider / aragog: fully-liquid early return when ``i_top <= 1``.
  - spider / aragog: heated branch builds per-cell arrays up to
    ``i_top``, calls ``calc_lovepy_tides``, writes per-cell tides,
    and returns Imk2. SPIDER ordering gets reversed so i=0 sits at
    the CMB.
  - ``juliacall.JuliaError`` is wrapped into ``RuntimeError``.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip('juliacall')

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_interior_t(module: str, nlev_s: int = 5):
    """Build a minimal Interior_t-like stand-in. The wrapper only
    reads ``density``, ``visc``, ``shear``, ``bulk``, ``mass``,
    ``radius``, and ``nlev_s``; it writes ``tides``.
    """
    interior_o = types.SimpleNamespace()
    interior_o.nlev_s = nlev_s
    interior_o.density = np.linspace(3000.0, 12000.0, nlev_s)
    interior_o.visc = np.full(nlev_s, 1e18)  # high viscosity (heating)
    interior_o.shear = np.full(nlev_s, 5e10)
    interior_o.bulk = np.full(nlev_s, 1.5e11)
    interior_o.mass = np.full(nlev_s, 1e22)
    interior_o.radius = np.linspace(3.4e6, 6.4e6, nlev_s)
    interior_o.tides = np.zeros(nlev_s)
    return interior_o


def _make_config(module: str, visc_thresh: float = 1e9, ncalc: int = 1000):
    cfg = types.SimpleNamespace()
    cfg.interior_energetics = types.SimpleNamespace()
    cfg.interior_energetics.module = module
    cfg.orbit = types.SimpleNamespace()
    cfg.orbit.lovepy = types.SimpleNamespace()
    cfg.orbit.lovepy.visc_thresh = visc_thresh
    cfg.orbit.lovepy.ncalc = ncalc
    return cfg


# ---------------------------------------------------------------------------
# import_lovepy.
# ---------------------------------------------------------------------------


def test_import_lovepy_calls_jl_seval_with_using_lovepy(monkeypatch):
    """``import_lovepy`` issues a single ``jl.seval('using LovePy')``
    call. A regression that dropped or changed the import string
    would silently leave the LovePy.jl symbols unbound.
    """
    from proteus.orbit import lovepy as lovepy_mod

    fake_jl = MagicMock(name='jl')
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    lovepy_mod.import_lovepy()
    fake_jl.seval.assert_called_once_with('using LovePy')
    assert fake_jl.seval.call_count == 1  # exactly one import, not repeated


# ---------------------------------------------------------------------------
# _jlarr / _jlsca: julia type conversion contract.
# ---------------------------------------------------------------------------


def test_jlarr_converts_to_julia_array_with_documented_element_type(monkeypatch):
    """``_jlarr`` flattens the numpy array, converts via
    ``juliacall.convert``, and targets the
    ``jl.Array[jl.LovePy.prec, 1]`` Julia type. Pin the call
    signature so a regression that broadened the dim from 1 to 2 or
    swapped the element type surfaces.
    """
    from proteus.orbit import lovepy as lovepy_mod

    fake_juliacall = MagicMock(name='juliacall')
    fake_juliacall.convert = MagicMock(return_value='converted_array')
    fake_jl = MagicMock(name='jl')
    fake_jl.Array = MagicMock()
    fake_jl.LovePy = MagicMock()
    fake_jl.LovePy.prec = 'prec_sentinel'
    fake_jl.Array.__getitem__ = MagicMock(return_value='destination_type')
    monkeypatch.setattr(lovepy_mod, 'juliacall', fake_juliacall)
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)

    arr = np.array([1.0, 2.0, 3.0])
    out = lovepy_mod._jlarr(arr)
    fake_jl.Array.__getitem__.assert_called_once_with(('prec_sentinel', 1))
    fake_juliacall.convert.assert_called_once()
    call_args, _ = fake_juliacall.convert.call_args
    assert call_args[0] == 'destination_type'
    np.testing.assert_array_equal(call_args[1], arr.astype(float).flatten())
    assert out == 'converted_array'


def test_jlsca_converts_to_julia_prec_scalar(monkeypatch):
    """``_jlsca`` converts a python float to the LovePy precision
    type via ``juliacall.convert(jl.LovePy.prec, sca)``. Pin the
    destination-type argument.
    """
    from proteus.orbit import lovepy as lovepy_mod

    fake_juliacall = MagicMock(name='juliacall')
    fake_juliacall.convert = MagicMock(return_value='converted_scalar')
    fake_jl = MagicMock(name='jl')
    fake_jl.LovePy = MagicMock()
    fake_jl.LovePy.prec = 'prec_sentinel'
    monkeypatch.setattr(lovepy_mod, 'juliacall', fake_juliacall)
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)

    out = lovepy_mod._jlsca(0.5)
    fake_juliacall.convert.assert_called_once_with('prec_sentinel', 0.5)
    assert out == 'converted_scalar'


# ---------------------------------------------------------------------------
# run_lovepy: dummy / boundary fully-liquid early return.
# ---------------------------------------------------------------------------


def test_run_lovepy_dummy_returns_zero_when_top_cell_below_visc_thresh(monkeypatch):
    """Under ``interior_energetics.module='dummy'``, ``run_lovepy``
    returns 0.0 immediately when the top-cell viscosity is below
    the configured threshold (fully-liquid mantle, no tidal
    dissipation).

    Discrimination: the Julia ``calc_lovepy_tides`` is not called
    on the early-return path; pin the call count at 0.
    """
    from proteus.orbit import lovepy as lovepy_mod

    # Patch jl and the helpers so the wrapper's internals never call
    # real Julia.
    fake_jl = MagicMock(name='jl')
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    interior_o = _make_interior_t(module='dummy', nlev_s=3)
    # Drop top-cell viscosity below the threshold.
    interior_o.visc[0] = 1.0
    cfg = _make_config(module='dummy', visc_thresh=1e9)
    hf_row = {'orbital_period': 86400.0 * 365.0, 'eccentricity': 0.1}

    out = lovepy_mod.run_lovepy(hf_row, dirs={}, interior_o=interior_o, config=cfg)

    assert out == pytest.approx(0.0, abs=1e-12)
    fake_jl.calc_lovepy_tides.assert_not_called()


# ---------------------------------------------------------------------------
# run_lovepy: spider / aragog fully-liquid early return.
# ---------------------------------------------------------------------------


def test_run_lovepy_aragog_returns_zero_when_full_mantle_below_visc_thresh(monkeypatch):
    """Under ``interior_energetics.module='aragog'``,
    ``run_lovepy`` returns 0.0 when ``i_top <= 1`` (no contiguous
    region of high-viscosity cells from the bottom up).

    Discrimination: the Julia tides call does not fire.
    """
    from proteus.orbit import lovepy as lovepy_mod

    fake_jl = MagicMock(name='jl')
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    interior_o = _make_interior_t(module='aragog', nlev_s=4)
    # No cell above threshold -> i_top stays at 0.
    interior_o.visc[:] = 1.0
    cfg = _make_config(module='aragog', visc_thresh=1e9)
    hf_row = {'orbital_period': 1e7, 'eccentricity': 0.0}

    out = lovepy_mod.run_lovepy(hf_row, dirs={}, interior_o=interior_o, config=cfg)
    assert out == pytest.approx(0.0, abs=1e-12)
    fake_jl.calc_lovepy_tides.assert_not_called()


# ---------------------------------------------------------------------------
# run_lovepy: dummy / boundary heated branch.
# ---------------------------------------------------------------------------


def test_run_lovepy_dummy_heated_branch_writes_tides_and_returns_imk2(monkeypatch):
    """Heated dummy / boundary path: ``run_lovepy`` packages the
    two-cell three-edge arrays, calls ``calc_lovepy_tides``, writes
    the per-cell power into ``interior_o.tides[0]``, and returns
    ``float(Imk2)``.

    Discrimination: per-cell tides slot is populated; return value
    matches the Imk2 mock; calc_lovepy_tides called once.
    """
    from proteus.orbit import lovepy as lovepy_mod

    fake_jl = MagicMock(name='jl')
    fake_jl.calc_lovepy_tides = MagicMock(return_value=(np.array([0.0, 1.5e-6]), 0.05, -0.0125))
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    interior_o = _make_interior_t(module='dummy', nlev_s=3)
    cfg = _make_config(module='dummy', visc_thresh=1e9, ncalc=1000)
    hf_row = {'orbital_period': 1e7, 'eccentricity': 0.1}

    out = lovepy_mod.run_lovepy(hf_row, dirs={}, interior_o=interior_o, config=cfg)
    assert fake_jl.calc_lovepy_tides.call_count == 1
    # Tides slot 0 populated from power_prf[1].
    assert interior_o.tides[0] == pytest.approx(1.5e-6, rel=1e-12)
    # Imk2 returned as Python float.
    assert isinstance(out, float)
    assert out == pytest.approx(-0.0125, rel=1e-12)


# ---------------------------------------------------------------------------
# run_lovepy: spider / aragog heated branch.
# ---------------------------------------------------------------------------


def test_run_lovepy_aragog_heated_branch_writes_per_cell_tides(monkeypatch):
    """Heated aragog path: ``run_lovepy`` builds arrays up to
    ``i_top``, calls ``calc_lovepy_tides``, writes per-cell power
    into the prefix ``[:i_top]`` of ``interior_o.tides`` (with the
    bottom-cell value duplicated to slot 0), and returns Imk2.

    Discrimination: per-cell tides array gets the power values in
    positions [0, i_top); positions [i_top, nlev_s) stay at zero.
    """
    from proteus.orbit import lovepy as lovepy_mod

    # 5-cell mantle with all cells above visc_thresh, so i_top = 4.
    interior_o = _make_interior_t(module='aragog', nlev_s=5)
    # Visc above threshold for all cells.
    interior_o.visc[:] = 1e18
    cfg = _make_config(module='aragog', visc_thresh=1e9, ncalc=1000)
    hf_row = {'orbital_period': 1e7, 'eccentricity': 0.05}

    # power_prf has i_top entries.
    power_prf = np.array([1e-6, 2e-6, 3e-6, 4e-6])  # length i_top = 4
    fake_jl = MagicMock(name='jl')
    fake_jl.calc_lovepy_tides = MagicMock(return_value=(power_prf, 5.0, -0.025))
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    out = lovepy_mod.run_lovepy(hf_row, dirs={}, interior_o=interior_o, config=cfg)
    # tides[0:i_top] holds the power, with tides[0] duplicated from tides[1].
    assert interior_o.tides[1] == pytest.approx(2e-6, rel=1e-12)
    assert interior_o.tides[2] == pytest.approx(3e-6, rel=1e-12)
    assert interior_o.tides[3] == pytest.approx(4e-6, rel=1e-12)
    # tides[0] is duplicated from tides[1].
    assert interior_o.tides[0] == pytest.approx(2e-6, rel=1e-12)
    # Tail beyond i_top stays zero.
    assert interior_o.tides[4] == pytest.approx(0.0, abs=1e-30)
    assert isinstance(out, float)
    assert out == pytest.approx(-0.025, rel=1e-12)


def test_run_lovepy_spider_heated_branch_reverses_order(monkeypatch):
    """Heated SPIDER path: arrays are reversed at entry so i=0 sits
    at the CMB, then reversed again on write-back so the
    ``interior_o.tides`` array remains in surface-to-CMB ordering
    on disk.

    Discrimination: post-call ``interior_o.tides`` reflects the
    SPIDER convention (the LovePy output is mirrored).
    """
    from proteus.orbit import lovepy as lovepy_mod

    interior_o = _make_interior_t(module='spider', nlev_s=5)
    # All cells above threshold.
    interior_o.visc[:] = 1e18
    cfg = _make_config(module='spider', visc_thresh=1e9, ncalc=1000)
    hf_row = {'orbital_period': 1e7, 'eccentricity': 0.05}

    power_prf = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    fake_jl = MagicMock(name='jl')
    fake_jl.calc_lovepy_tides = MagicMock(return_value=(power_prf, 5.0, -0.030))
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    out = lovepy_mod.run_lovepy(hf_row, dirs={}, interior_o=interior_o, config=cfg)
    # Under SPIDER ordering, the tides array is reversed on write
    # so the entry that was at index 1 (post-duplication) lands at
    # ``-2`` in the surface-to-CMB array. tides[-1] holds the prefix
    # duplicated entry, tides[-2] the original power_prf[1], etc.
    assert interior_o.tides[-1] == pytest.approx(2e-6, rel=1e-12)
    assert interior_o.tides[-2] == pytest.approx(2e-6, rel=1e-12)
    assert interior_o.tides[-3] == pytest.approx(3e-6, rel=1e-12)
    assert interior_o.tides[-4] == pytest.approx(4e-6, rel=1e-12)
    # Far end (surface side) stays zero.
    assert interior_o.tides[0] == pytest.approx(0.0, abs=1e-30)
    assert out == pytest.approx(-0.030, rel=1e-12)


# ---------------------------------------------------------------------------
# run_lovepy: JuliaError wrapped into RuntimeError.
# ---------------------------------------------------------------------------


def test_run_lovepy_julia_error_wrapped_into_runtime_error(monkeypatch):
    """``juliacall.JuliaError`` from ``calc_lovepy_tides`` is caught
    and re-raised as ``RuntimeError``. ``UpdateStatusfile`` is
    called with status code 26 before the re-raise.
    """
    import juliacall as real_juliacall

    from proteus.orbit import lovepy as lovepy_mod

    fake_jl = MagicMock(name='jl')
    fake_jl.calc_lovepy_tides = MagicMock(
        side_effect=real_juliacall.JuliaError('mock LovePy crash')
    )
    monkeypatch.setattr(lovepy_mod, 'jl', fake_jl)
    monkeypatch.setattr(lovepy_mod, '_jlarr', lambda a: a)
    monkeypatch.setattr(lovepy_mod, '_jlsca', lambda s: s)

    updates: list[tuple] = []

    def fake_update_statusfile(dirs, code):
        updates.append((dirs, code))

    monkeypatch.setattr(lovepy_mod, 'UpdateStatusfile', fake_update_statusfile)

    interior_o = _make_interior_t(module='dummy', nlev_s=3)
    cfg = _make_config(module='dummy', visc_thresh=1e9, ncalc=1000)
    hf_row = {'orbital_period': 1e7, 'eccentricity': 0.1}

    with pytest.raises(RuntimeError, match=r'(?i)lovepy'):
        lovepy_mod.run_lovepy(
            hf_row, dirs={'output': '/tmp'}, interior_o=interior_o, config=cfg
        )
    # UpdateStatusfile recorded the error code.
    assert len(updates) == 1
    assert updates[0][1] == 26
