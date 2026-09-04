"""Unit tests for proteus.orbit.obliqua: the Obliqua tidal heating wrapper.

Obliqua is the Julia-backed tidal heating module for the joint
solid/mushy/fluid interior tidal response. The Python wrapper in
``src/proteus/orbit/obliqua.py`` packages PROTEUS interior arrays and
config into Julia types via ``juliacall``, calls
``Obliqua.run_tides``, and writes the per-cell tidal power density
back into ``Interior_t.tides`` and the Love-number spectrum into
``Tides_t``.

These unit tests mock the Julia side (``jl.Obliqua.*`` and
``juliacall.convert``) so the wrapper's orchestration logic executes
without a real Julia + Obliqua install, following the same approach
as ``test_lovepy_mocked.py`` for the sibling LovePy module. The real
``Tides_t`` container from ``proteus.orbit.common`` is used
unmocked, since it is pure Python dataclass logic, not a Julia
boundary.

Exercises:

- ``import_obliqua``: single ``jl.seval('using Obliqua')`` call.
- ``to_julia_dict``: recursive dict/list conversion, nesting preserved.
- ``_jlarr`` / ``_jlsca_float`` / ``_jlsca_prec``: ``juliacall.convert``
  destination-type contract.
- ``run_obliqua`` dispatch by ``config.orbit.perturber`` (star vs.
  satellite orbital state) and by ``interior_energetics.module``
  (dummy two-cell construction; SPIDER array reversal on the way in
  and back out; direct aragog-style write with no reversal).
- ``run_obliqua``: total tidal power is conserved (sum-invariant)
  under the SPIDER reversal, since reversal is a permutation.
- ``run_obliqua``: ``juliacall.JuliaError`` is wrapped into
  ``RuntimeError``, with ``UpdateStatusfile`` called with code 26.
- ``run_obliqua``: results are stored into ``tides_o`` under
  ``primary='planet'`` keyed by the configured perturber.
- ``lookup_from_interior``: JSON IC -> Obliqua call -> netCDF lookup
  table round trip, including the core-density/core-mass split
  (density[0] becomes ``core_density``; the rest is passed as
  ``rho``).
- ``LN_from_lookup``: pure NumPy/SciPy post-processing with no Julia
  boundary, so these are genuine, mock-free physics tests: exact
  interpolation at a lookup node, the Love-number reality/symmetry
  condition (a real-valued response function must satisfy
  ``LNk(-sigma) == conj(LNk(sigma))``), the missing-degree error
  contract, the negative-(m,k) zeroing convention, the forcing
  frequency formula ``sigma_s = m*axial_freq - k*orbit_freq``, and the
  lookup-table cache/regeneration dispatch (``.json`` triggers
  ``lookup_from_interior`` once and caches the result; ``.nc`` loads
  directly with no regeneration).
- ``read_ncdf``/``read_ncdfs``: netCDF variable round trip, and that
  ``read_ncdfs`` orders its output by the caller's ``times`` list, not
  filesystem order.
- ``setup_logging``: ``jl.Obliqua.setup_logging`` call-argument
  contract (log path, verbosity passthrough).
- ``sync_log_files``: copy-and-clear contract, the missing-file
  fallback, and a pinned discrepancy in the *return value* (see
  below).

Known testability gap (not worked around): the aragog/non-SPIDER
branch scales a bulk-averaged tidal power by ``sum(mass)`` purely for
a ``log.debug`` line; the scaled value has no other observable
effect, so it is not independently pinned here (log-line-only
assertions are an explicitly discouraged pattern -- see
``.github/.claude/rules/proteus-tests.md`` section 16). The test for
that branch only confirms the division executes without raising and
that the unflipped profile is written to ``interior_o.tides``.

A second, similar gap is pinned rather than worked around in
``sync_log_files``: the function's docstring says it "returns the
list of lines that were copied," but the returned list is the
*original* lines as read, not the NULL-prefix-cleaned lines actually
written to the PROTEUS logfile (the cleaned text is a locally
rebound loop variable, never written back into the list). A caller
scanning the returned lines for a failure-mode marker at the start
of line 0 would see the uncleaned text.

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_building.md
- docs/How-to/test_categorization.md
"""

from __future__ import annotations

import json
import os
import types
from unittest.mock import MagicMock

import netCDF4 as nc
import numpy as np
import pytest

pytest.importorskip('juliacall')

from proteus.orbit.common import Tides_t

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_interior_t(nlev_s: int):
    """Minimal Interior_t-like stand-in with nlev_s cells (nlev_s + 1
    radius edges), matching the layout in
    ``proteus.interior_energetics.common.Interior_t``.
    """
    interior_o = types.SimpleNamespace()
    interior_o.nlev_s = nlev_s
    interior_o.density = np.linspace(3000.0, 5000.0, nlev_s)
    interior_o.visc = np.full(nlev_s, 1e20)
    interior_o.shear = np.full(nlev_s, 6e10)
    interior_o.bulk = np.full(nlev_s, 2e11)
    interior_o.phi = np.zeros(nlev_s)
    interior_o.mass = np.full(nlev_s, 1e20)
    interior_o.radius = np.linspace(3.0e6, 6.4e6, nlev_s + 1)
    interior_o.tides = np.zeros(nlev_s)
    return interior_o


def _make_config(module: str, perturber: str):
    """Fake Config namespace exposing exactly the attribute paths that
    ``run_obliqua`` reads, mirroring the ``[tool.proteus...]``
    ``Obliqua``/``ObliquaSolid``/``ObliquaMushy``/``ObliquaFluid``
    field names in ``src/proteus/config/_orbit.py``.
    """
    cfg = types.SimpleNamespace()
    cfg.orbit = types.SimpleNamespace()
    cfg.orbit.perturber = perturber

    ob = types.SimpleNamespace()
    ob.store_3D = False
    ob.enforce_ec = True
    ob.optimize_scales = False
    ob.solid_shell = True
    ob.min_frac = 0.02
    ob.visc_l = 1e2
    ob.visc_lus = 5e5
    ob.visc_s = 1e22
    ob.visc_sus = 5e5
    ob.n = [2]
    ob.m = [0, 2]
    ob.k_min = 'none'
    ob.k_max = 'none'
    ob.material_mu = 'andrade'
    ob.material_k = 'andrade'
    ob.alpha = 0.3
    ob.module_solid = 'solid0d'
    ob.module_mushy = 'none'
    ob.module_fluid = 'fluid0d'

    ob.solid = types.SimpleNamespace(
        ncalc=1000,
        dr_min=300,
        dr_max=3000,
        core='liquid',
        core_props='core',
        inertial_terms=True,
        bulk_l=1e9,
        porosity_thresh=3e-2,
        dbulk_power=0.5,
    )
    ob.mushy = types.SimpleNamespace(b_width=0.5, t_width=0.03)
    ob.fluid = types.SimpleNamespace(
        sigma_R=1e-3,
        sigma_R_inf=0.5,
        sigma_R_prf='exp',
        H_R=1e4,
        efficiency=0.3,
    )
    cfg.orbit.obliqua = ob

    cfg.interior_energetics = types.SimpleNamespace()
    cfg.interior_energetics.module = module
    cfg.interior_energetics.grain_size = 1e-3
    cfg.interior_energetics.boundary = types.SimpleNamespace(
        core_density=1e4,
        core_shear=8e10,
        core_bulk=1.4e11,
    )
    return cfg


def _make_fake_jl(power_prf, power_blk, nmk, sigma, lnk):
    """Fake ``jl`` with the ``Obliqua`` namespace ``run_obliqua`` calls
    directly: ``interior.get_permeability/limit_porosity/get_drained_bulk``
    and ``run_tides``.
    """
    fake_jl = MagicMock(name='jl')
    fake_jl.Obliqua.interior.get_permeability = MagicMock(return_value='perm')
    fake_jl.Obliqua.interior.limit_porosity = MagicMock(
        return_value=('perm_limited', 'phi_limited')
    )
    fake_jl.Obliqua.interior.get_drained_bulk = MagicMock(return_value='bulkd')
    fake_jl.Obliqua.run_tides = MagicMock(return_value=(power_prf, power_blk, nmk, sigma, lnk))
    return fake_jl


def _write_lookup_netcdf(path, nmk_rows, sigma, lnk):
    """Write a minimal real netCDF lookup file in the schema
    ``Tides_t.add_from_file`` / ``lookup_from_interior`` use: integer
    ``n``/``m``/``k`` mode-index variables plus float ``sigma`` and
    ``LNk_real``/``LNk_imag``, all on a single ``mode`` dimension.
    """
    nmk_rows = np.asarray(nmk_rows, dtype=np.int64)
    sigma = np.asarray(sigma, dtype=np.float64)
    lnk = np.asarray(lnk, dtype=np.complex128)
    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('mode', len(sigma))
        ds.createVariable('n', 'i4', ('mode',))[:] = nmk_rows[:, 0]
        ds.createVariable('m', 'i4', ('mode',))[:] = nmk_rows[:, 1]
        ds.createVariable('k', 'i4', ('mode',))[:] = nmk_rows[:, 2]
        ds.createVariable('sigma', 'f8', ('mode',))[:] = sigma
        ds.createVariable('LNk_real', 'f8', ('mode',))[:] = np.real(lnk)
        ds.createVariable('LNk_imag', 'f8', ('mode',))[:] = np.imag(lnk)


def _seed_satellite_dict_cache(tides_o: Tides_t, nmk_rows, sigma, lnk):
    """Populate the ``('satellite_dict', 'planet')`` cache entry
    directly (bypassing file I/O), matching what
    ``Tides_t.add_from_file`` would have produced.
    """
    entry = tides_o.add(primary='satellite_dict', perturber='planet')
    entry.nmk = np.asarray(nmk_rows, dtype=int)
    entry.sigma = np.asarray(sigma, dtype=float)
    entry.LNk = np.asarray(lnk, dtype=complex)
    return entry


def _seed_planet_modes(tides_o: Tides_t, nmk_rows):
    """Populate the ``('planet', 'satellite')`` mode table that
    ``LN_from_lookup`` reads as its starting point (normally written
    earlier by ``run_obliqua``'s satellite-perturber branch).
    """
    entry = tides_o.add(primary='planet', perturber='satellite')
    entry.nmk = np.asarray(nmk_rows, dtype=int)
    return entry


def _make_satellite_config(love_number_sat):
    """Fake Config namespace exposing only
    ``config.orbit.satellite.love_number_sat``, the single field
    ``lookup_from_interior``/``LN_from_lookup`` read from ``config``
    directly (the rest comes from ``config.orbit.obliqua``, covered by
    ``_make_config`` above for the tests that also call ``run_tides``).
    """
    cfg = _make_config(module='aragog', perturber='satellite')
    cfg.orbit.satellite = types.SimpleNamespace(love_number_sat=love_number_sat)
    return cfg


def _patch_identity_conversions(monkeypatch, obliqua_mod):
    """Patch the Julia-conversion helpers to identity so run_obliqua
    tests can inspect plain numpy/Python values in call args, and
    isolate the orchestration logic from the dedicated conversion
    tests below.
    """
    monkeypatch.setattr(obliqua_mod, '_jlarr', lambda a: np.asarray(a))
    monkeypatch.setattr(obliqua_mod, '_jlsca_float', lambda s: s)
    monkeypatch.setattr(obliqua_mod, '_jlsca_prec', lambda s: s)
    monkeypatch.setattr(obliqua_mod, 'to_julia_dict', lambda cfg: cfg)
    monkeypatch.setattr(obliqua_mod, 'sync_log_files', lambda outdir: [])


# ---------------------------------------------------------------------------
# import_obliqua.
# ---------------------------------------------------------------------------


def test_import_obliqua_calls_jl_seval_with_using_obliqua(monkeypatch):
    """``import_obliqua`` issues a single ``jl.seval('using Obliqua')``
    call. A regression that dropped or misspelled the import string
    would silently leave the ``Obliqua`` Julia symbols unbound.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_jl = MagicMock(name='jl')
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)
    obliqua_mod.import_obliqua()
    fake_jl.seval.assert_called_once_with('using Obliqua')
    # Discrimination: exactly one import call, not zero or repeated.
    assert fake_jl.seval.call_count == 1


# ---------------------------------------------------------------------------
# to_julia_dict: recursive conversion.
# ---------------------------------------------------------------------------


def test_to_julia_dict_recursively_converts_nested_dict_and_list(monkeypatch):
    """``to_julia_dict`` recurses into nested dicts and lists,
    converting every dict level via ``jl.Dict()`` while leaving
    scalars untouched. Standing in the real Python ``dict`` for
    ``jl.Dict`` makes the recursion observable directly: a regression
    that stopped recursing into list elements, or that converted a
    list itself into a Julia object instead of mapping over it, would
    change the returned structure.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_jl = types.SimpleNamespace(Dict=dict)
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    nested = {
        'a': 1.0,
        'b': [1, {'c': 2.0}, 3],
        'd': {'e': {'f': 'leaf'}},
    }
    out = obliqua_mod.to_julia_dict(nested)

    assert out == nested
    # Discrimination: an empty-list edge case must round-trip to an
    # empty list, not be dropped or replaced with None.
    assert obliqua_mod.to_julia_dict({'empty': []}) == {'empty': []}


# ---------------------------------------------------------------------------
# _jlarr / _jlsca_float / _jlsca_prec: julia type conversion contract.
# ---------------------------------------------------------------------------


def test_jlarr_flattens_and_converts_without_reordering(monkeypatch):
    """``_jlarr`` flattens the numpy array and converts it via
    ``juliacall.convert`` targeting ``jl.Array[jl.Obliqua.prec, 1]``.

    Pins the *current* implementation: despite the inline source
    comment claiming the array is reversed ("Make copy of array,
    reverse order..."), the implementation does not reverse element
    order. An asymmetric input (strictly increasing, not a palindrome)
    is used so this test would fail if reversal were silently added
    or removed.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_juliacall = MagicMock(name='juliacall')
    fake_juliacall.convert = MagicMock(return_value='converted_array')
    fake_jl = MagicMock(name='jl')
    fake_jl.Array = MagicMock()
    fake_jl.Obliqua = MagicMock()
    fake_jl.Obliqua.prec = 'prec_sentinel'
    fake_jl.Array.__getitem__ = MagicMock(return_value='destination_type')
    monkeypatch.setattr(obliqua_mod, 'juliacall', fake_juliacall)
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    arr = np.array([1.0, 2.0, 3.0])  # asymmetric: catches accidental reversal
    out = obliqua_mod._jlarr(arr)

    fake_jl.Array.__getitem__.assert_called_once_with(('prec_sentinel', 1))
    fake_juliacall.convert.assert_called_once()
    call_args, _ = fake_juliacall.convert.call_args
    assert call_args[0] == 'destination_type'
    np.testing.assert_array_equal(call_args[1], arr)
    assert out == 'converted_array'


def test_jlsca_float_converts_to_julia_float64_type(monkeypatch):
    """``_jlsca_float`` converts via
    ``juliacall.convert(jl.Obliqua.Float64, sca)``. Pins the
    destination-type argument so a regression that swapped in the
    ``prec`` type (used by ``_jlsca_prec`` instead) would surface.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_juliacall = MagicMock(name='juliacall')
    fake_juliacall.convert = MagicMock(return_value='converted_float64')
    fake_jl = MagicMock(name='jl')
    fake_jl.Obliqua = MagicMock()
    fake_jl.Obliqua.Float64 = 'float64_sentinel'
    monkeypatch.setattr(obliqua_mod, 'juliacall', fake_juliacall)
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    out = obliqua_mod._jlsca_float(0.5)
    fake_juliacall.convert.assert_called_once_with('float64_sentinel', 0.5)
    assert out == 'converted_float64'


def test_jlsca_prec_converts_to_julia_prec_type(monkeypatch):
    """``_jlsca_prec`` converts via
    ``juliacall.convert(jl.Obliqua.prec, sca)`` -- a distinct
    destination type from ``_jlsca_float``. Pinning both destination
    sentinels separately discriminates a regression that merged or
    swapped the two conversion helpers.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_juliacall = MagicMock(name='juliacall')
    fake_juliacall.convert = MagicMock(return_value='converted_prec')
    fake_jl = MagicMock(name='jl')
    fake_jl.Obliqua = MagicMock()
    fake_jl.Obliqua.prec = 'prec_sentinel'
    monkeypatch.setattr(obliqua_mod, 'juliacall', fake_juliacall)
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    out = obliqua_mod._jlsca_prec(0.5)
    fake_juliacall.convert.assert_called_once_with('prec_sentinel', 0.5)
    assert out == 'converted_prec'


# ---------------------------------------------------------------------------
# run_obliqua: perturber dispatch (star vs. satellite orbital state).
# ---------------------------------------------------------------------------


def test_run_obliqua_star_perturber_reads_star_orbital_state(monkeypatch, tmp_path):
    """Under ``config.orbit.perturber == 'star'``, ``run_obliqua``
    reads the star-planet orbital state (``orbital_period``,
    ``eccentricity``, ``semimajorax``, ``M_star``), not the satellite
    fields. Discrimination: pin that ``omega`` derives from
    ``orbital_period`` (not ``axial_period``, which feeds the
    separate ``axial`` argument), and that ``M_pert`` is ``M_star``.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    nlev_s = 3
    interior_o = _make_interior_t(nlev_s)
    cfg = _make_config(module='dummy', perturber='star')

    hf_row = {
        'Time': 100.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.1,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    power_prf = np.array([0.0, 5e-7])
    fake_jl = _make_fake_jl(
        power_prf=power_prf,
        power_blk=1.0,
        nmk=[(2, 0, 1), (2, 2, 3)],
        sigma=[1e-6, 2e-6],
        lnk=[0.01 - 0.02j, 0.03 - 0.04j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    fake_jl.Obliqua.run_tides.assert_called_once()
    call_args = fake_jl.Obliqua.run_tides.call_args[0]
    omega, axial, ecc, sma, m_pert = call_args[:5]

    assert omega == pytest.approx(2 * np.pi / hf_row['orbital_period'], rel=1e-12)
    # Discrimination: omega must not have been derived from axial_period.
    assert omega != pytest.approx(2 * np.pi / hf_row['axial_period'], rel=1e-6)
    assert axial == pytest.approx(2 * np.pi / hf_row['axial_period'], rel=1e-12)
    assert ecc == pytest.approx(0.1, rel=1e-12)
    assert sma == pytest.approx(1.5e11, rel=1e-12)
    assert m_pert == pytest.approx(2.0e30, rel=1e-12)


def test_run_obliqua_satellite_perturber_reads_satellite_orbital_state(monkeypatch, tmp_path):
    """Under ``config.orbit.perturber == 'satellite'``, ``run_obliqua``
    reads the satellite-suffixed fields
    (``orbital_period_sat``/``eccentricity_sat``/``semimajorax_sat``/
    ``M_sat``) instead of the star fields. Edge case: eccentricity is
    exercised at the boundary value 0.0 (circular orbit).
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    nlev_s = 3
    interior_o = _make_interior_t(nlev_s)
    cfg = _make_config(module='dummy', perturber='satellite')

    hf_row = {
        'Time': 50.0,
        'axial_period': 86400.0,
        'orbital_period_sat': 86400.0 * 27.3,
        'eccentricity_sat': 0.0,
        'semimajorax_sat': 3.84e8,
        'M_sat': 7.3e22,
    }

    power_prf = np.array([0.0, 2e-8])
    fake_jl = _make_fake_jl(
        power_prf=power_prf,
        power_blk=1.0,
        nmk=[(2, 0, 1)],
        sigma=[1e-7],
        lnk=[0.005 - 0.001j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    call_args = fake_jl.Obliqua.run_tides.call_args[0]
    omega, _axial, ecc, sma, m_pert = call_args[:5]

    assert ecc == pytest.approx(0.0, abs=1e-15)
    assert omega == pytest.approx(2 * np.pi / hf_row['orbital_period_sat'], rel=1e-12)
    assert sma == pytest.approx(3.84e8, rel=1e-12)
    assert m_pert == pytest.approx(7.3e22, rel=1e-12)


# ---------------------------------------------------------------------------
# run_obliqua: interior-module branching (dummy / spider / aragog-like).
# ---------------------------------------------------------------------------


def test_run_obliqua_dummy_interior_writes_single_tide_from_second_profile_entry(
    monkeypatch, tmp_path
):
    """Under ``interior_energetics.module == 'dummy'``, the two-cell
    profile hack writes ``interior_o.tides[0] = power_prf[1]``
    (not ``power_prf[0]``). The asymmetric mock profile below
    discriminates an off-by-one index regression.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    interior_o = _make_interior_t(nlev_s=1)
    cfg = _make_config(module='dummy', perturber='star')
    hf_row = {
        'Time': 0.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.05,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    power_prf = np.array([1e-9, 7e-7])  # [0]=1e-9, [1]=7e-7: distinguishable
    fake_jl = _make_fake_jl(
        power_prf=power_prf,
        power_blk=1.0,
        nmk=[(2, 0, 1)],
        sigma=[1e-6],
        lnk=[0.01 - 0.02j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    assert interior_o.tides[0] == pytest.approx(7e-7, rel=1e-12)
    # Discrimination: not the first profile entry.
    assert interior_o.tides[0] != pytest.approx(1e-9, rel=1e-6)


def test_run_obliqua_spider_interior_reverses_and_conserves_total_power(monkeypatch, tmp_path):
    """Under ``interior_energetics.module == 'spider'``, arrays are
    reversed on the way into Obliqua (so index 0 sits at the CMB) and
    the returned profile is reversed again on write-back, mirroring
    the LovePy SPIDER convention.

    Physics invariant: reversal is a permutation, so the total tidal
    power dissipated (``sum(interior_o.tides)``) must equal
    ``sum(power_prf)`` exactly -- a regression that instead dropped or
    duplicated an entry during the flip would break this sum even
    though it might still "look like" the right values individually.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    nlev_s = 4
    interior_o = _make_interior_t(nlev_s)
    cfg = _make_config(module='spider', perturber='star')
    hf_row = {
        'Time': 0.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.1,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    power_prf = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    fake_jl = _make_fake_jl(
        power_prf=power_prf,
        power_blk=1.0,
        nmk=[(2, 0, 1)],
        sigma=[1e-6],
        lnk=[0.02 - 0.03j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    # Input side: rho/radius must have been reversed before being passed to
    # run_tides (call_args[0][5] is rho, [0][6] is radius -- see the
    # positional run_tides(...) call in obliqua.py), not just the output.
    call_args = fake_jl.Obliqua.run_tides.call_args[0]
    rho_passed = call_args[5]
    np.testing.assert_allclose(rho_passed, interior_o.density[::-1], rtol=1e-12)
    # Discrimination: input was not left in surface-first order.
    assert not np.allclose(rho_passed, interior_o.density)

    np.testing.assert_allclose(interior_o.tides, power_prf[::-1], rtol=1e-12)
    assert np.sum(interior_o.tides) == pytest.approx(np.sum(power_prf), rel=1e-12)


def test_run_obliqua_aragog_like_interior_writes_direct_profile(monkeypatch, tmp_path):
    """Under a non-dummy, non-spider ``interior_energetics.module``
    (e.g. ``'aragog'``), the profile is written directly with no
    reversal: ``interior_o.tides[:] = power_prf[:]``.

    This branch also computes ``power_blk / sum(mass)`` purely for a
    ``log.debug`` line (see module docstring for the testability
    gap); this test only confirms that division executes on a
    realistic non-zero mass array without raising, alongside the
    direct (unflipped) tides write.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    nlev_s = 4
    interior_o = _make_interior_t(nlev_s)
    cfg = _make_config(module='aragog', perturber='star')
    hf_row = {
        'Time': 0.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.1,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    power_prf = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    fake_jl = _make_fake_jl(
        power_prf=power_prf,
        power_blk=1e4,
        nmk=[(2, 0, 1)],
        sigma=[1e-6],
        lnk=[0.02 - 0.03j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    np.testing.assert_allclose(interior_o.tides, power_prf, rtol=1e-12)
    # Discrimination: unlike SPIDER, the profile is NOT reversed.
    assert not np.allclose(interior_o.tides, power_prf[::-1])


# ---------------------------------------------------------------------------
# run_obliqua: error handling.
# ---------------------------------------------------------------------------


def test_run_obliqua_julia_error_wrapped_into_runtime_error(monkeypatch, tmp_path):
    """``juliacall.JuliaError`` raised from ``run_tides`` is caught and
    re-raised as ``RuntimeError``. ``UpdateStatusfile`` is called with
    status code 26 before the re-raise, matching the LovePy failure
    contract.
    """
    import juliacall as real_juliacall

    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    interior_o = _make_interior_t(nlev_s=3)
    cfg = _make_config(module='dummy', perturber='star')
    hf_row = {
        'Time': 0.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.1,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    fake_jl = MagicMock(name='jl')
    fake_jl.Obliqua.interior.get_permeability = MagicMock(return_value='perm')
    fake_jl.Obliqua.interior.limit_porosity = MagicMock(return_value=('perm', 'phi'))
    fake_jl.Obliqua.interior.get_drained_bulk = MagicMock(return_value='bulkd')
    fake_jl.Obliqua.run_tides = MagicMock(
        side_effect=real_juliacall.JuliaError('mock Obliqua crash')
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    updates: list[tuple] = []
    monkeypatch.setattr(
        obliqua_mod,
        'UpdateStatusfile',
        lambda dirs, code: updates.append((dirs, code)),
    )

    tides_o = Tides_t()
    with pytest.raises(RuntimeError, match=r'(?i)obliqua'):
        obliqua_mod.run_obliqua(
            hf_row,
            dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
            interior_o=interior_o,
            tides_o=tides_o,
            config=cfg,
        )

    assert len(updates) == 1
    assert updates[0][1] == 26


# ---------------------------------------------------------------------------
# run_obliqua: storage into tides_o.
# ---------------------------------------------------------------------------


def test_run_obliqua_stores_love_spectrum_keyed_by_configured_perturber(monkeypatch, tmp_path):
    """Results are stored in ``tides_o`` under
    ``primary='planet', perturber=config.orbit.perturber``: the
    ``nmk`` mode table (stacked to an integer array), the forcing
    frequencies ``sigma``, and the complex Love numbers ``LNk``. The
    return value is ``mean(imag(LNk))``, pinned with a sign
    discrimination guard (the real part carries no dissipative
    information under this convention).
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    interior_o = _make_interior_t(nlev_s=3)
    cfg = _make_config(module='dummy', perturber='star')
    hf_row = {
        'Time': 0.0,
        'axial_period': 86400.0,
        'orbital_period': 86400.0 * 365.0,
        'eccentricity': 0.1,
        'semimajorax': 1.5e11,
        'M_star': 2.0e30,
    }

    nmk = [(2, 0, 1), (2, 2, 3)]
    sigma = [1e-6, 2e-6]
    lnk = np.array([0.01 - 0.02j, 0.03 - 0.06j])
    fake_jl = _make_fake_jl(
        power_prf=np.array([0.0, 5e-7]),
        power_blk=1.0,
        nmk=nmk,
        sigma=sigma,
        lnk=lnk,
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    tides_o = Tides_t()
    out = obliqua_mod.run_obliqua(
        hf_row,
        dirs={'output/data': str(tmp_path), 'output': str(tmp_path)},
        interior_o=interior_o,
        tides_o=tides_o,
        config=cfg,
    )

    storage = tides_o.get(primary='planet', perturber='star')
    np.testing.assert_array_equal(storage.nmk, np.array(nmk, dtype=int))
    np.testing.assert_allclose(storage.sigma, sigma, rtol=1e-12)
    np.testing.assert_allclose(storage.LNk, lnk, rtol=1e-12)

    expected = np.mean(np.imag(lnk))
    assert out == pytest.approx(expected, rel=1e-12)
    # Sign discrimination: this Im(k2) convention is negative here; a
    # regression that returned mean(real(LNk)) instead would be
    # positive and would fail this sign check.
    assert out < 0.0


# ---------------------------------------------------------------------------
# lookup_from_interior: JSON IC -> netCDF lookup table.
# ---------------------------------------------------------------------------


def _write_interior_json(path, density, radius, visc, shear, bulk, phi):
    payload = {
        'omega': 1e-6,
        'axial': 7e-5,
        'ecc': 0.05,
        'sma': 3.8e8,
        'S_mass': 7.3e22,
        'density': density,
        'radius': radius,
        'visc': visc,
        'shear': shear,
        'bulk': bulk,
        'phi': phi,
    }
    with open(path, 'w') as f:
        json.dump(payload, f)


def test_lookup_from_interior_raises_when_love_number_path_unset(tmp_path):
    """``lookup_from_interior`` raises ``ValueError`` immediately when
    ``config.orbit.satellite.love_number_sat`` is unset, before
    touching any file or the Julia boundary. Edge case: empty string
    is treated the same as ``None`` (both are falsy).
    """
    from proteus.orbit import obliqua as obliqua_mod

    cfg = _make_satellite_config(love_number_sat=None)
    with pytest.raises(ValueError, match=r'love_number_sat'):
        obliqua_mod.lookup_from_interior(dirs={'output/data': str(tmp_path)}, config=cfg)

    cfg_empty = _make_satellite_config(love_number_sat='')
    with pytest.raises(ValueError, match=r'love_number_sat'):
        obliqua_mod.lookup_from_interior(dirs={'output/data': str(tmp_path)}, config=cfg_empty)


def test_lookup_from_interior_splits_core_density_from_mantle_profile(monkeypatch, tmp_path):
    """The first entry of the JSON ``density`` array is extracted as
    ``core_density`` (used only in ``cfg['struct']['core_density']``)
    and excluded from ``rho`` -- the array actually passed to
    ``run_tides`` covers the mantle only. An asymmetric density
    profile discriminates a regression that passed the full array
    (including the core) as ``rho``, or dropped the wrong end.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    json_path = tmp_path / 'interior.json'
    _write_interior_json(
        json_path,
        density=[8000.0, 4000.0, 4200.0, 4500.0],  # [0]=core, [1:]=mantle
        radius=[3.0e6, 3.8e6, 4.6e6, 5.4e6],
        visc=[1e20, 1e19, 1e18],
        shear=[6e10, 5e10, 4e10],
        bulk=[2e11, 1.9e11, 1.8e11],
        phi=[0.0, 0.0, 0.05],
    )
    cfg = _make_satellite_config(love_number_sat=str(json_path))

    fake_jl = _make_fake_jl(
        power_prf=np.array([1e-6]),
        power_blk=1.0,
        nmk=[(2, 0, 1)],
        sigma=[1e-6],
        lnk=[0.01 - 0.02j],
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    obliqua_mod.lookup_from_interior(dirs={'output/data': str(tmp_path)}, config=cfg)

    call_args = fake_jl.Obliqua.run_tides.call_args[0]
    rho_passed, cfg_passed = call_args[5], call_args[13]

    np.testing.assert_allclose(rho_passed, [4000.0, 4200.0, 4500.0], rtol=1e-12)
    # Discrimination: the core entry must not leak into rho.
    assert not np.any(np.isclose(rho_passed, 8000.0))
    assert cfg_passed['struct']['core_density'] == pytest.approx(8000.0, rel=1e-12)


def test_lookup_from_interior_writes_netcdf_matching_run_tides_output(monkeypatch, tmp_path):
    """The written ``moon_tides.nc`` lookup file exactly reproduces
    the ``(nmk, sigma, LNk)`` triple returned by ``run_tides`` --
    a round trip through real netCDF I/O (not mocked), pinned against
    an asymmetric two-mode result so a column swap (e.g. writing ``m``
    into the ``k`` variable) would be caught.
    """
    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    json_path = tmp_path / 'interior.json'
    _write_interior_json(
        json_path,
        density=[8000.0, 4000.0],
        radius=[3.0e6, 4.0e6],
        visc=[1e20],
        shear=[6e10],
        bulk=[2e11],
        phi=[0.0],
    )
    cfg = _make_satellite_config(love_number_sat=str(json_path))

    nmk = [(2, 0, 1), (2, 2, 3)]
    sigma = [1e-6, 2e-6]
    lnk = [0.01 - 0.02j, 0.03 - 0.04j]
    fake_jl = _make_fake_jl(
        power_prf=np.array([1e-6]), power_blk=1.0, nmk=nmk, sigma=sigma, lnk=lnk
    )
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    obliqua_mod.lookup_from_interior(dirs={'output/data': str(tmp_path)}, config=cfg)

    out = obliqua_mod.read_ncdf(str(tmp_path / 'moon_tides.nc'))
    np.testing.assert_array_equal(out['n'], [2, 2])
    np.testing.assert_array_equal(out['m'], [0, 2])
    np.testing.assert_array_equal(out['k'], [1, 3])
    np.testing.assert_allclose(out['sigma'], sigma, rtol=1e-12)
    np.testing.assert_allclose(out['LNk_real'], np.real(lnk), rtol=1e-12)
    np.testing.assert_allclose(out['LNk_imag'], np.imag(lnk), rtol=1e-12)


def test_lookup_from_interior_julia_error_wrapped_into_runtime_error(monkeypatch, tmp_path):
    """``juliacall.JuliaError`` raised from ``run_tides`` during
    lookup-table generation is caught and re-raised as
    ``RuntimeError``, with ``UpdateStatusfile`` called with code 26,
    matching the ``run_obliqua`` failure contract.
    """
    import juliacall as real_juliacall

    from proteus.orbit import obliqua as obliqua_mod

    _patch_identity_conversions(monkeypatch, obliqua_mod)

    json_path = tmp_path / 'interior.json'
    _write_interior_json(
        json_path,
        density=[8000.0, 4000.0],
        radius=[3.0e6, 4.0e6],
        visc=[1e20],
        shear=[6e10],
        bulk=[2e11],
        phi=[0.0],
    )
    cfg = _make_satellite_config(love_number_sat=str(json_path))

    fake_jl = MagicMock(name='jl')
    fake_jl.Obliqua.interior.get_permeability = MagicMock(return_value='perm')
    fake_jl.Obliqua.interior.limit_porosity = MagicMock(return_value=('perm', 'phi'))
    fake_jl.Obliqua.interior.get_drained_bulk = MagicMock(return_value='bulkd')
    fake_jl.Obliqua.run_tides = MagicMock(side_effect=real_juliacall.JuliaError('mock crash'))
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    updates: list[tuple] = []
    monkeypatch.setattr(
        obliqua_mod, 'UpdateStatusfile', lambda dirs, code: updates.append((dirs, code))
    )

    with pytest.raises(RuntimeError, match=r'(?i)obliqua'):
        obliqua_mod.lookup_from_interior(dirs={'output/data': str(tmp_path)}, config=cfg)

    assert len(updates) == 1
    assert updates[0][1] == 26


# ---------------------------------------------------------------------------
# LN_from_lookup: pure NumPy/SciPy post-processing, no Julia boundary.
# ---------------------------------------------------------------------------


def _default_lookup_table():
    """Degree-2 lookup table used by most ``LN_from_lookup`` tests:
    three positive forcing frequencies with distinct, asymmetric
    complex Love numbers so interpolation/symmetry bugs are visible.
    """
    nmk_lookup = [(2, 0, 1), (2, 0, 2), (2, 0, 3)]
    sigma_lookup = [1e-6, 2e-6, 3e-6]
    lnk_lookup = [0.01 - 0.02j, 0.02 - 0.03j, 0.03 - 0.05j]
    return nmk_lookup, sigma_lookup, lnk_lookup


@pytest.mark.physics_invariant
def test_ln_from_lookup_interpolates_exact_value_at_lookup_node():
    """At a forcing frequency that lands exactly on a tabulated
    ``sigma`` node, linear interpolation must reproduce that node's
    Love number exactly (up to float rounding). ``m=2, k=0`` with
    ``axial_freq_s = 1e-6`` puts ``sigma_s`` exactly on the table's
    second node.
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, 2, 0)])
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    _seed_satellite_dict_cache(tides_o, nmk_lookup, sigma_lookup, lnk_lookup)

    hf_row = {
        'axial_period_sat': 2 * np.pi / 1e-6,
        'orbital_period_sat': 86400.0 * 27.3,
    }
    cfg = _make_satellite_config(love_number_sat='unused.nc')

    obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)

    storage = tides_o.get(primary='satellite', perturber='planet')
    assert storage.LNk[0] == pytest.approx(0.02 - 0.03j, rel=1e-9)
    # Discrimination: not a neighboring node's value.
    assert abs(storage.LNk[0] - (0.01 - 0.02j)) > 1e-4
    assert abs(storage.LNk[0] - (0.03 - 0.05j)) > 1e-4


@pytest.mark.physics_invariant
def test_ln_from_lookup_enforces_love_number_reality_symmetry():
    """Physical invariant: the tidal Love number is the frequency
    response of a real-valued physical system, so it must satisfy
    the reality/symmetry condition ``LNk(-sigma) == conj(LNk(sigma))``.
    Evaluating at ``m=-2, k=0`` (sigma_s = -2e-6, the negative of the
    node used in the previous test) must return the complex conjugate
    of that node's Love number, not the same value or its negative.
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, -2, 0)])
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    _seed_satellite_dict_cache(tides_o, nmk_lookup, sigma_lookup, lnk_lookup)

    hf_row = {
        'axial_period_sat': 2 * np.pi / 1e-6,
        'orbital_period_sat': 86400.0 * 27.3,
    }
    cfg = _make_satellite_config(love_number_sat='unused.nc')

    obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)

    storage = tides_o.get(primary='satellite', perturber='planet')
    expected = np.conj(0.02 - 0.03j)
    assert storage.LNk[0] == pytest.approx(expected, rel=1e-9)
    # Discrimination: not the un-conjugated value (sign of imaginary
    # part flipped relative to a same-sigma, no-symmetry bug).
    assert storage.LNk[0].imag > 0.0


def test_ln_from_lookup_raises_for_degree_missing_from_lookup_table():
    """A planet-side mode at a tidal degree absent from the lookup
    table raises ``ValueError`` naming the missing degree, rather
    than silently returning zero or extrapolating across degrees
    (Love numbers are not comparable across different ``n``).
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(3, 2, 0)])  # degree 3: not in the lookup
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()  # degree 2 only
    _seed_satellite_dict_cache(tides_o, nmk_lookup, sigma_lookup, lnk_lookup)

    hf_row = {'axial_period_sat': 2 * np.pi / 1e-6, 'orbital_period_sat': 86400.0 * 27.3}
    cfg = _make_satellite_config(love_number_sat='unused.nc')

    with pytest.raises(ValueError, match=r'degree n = 3'):
        obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)


def test_ln_from_lookup_zeroes_only_the_negative_m_and_k_modes():
    """The ``(m < 0) & (k < 0)`` convention zeroes exactly the modes
    with both indices negative; a sibling mode with the same degree
    but non-negative ``k`` in the same call is interpolated normally,
    discriminating a regression that zeroed every mode (or none).
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, -1, -3), (2, 2, 0)])
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    _seed_satellite_dict_cache(tides_o, nmk_lookup, sigma_lookup, lnk_lookup)

    hf_row = {
        'axial_period_sat': 2 * np.pi / 1e-6,
        'orbital_period_sat': 86400.0 * 27.3,
    }
    cfg = _make_satellite_config(love_number_sat='unused.nc')

    obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)

    storage = tides_o.get(primary='satellite', perturber='planet')
    assert storage.LNk[0] == pytest.approx(0.0 + 0.0j, abs=1e-15)
    # The sibling mode (m=2, k=0) lands on the same exact node as in
    # the first interpolation test and must NOT be zeroed.
    assert storage.LNk[1] == pytest.approx(0.02 - 0.03j, rel=1e-9)


@pytest.mark.physics_invariant
def test_ln_from_lookup_forcing_frequency_matches_m_axial_minus_k_orbital():
    """``sigma_s = m * axial_freq_s - k * orbital_freq_s``, pinned
    with sign and column-order discrimination guards: a formula that
    swapped ``m``/``k`` or used ``+`` instead of ``-`` would land on a
    different value than the one asserted here.
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, 3, 5)])
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    _seed_satellite_dict_cache(tides_o, nmk_lookup, sigma_lookup, lnk_lookup)

    axial_freq_s = 1e-6
    orbit_freq_s = 4e-7
    hf_row = {
        'axial_period_sat': 2 * np.pi / axial_freq_s,
        'orbital_period_sat': 2 * np.pi / orbit_freq_s,
    }
    cfg = _make_satellite_config(love_number_sat='unused.nc')

    obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)

    storage = tides_o.get(primary='satellite', perturber='planet')
    expected = 3 * axial_freq_s - 5 * orbit_freq_s  # = 1e-6
    assert storage.sigma[0] == pytest.approx(expected, rel=1e-12)
    # Discrimination: m/k swapped (5*axial - 3*orbit = 3.8e-6).
    assert storage.sigma[0] != pytest.approx(5 * axial_freq_s - 3 * orbit_freq_s, rel=1e-6)
    # Discrimination: sign flipped to '+' (3*axial + 5*orbit = 5e-6).
    assert storage.sigma[0] != pytest.approx(3 * axial_freq_s + 5 * orbit_freq_s, rel=1e-6)


def test_ln_from_lookup_generates_lookup_once_from_json_path_and_caches_it(
    monkeypatch, tmp_path
):
    """When ``love_number_sat`` points at a ``.json`` IC file and no
    lookup is cached yet, ``LN_from_lookup`` calls
    ``lookup_from_interior`` once to generate ``moon_tides.nc``, then
    caches the loaded table under ``('satellite_dict', 'planet')``. A
    second call on the same ``tides_o`` must reuse the cache rather
    than regenerating it.
    """
    from proteus.orbit import obliqua as obliqua_mod

    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    nc_path = tmp_path / 'moon_tides.nc'

    calls: list[tuple] = []

    def fake_lookup_from_interior(dirs, config):
        calls.append((dirs, config))
        _write_lookup_netcdf(nc_path, nmk_lookup, sigma_lookup, lnk_lookup)

    monkeypatch.setattr(obliqua_mod, 'lookup_from_interior', fake_lookup_from_interior)

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, 2, 0)])
    cfg = _make_satellite_config(love_number_sat=str(tmp_path / 'interior_source.json'))
    hf_row = {'axial_period_sat': 2 * np.pi / 1e-6, 'orbital_period_sat': 86400.0 * 27.3}
    dirs = {'output/data': str(tmp_path)}

    obliqua_mod.LN_from_lookup(hf_row, dirs=dirs, tides_o=tides_o, config=cfg)
    assert len(calls) == 1
    storage = tides_o.get(primary='satellite', perturber='planet')
    assert storage.LNk[0] == pytest.approx(0.02 - 0.03j, rel=1e-9)

    # Second call: cache already populated, must not regenerate.
    _seed_planet_modes(tides_o, [(2, 2, 0)])
    obliqua_mod.LN_from_lookup(hf_row, dirs=dirs, tides_o=tides_o, config=cfg)
    assert len(calls) == 1


def test_ln_from_lookup_loads_nc_path_directly_without_regenerating(monkeypatch, tmp_path):
    """When ``love_number_sat`` already points at a ``.nc`` lookup
    file, ``LN_from_lookup`` must load it directly and must NOT call
    ``lookup_from_interior`` at all (that branch is a no-op ``pass``
    in the source; regenerating on an ``.nc`` path would be wasted
    Julia work at best, or silently overwrite a hand-provided table).
    """
    from proteus.orbit import obliqua as obliqua_mod

    def fail_if_called(dirs, config):
        raise AssertionError('lookup_from_interior must not be called for a .nc path')

    monkeypatch.setattr(obliqua_mod, 'lookup_from_interior', fail_if_called)

    nc_path = tmp_path / 'provided_lookup.nc'
    nmk_lookup, sigma_lookup, lnk_lookup = _default_lookup_table()
    _write_lookup_netcdf(nc_path, nmk_lookup, sigma_lookup, lnk_lookup)

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, 2, 0)])
    cfg = _make_satellite_config(love_number_sat=str(nc_path))
    hf_row = {'axial_period_sat': 2 * np.pi / 1e-6, 'orbital_period_sat': 86400.0 * 27.3}

    obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)

    storage = tides_o.get(primary='satellite', perturber='planet')
    assert storage.LNk[0] == pytest.approx(0.02 - 0.03j, rel=1e-9)


def test_ln_from_lookup_raises_when_path_unset_and_no_cache():
    """With no cached lookup table and no ``love_number_sat`` path,
    ``LN_from_lookup`` raises ``ValueError`` rather than silently
    returning an empty or default Love-number spectrum.
    """
    from proteus.orbit import obliqua as obliqua_mod

    tides_o = Tides_t()
    _seed_planet_modes(tides_o, [(2, 2, 0)])
    cfg = _make_satellite_config(love_number_sat=None)
    hf_row = {'axial_period_sat': 2 * np.pi / 1e-6, 'orbital_period_sat': 86400.0 * 27.3}

    with pytest.raises(ValueError, match=r'love_number_sat'):
        obliqua_mod.LN_from_lookup(hf_row, dirs={}, tides_o=tides_o, config=cfg)


# ---------------------------------------------------------------------------
# read_ncdf / read_ncdfs.
# ---------------------------------------------------------------------------


def test_read_ncdf_returns_all_variables_as_a_dict(tmp_path):
    """``read_ncdf`` returns every variable in the file as a dict
    entry, keyed by variable name, values matching exactly. Uses
    asymmetric non-repeating values so a column/key mixup is visible.
    """
    from proteus.orbit import obliqua as obliqua_mod

    nc_path = tmp_path / 'lookup.nc'
    nmk = [(2, 0, 1), (3, 1, 2)]
    sigma = [1e-6, 5e-6]
    lnk = [0.01 - 0.02j, 0.07 - 0.11j]
    _write_lookup_netcdf(nc_path, nmk, sigma, lnk)

    out = obliqua_mod.read_ncdf(str(nc_path))

    assert set(out.keys()) == {'n', 'm', 'k', 'sigma', 'LNk_real', 'LNk_imag'}
    np.testing.assert_array_equal(out['n'], [2, 3])
    np.testing.assert_array_equal(out['m'], [0, 1])
    np.testing.assert_array_equal(out['k'], [1, 2])
    np.testing.assert_allclose(out['sigma'], sigma, rtol=1e-12)
    np.testing.assert_allclose(out['LNk_real'], np.real(lnk), rtol=1e-12)
    np.testing.assert_allclose(out['LNk_imag'], np.imag(lnk), rtol=1e-12)


def test_read_ncdfs_orders_output_by_requested_times_not_filesystem_order(tmp_path):
    """``read_ncdfs`` returns files in the order given by ``times``,
    not filesystem/lexical-name order. ``times=[2, 10]`` is chosen so
    lexical filename order ('10_obliqua.nc' < '2_obliqua.nc', since
    '1' < '2') is the *opposite* of the requested numeric order: a
    regression that globbed and sorted filenames as strings instead
    of indexing by the given ``times`` list would return the t=10
    entry first and the t=2 entry second -- the reverse of what this
    test asserts.
    """
    from proteus.orbit import obliqua as obliqua_mod

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    _write_lookup_netcdf(data_dir / '2_obliqua.nc', [(2, 0, 1)], [1e-6], [0.01 - 0.02j])
    _write_lookup_netcdf(data_dir / '10_obliqua.nc', [(2, 0, 1)], [1e-6], [0.09 - 0.09j])

    out = obliqua_mod.read_ncdfs(str(tmp_path), times=[2, 10])

    assert len(out) == 2
    np.testing.assert_allclose(out[0]['LNk_real'], [0.01], rtol=1e-12)  # t=2
    np.testing.assert_allclose(out[1]['LNk_real'], [0.09], rtol=1e-12)  # t=10


# ---------------------------------------------------------------------------
# setup_logging.
# ---------------------------------------------------------------------------


def test_setup_logging_calls_jl_with_joined_path_and_verbosity(monkeypatch, tmp_path):
    """``setup_logging`` calls ``jl.Obliqua.setup_logging`` with the
    joined ``dirs['output']``/``Obliqua_LOGFILE_NAME`` path and the
    verbosity value passed straight through. A non-default verbosity
    (2) discriminates a regression that hardcoded a default instead
    of forwarding the argument.
    """
    from proteus.orbit import obliqua as obliqua_mod

    fake_jl = MagicMock(name='jl')
    monkeypatch.setattr(obliqua_mod, 'jl', fake_jl)

    obliqua_mod.setup_logging(dirs={'output': str(tmp_path)}, verbosity=2)

    expected_path = os.path.join(str(tmp_path), obliqua_mod.Obliqua_LOGFILE_NAME)
    fake_jl.Obliqua.setup_logging.assert_called_once_with(expected_path, 2)


# ---------------------------------------------------------------------------
# sync_log_files.
# ---------------------------------------------------------------------------


def test_sync_log_files_returns_empty_list_when_obliqua_logfile_missing(tmp_path):
    """When the Obliqua logfile does not exist yet (e.g. before the
    first tidal-heating call of a run), ``sync_log_files`` catches the
    resulting ``OSError`` and returns an empty list rather than
    raising, and does not touch the destination PROTEUS logfile.
    """
    from proteus.orbit import obliqua as obliqua_mod

    proteus_log = tmp_path / 'proteus_00.log'
    proteus_log.write_text('pre-existing content\n')

    assert obliqua_mod.sync_log_files(str(tmp_path)) == []
    # Discrimination: the early OSError return must not partially
    # execute the copy (e.g. opening the destination in append mode
    # before the source read is attempted).
    assert proteus_log.read_text() == 'pre-existing content\n'


def test_sync_log_files_copies_cleaned_text_but_returns_uncleaned_first_line(tmp_path):
    """Pins the observed (not necessarily intended) contract: the
    on-disk PROTEUS logfile receives the first line with its leading
    NULL-character prefix stripped (any text before the first ``[``
    is dropped), but the list this function *returns* is the
    original, uncleaned lines -- see the module docstring's
    "Known testability gap" note. This test documents that mismatch
    rather than silently assuming the return value is safe to scan
    for failure markers.
    """
    from proteus.orbit import obliqua as obliqua_mod

    (tmp_path / 'proteus_00.log').write_text('')  # so GetCurrentLogfileIndex resolves to 0
    obliqua_log = tmp_path / obliqua_mod.Obliqua_LOGFILE_NAME
    raw_first_line = '\x00\x00[2024-01-01] line one\n'
    obliqua_log.write_text(raw_first_line + 'line two\n')

    out = obliqua_mod.sync_log_files(str(tmp_path))

    proteus_log_text = (tmp_path / 'proteus_00.log').read_text()
    assert proteus_log_text == '[2024-01-01] line one\nline two\n'
    # Obliqua's own logfile is cleared after the copy.
    assert obliqua_log.read_text() == ''
    # Pinned discrepancy: the returned lines still carry the raw,
    # uncleaned first line, not the cleaned text written above.
    assert out[0] == raw_first_line
    assert out[1] == 'line two\n'


def test_sync_log_files_copies_first_line_unchanged_when_no_bracket_present(tmp_path):
    """Edge case: if the first line contains no ``[`` at all, the
    NULL-stripping branch is not taken and the line is copied
    through unchanged (not truncated or dropped), and the return
    value matches the same unchanged text (unlike the NULL-prefixed
    case, there is no cleaning for the return value to omit).
    """
    from proteus.orbit import obliqua as obliqua_mod

    (tmp_path / 'proteus_00.log').write_text('')
    obliqua_log = tmp_path / obliqua_mod.Obliqua_LOGFILE_NAME
    obliqua_log.write_text('no bracket on this line\n')

    out = obliqua_mod.sync_log_files(str(tmp_path))

    assert (tmp_path / 'proteus_00.log').read_text() == 'no bracket on this line\n'
    assert out == ['no bracket on this line\n']
    # Obliqua's own logfile is cleared after the copy, same as the
    # bracket-present case.
    assert obliqua_log.read_text() == ''
