# Chemistry coupled to the climate: atmos_chem/vulcan_agni.py (SnapshotClimate) and the
# coupled run_vulcan, with the real AGNI on one banded snapshot of the evolution.
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PROTEUS_ROOT

# vulcan.climate is imported in the tests, not here: a unit run collects this file after
# tests/atmos_chem/test_vulcan.py has put a mock in place of vulcan
pytest.importorskip('vulcan')

from proteus import Proteus  # noqa: E402
from proteus.atmos_chem.vulcan_agni import VULCAN_TMP_FLOOR, SnapshotClimate  # noqa: E402
from proteus.atmos_clim.agni import (  # noqa: E402
    SOCRATES_MAX_PATH,
    activate_julia,
    deallocate_atmos,
    init_agni_atmos,
    run_agni,
)
from proteus.atmos_clim.common import read_ncdf_profile  # noqa: E402
from proteus.star.wrapper import write_spectrum  # noqa: E402
from proteus.utils.coupler import ZeroHelpfileRow  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

RUN_NIGHTLY_SMOKE = os.environ.get('PROTEUS_CI_NIGHTLY', '0') == '1'

# AGNI ends a solve once the flux residual is within solution_atol + solution_rtol |F|, so
# solving a converged state again moves T only within that band: 0.15 K was measured for
# this case, on the same struct with and without reloading the snapshot. 1 K keeps margin.
ATOL_FIXED_POINT_K = 1.0

# Swapping the H2O and CO2 abundances moves this banded climate by about 27 K, far outside
# the fixed-point tolerance, so a composition that never reached AGNI would fail the test.
MIN_COMPOSITION_RESPONSE_K = 10.0 * ATOL_FIXED_POINT_K


def _stellar_spectrum(n_points: int = 2000, t_star: float = 5772.0):
    """Black-body stellar spectrum at 1 AU in `.sflux` units [nm, erg s-1 cm-2 nm-1]."""
    h_planck, c_light, k_boltz = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    wl = np.logspace(np.log10(0.5), np.log10(1.0e6), n_points)
    exponent = np.minimum(h_planck * c_light / (wl * 1.0e-9 * k_boltz * t_star), 700.0)
    flux_si = np.pi * (2.0 * h_planck * c_light**2 / (wl * 1.0e-9) ** 5) / np.expm1(exponent)
    return wl, flux_si * 1.0e-6 * (6.957e8 / 1.495978707e11) ** 2


def _hf_row():
    """Helpfile row of a 10 bar H2O-CO2-H2 atmosphere over a 1500 K surface at 100 yr."""
    hf_row = ZeroHelpfileRow()
    hf_row.update(
        Time=100.0,
        T_surf=1500.0,
        T_magma=1500.0,
        P_surf=10.0,
        gravity=9.81,
        R_int=6.371e6,
        R_xuv=6.371e6,
        p_xuv=1.0e-5,
        F_ins=1361.0,
        albedo_pl=0.0,
        H2O_vmr=0.6,
        CO2_vmr=0.3,
        H2_vmr=0.1,
        R_star=6.957e8,
        semimajorax=1.495978707e11,
        M_int=5.97e24,
    )
    return hf_row


def _make_snapshot(tmpdir: str):
    """Solve and write one banded AGNI atmosphere snapshot, as the evolution writes one.

    Returns the directories, the configuration, the helpfile row and the snapshot profiles:
    layer pressures [Pa] and temperatures [K], gas names and mixing ratios [layer, gas].
    """
    runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')
    cfg = runner.config
    cfg.atmos_clim.module = 'agni'
    cfg.atmos_clim.agni.spectral_file = None  # banded, built from FWL_DATA
    cfg.atmos_clim.rayleigh = False
    cfg.atmos_clim.agni.solve_energy = True
    cfg.atmos_clim.agni.chemistry = 'none'
    cfg.atmos_clim.agni.rainout = False
    cfg.atmos_clim.agni.oceans = False
    cfg.atmos_clim.surf_state = 'fixed'
    cfg.atmos_clim.num_levels = 30
    cfg.interior_energetics.module = 'dummy'
    cfg.interior_struct.module = 'dummy'
    cfg.params.out.path = str(Path(tmpdir) / 'output')
    cfg.params.out.plot_mod = 0
    cfg.params.out.write_mod = 0
    runner.init_directories()
    dirs = runner.directories
    # The SOCRATES tools read file names of at most 132 bytes and loop on a longer one
    longest = os.path.join(dirs['output/offchem'], 'agni', 'socstar.dat')
    assert len(os.fsencode(longest)) <= SOCRATES_MAX_PATH, longest
    os.makedirs(os.path.join(dirs['output'], 'data'), exist_ok=True)
    os.makedirs(dirs['output/offchem'], exist_ok=True)

    hf_row = _hf_row()
    wl, fl = _stellar_spectrum()
    write_spectrum(wl, fl, hf_row, dirs['output'])

    activate_julia(dirs, 0)
    atmos = init_agni_atmos(dirs, cfg, hf_row)
    atmos, output = run_agni(atmos, 0, dirs, cfg, hf_row, write_data=True)
    assert output['agni_converged'], 'the snapshot climate must converge'
    hf_row['F_atm'] = output['F_atm']
    gases = [str(g) for g in atmos.gas_names]
    snapshot = dict(
        p=np.array(atmos.p, dtype=float),
        tmp=np.array(atmos.tmp, dtype=float),
        gases=gases,
        x=np.array([np.array(atmos.gas_vmr[g], dtype=float) for g in gases]).T,
    )
    deallocate_atmos(atmos)
    return dirs, cfg, hf_row, snapshot


@pytest.fixture(scope='module')
def snapshot():
    """One banded AGNI snapshot shared by the cases of this module, in a short temporary folder."""
    with tempfile.TemporaryDirectory(prefix='va') as tmpdir:
        yield _make_snapshot(tmpdir)


@pytest.mark.physics_invariant
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_climate_update_with_the_snapshot_composition_returns_the_snapshot(snapshot):
    """Given the snapshot's own composition, the climate update returns the snapshot climate.

    Physical scenario: a 10 bar H2O-CO2-H2 atmosphere over a 1500 K surface, solved by AGNI
    with banded radiation as the evolution solves it. Chemistry that leaves the composition
    as it was must leave the climate as it was: the snapshot is a fixed point of the update.

    Invariant: the snapshot is a fixed point of the climate update, within the convergence
    band of AGNI. The update must also respond to the composition, so the same update with
    the H2O and CO2 abundances swapped must move the temperature by far more than that.
    """
    dirs, cfg, hf_row, snap = snapshot
    climate = SnapshotClimate(dirs, cfg, hf_row)
    try:
        pco = snap['p'][::-1] * 10.0  # VULCAN: bottom up, dyn cm-2
        update = climate(pco, snap['tmp'][::-1], snap['gases'], snap['x'][::-1])
        t_same = np.array(climate.atmos.tmp, dtype=float)

        swapped = snap['x'].copy()
        i_h2o, i_co2 = snap['gases'].index('H2O'), snap['gases'].index('CO2')
        swapped[:, [i_h2o, i_co2]] = swapped[:, [i_co2, i_h2o]]
        climate(pco, snap['tmp'][::-1], snap['gases'], swapped[::-1])
        t_swapped = np.array(climate.atmos.tmp, dtype=float)
    finally:
        climate.close()

    # Fixed point: the snapshot climate comes back within AGNI's convergence band
    np.testing.assert_allclose(t_same, snap['tmp'], rtol=0, atol=ATOL_FIXED_POINT_K)
    # Discrimination: the update responds to the composition far beyond that band
    assert np.max(np.abs(t_swapped - snap['tmp'])) > MIN_COMPOSITION_RESPONSE_K
    # VULCAN receives AGNI's temperature, floored, on AGNI's layers in dyn cm-2 (x10 from Pa)
    np.testing.assert_allclose(update.p, snap['p'] * 10.0, rtol=1e-12)
    np.testing.assert_allclose(update.tmp, np.maximum(t_same, VULCAN_TMP_FLOOR), rtol=1e-12)
    assert update.tmp.min() >= VULCAN_TMP_FLOOR
    assert np.all(update.kzz > 0)


def _read_nc(path, *names):
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        out = {name: np.array(ds[name][:]) for name in names}
        out['attrs'] = {key: ds.getncattr(key) for key in ds.ncattrs()}
    return out


@pytest.mark.physics_invariant
# The number of climate updates varies from run to run (2 or 3 seen on CI), so the limit
# covers a third update (a 3-update run took 433 s on CI)
@pytest.mark.timeout(1200)
@pytest.mark.skipif(
    not RUN_NIGHTLY_SMOKE,
    reason='AGNI coupling test requires Julia/AGNI binaries (nightly only)',
)
def test_coupled_run_ends_where_chemistry_and_climate_agree(snapshot, record_property):
    """A coupled VULCAN run ends at a state that both the chemistry and the climate accept.

    Physical scenario: the CHO kinetics of the same snapshot, with photochemistry, coupled to
    its climate (couple_agni, updates at steady states). At 1500 K the kinetics turn part of
    the CO2 and H2 into CO, which the snapshot does not hold, so the climate of the end
    state is solved with a composition the evolution never gave AGNI.

    Invariants: VULCAN saves the state it is steady for, whose temperature differs from the
    climate of its final composition by the recorded last change, at most agni_T_rtol; and
    that climate is in radiative-convective equilibrium, with the same net flux at every
    height.
    """
    from vulcan.climate import interp_logp

    from proteus.atmos_chem.vulcan import run_vulcan

    dirs, cfg, hf_row, snap = snapshot
    cfg = copy.deepcopy(cfg)
    cfg.atmos_chem.module = 'vulcan'
    cfg.atmos_chem.photo_on = True
    cfg.atmos_chem.vulcan.network = 'CHO'
    cfg.atmos_chem.vulcan.make_funs = True
    cfg.atmos_chem.vulcan.couple_agni = True
    cfg.atmos_chem.vulcan.agni_call_frq = 0

    assert run_vulcan(dirs, cfg, hf_row)

    data = os.path.join(dirs['output'], 'data')
    chem = _read_nc(os.path.join(data, '100p000_chem.nc'), 'p', 'tmp', 'x_gas', 'species')
    coupled = read_ncdf_profile(
        os.path.join(data, '100p000_chematm.nc'),
        extra_keys=['p', 'tmp', 'fl_tot', 'x_gas'],
        combine_edges=False,
    )
    coupled['co'] = coupled['CO_vmr']

    # VULCAN ended at a steady state, after the climate update made there changed T little
    attrs = chem['attrs']
    rel_dt = np.atleast_1d(attrs['climate_max_rel_dT'])
    record_property('climate_updates', int(attrs['climate_updates']))
    record_property('climate_max_rel_dT', rel_dt.tolist())
    assert attrs['termination'] == 'converged to steady state'
    assert attrs['climate_updates'] >= 2
    assert rel_dt[-1] <= cfg.atmos_chem.vulcan.agni_T_rtol

    # The chemistry changed the composition, and AGNI solved the climate with it: CO, absent
    # from the snapshot, is a few per cent in both VULCAN and the coupled climate
    species = [str(sp) for sp in chem['species']]
    co_vulcan = chem['x_gas'][:, species.index('CO')]
    co_on_agni = interp_logp(chem['p'] / 10.0, co_vulcan, coupled['p'])  # Pa on both
    assert snap['x'][:, snap['gases'].index('CO')].max() == 0.0
    assert co_vulcan.max() > 1e-2
    # renormalised over AGNI's gases, which leaves out VULCAN's radicals (below 1e-4 here)
    np.testing.assert_allclose(coupled['co'], co_on_agni, rtol=1e-3)

    # VULCAN saved the temperature it is steady for, from before the last update; the
    # coupled climate (floored, on VULCAN's layers) differs from it by that update's change
    t_floored = np.maximum(coupled['tmp'], VULCAN_TMP_FLOOR)
    t_on_vulcan = interp_logp(coupled['p'] * 10.0, t_floored, chem['p'])  # dyn cm-2
    rel_change = np.max(np.abs(t_on_vulcan - chem['tmp']) / chem['tmp'])
    assert rel_change == pytest.approx(rel_dt[-1], rel=1e-6)

    # Radiative-convective equilibrium: the net upward flux is the same at every height. The
    # converged AGNI states of this case hold it to 3e-3 (snapshot) and 5e-4 (coupled) of
    # its value; 1e-2 bounds both, while a climate left unsolved for a new composition
    # carries the change in absorbed flux as a divergence
    f_net = coupled['fl_tot']
    assert (f_net.max() - f_net.min()) <= 1e-2 * abs(f_net.mean())
    assert f_net.mean() > 0  # the interior loses heat through this atmosphere
