#!/usr/bin/env python3
"""JAX vs numpy Aragog standalone parity test.

Runs the JAX and numpy entropy solvers on identical initial conditions
(same mesh, EOS, BCs) and compares T_magma, Phi_global, energy.

This is a standalone comparison (no full PROTEUS coupling loop) to
isolate solver parity from coupling effects.

Usage:
    python3 tests/validation/run_jax_parity.py

Output:
    output_files/jax_parity/ (plots + CSV)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROTEUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROTEUS_ROOT / 'aragog' / 'src'))
sys.path.insert(0, str(PROTEUS_ROOT / 'src'))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_enable_x64', True)

from scipy.integrate import solve_ivp  # noqa: E402

SECS_PER_YEAR = 31557600.0
SIGMA_SB = 5.670374419e-8

EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    str(PROTEUS_ROOT / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
))

OUTPUT_DIR = PROTEUS_ROOT / 'output_files' / 'jax_parity'

R_INNER = 5.371e6
R_OUTER = 6.371e6


def make_mesh(N):
    """Build mesh arrays for both JAX and numpy."""
    from aragog.jax.phase import MeshArrays

    r_stag = np.linspace(R_INNER, R_OUTER, N)
    dr = np.diff(r_stag)
    r_basic = np.zeros(N + 1)
    r_basic[0] = R_INNER
    r_basic[-1] = R_OUTER
    r_basic[1:-1] = 0.5 * (r_stag[:-1] + r_stag[1:])

    area = 4.0 * np.pi * r_basic**2
    volume = (4.0 / 3.0) * np.pi * np.diff(r_basic**3)
    ml = np.maximum(np.minimum(r_basic - R_INNER, R_OUTER - r_basic), 1.0)

    d_dr = np.zeros((N + 1, N))
    for i in range(1, N):
        d_dr[i, i - 1] = -1.0 / dr[i - 1]
        d_dr[i, i] = 1.0 / dr[i - 1]
    d_dr[0, :] = d_dr[1, :]
    d_dr[-1, :] = d_dr[-2, :]

    q_mat = np.zeros((N + 1, N))
    q_mat[0, 0] = 1.0
    q_mat[-1, -1] = 1.0
    for i in range(1, N):
        q_mat[i, i - 1] = 0.5
        q_mat[i, i] = 0.5

    P_stag = np.linspace(135e9, 1e5, N)
    P_basic = q_mat @ P_stag

    np_arrays = {
        'r_stag': r_stag, 'r_basic': r_basic, 'P_stag': P_stag,
        'P_basic': P_basic, 'area': area, 'volume': volume,
        'd_dr': d_dr, 'q_mat': q_mat, 'ml': ml, 'dr': dr,
    }

    jax_mesh = MeshArrays(
        d_dr_matrix=jnp.asarray(d_dr),
        quantity_matrix=jnp.asarray(q_mat),
        area=jnp.asarray(area),
        volume=jnp.asarray(volume),
        radii_basic=jnp.asarray(r_basic),
        radii_stag=jnp.asarray(r_stag),
        mixing_length=jnp.asarray(ml),
        mixing_length_sq=jnp.asarray(ml**2),
        mixing_length_cu=jnp.asarray(ml**3),
        P_stag=jnp.asarray(P_stag),
        P_basic=jnp.asarray(P_basic),
        gravity=jnp.full(N + 1, 10.0),
    )

    return jax_mesh, np_arrays


def run_scipy_bdf(eos_np, np_arrays, S_init, t_end, T_eq=255.0):
    """Run scipy BDF entropy solver (conduction + convection)."""
    from scipy.sparse import diags

    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    N = len(S_init)
    P_stag = np_arrays['P_stag']
    P_basic = np_arrays['P_basic']
    area = np_arrays['area']
    volume = np_arrays['volume']
    d_dr = np_arrays['d_dr']
    q_mat = np_arrays['q_mat']

    phase_stag = EntropyPhaseEvaluator(entropy_eos=eos_np, gravitational_acceleration=10.0)
    phase_stag.set_pressure(P_stag)
    phase_basic = EntropyPhaseEvaluator(entropy_eos=eos_np, gravitational_acceleration=10.0)
    phase_basic.set_pressure(P_basic)

    bandwidth = 2
    offsets = list(range(-bandwidth, bandwidth + 1))
    diag_data = [np.ones(N - abs(k)) for k in offsets]
    jac_sparsity = diags(diag_data, offsets, shape=(N, N), format='csc')

    def rhs(t, S):
        S = np.asarray(S)
        phase_stag.set_entropy(S)
        phase_stag.update()
        S_basic = q_mat @ S
        phase_basic.set_entropy(S_basic)
        phase_basic.update()

        T_stag = phase_stag.temperature()
        dTdr = d_dr @ T_stag
        k = (1.0 - phase_basic.melt_fraction()) * 4.0 + phase_basic.melt_fraction() * 2.0
        heat_flux = -k * dTdr

        # Grey-body surface BC
        T_surf = phase_basic.temperature()[-1]
        heat_flux[-1] = SIGMA_SB * (T_surf**4 - T_eq**4)
        heat_flux[0] = 0.0  # insulating core

        energy_flux = heat_flux * area
        cap = phase_stag.density() * phase_stag.temperature() * volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    # Dense output for time series
    t_eval = np.logspace(np.log10(max(1.0, t_end * 1e-4)), np.log10(t_end), 50)
    t0 = time.perf_counter()
    sol = solve_ivp(rhs, (0, t_end), S_init, method='BDF',
                    atol=0.1, rtol=1e-4, jac_sparsity=jac_sparsity,
                    t_eval=t_eval)
    wall_time = time.perf_counter() - t0

    if sol.status != 0:
        print(f'  scipy BDF failed: {sol.message}')
        return None

    # Extract T_magma and Phi_global at each saved time
    results = []
    for i, t in enumerate(sol.t):
        S = sol.y[:, i]
        T = eos_np.temperature(P_stag, S)
        phi = eos_np.melt_fraction(P_stag, S)
        rho = eos_np.density(P_stag, S)
        results.append({
            'time': t,
            'T_magma': T[-1],
            'T_core': T[0],
            'Phi_global': np.dot(phi, volume) / volume.sum(),
            'S_mean': np.mean(S),
            'S_surf': S[-1],
            'E_th': np.sum(rho * T * S * volume),
        })

    return {
        'times': np.array([r['time'] for r in results]),
        'T_magma': np.array([r['T_magma'] for r in results]),
        'T_core': np.array([r['T_core'] for r in results]),
        'Phi_global': np.array([r['Phi_global'] for r in results]),
        'S_mean': np.array([r['S_mean'] for r in results]),
        'S_surf': np.array([r['S_surf'] for r in results]),
        'wall_time': wall_time,
        'nfev': sol.nfev,
    }


def run_jax_tsit5(eos_jax, jax_mesh, S_init, t_end, T_eq=255.0):
    """Run JAX Tsit5 solver with grey-body BC.

    Runs in segments to capture time series.
    """
    from aragog.jax.phase import PhaseParams
    from aragog.jax.solver import BoundaryParams, solve_entropy

    N = len(S_init)
    params = PhaseParams(convection=True)
    bc = BoundaryParams(
        outer_bc_type=1,
        outer_bc_value=0.0,
        emissivity=1.0,
        T_eq=T_eq,
        inner_bc_type=0,
        inner_bc_value=0.0,
        core_density=10738.0,
        core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )
    heating = jnp.zeros(N)
    S = jnp.asarray(S_init)

    # Segment times for time series
    t_eval = np.logspace(np.log10(max(1.0, t_end * 1e-4)), np.log10(t_end), 50)
    t_prev = 0.0

    results = []
    P = jax_mesh.P_stag
    vol = np.asarray(jax_mesh.volume)

    t0_wall = time.perf_counter()
    total_steps = 0

    for t_seg in t_eval:
        try:
            result = solve_entropy(
                S, t_prev, t_seg,
                eos_jax, params, jax_mesh, bc, heating,
                atol=0.1, rtol=1e-4, max_steps=500_000,
                method='tsit5',
            )
        except Exception as e:
            print(f'  JAX Tsit5 failed at t={t_seg:.1f}: {e}')
            break

        if not result.success:
            print(f'  JAX Tsit5 failed at t={t_seg:.1f} (steps={result.n_steps})')
            break

        S = result.S_final
        total_steps += result.n_steps
        t_prev = t_seg

        T = np.asarray(eos_jax.temperature(P, S))
        phi = np.asarray(eos_jax.melt_fraction(P, S))
        rho = np.asarray(eos_jax.density(P, S))
        S_np = np.asarray(S)

        results.append({
            'time': t_seg,
            'T_magma': T[-1],
            'T_core': T[0],
            'Phi_global': np.dot(phi, vol) / vol.sum(),
            'S_mean': np.mean(S_np),
            'S_surf': S_np[-1],
            'E_th': np.sum(rho * T * S_np * vol),
        })

    wall_time = time.perf_counter() - t0_wall

    if not results:
        return None

    return {
        'times': np.array([r['time'] for r in results]),
        'T_magma': np.array([r['T_magma'] for r in results]),
        'T_core': np.array([r['T_core'] for r in results]),
        'Phi_global': np.array([r['Phi_global'] for r in results]),
        'S_mean': np.array([r['S_mean'] for r in results]),
        'S_surf': np.array([r['S_surf'] for r in results]),
        'wall_time': wall_time,
        'total_steps': total_steps,
    }


def run_case(eos_jax, eos_np, N, S_init_val, t_end, label):
    """Run one parity comparison case."""
    print(f'\n  Case: {label} (N={N}, S_init={S_init_val}, t_end={t_end} yr)')

    jax_mesh, np_arrays = make_mesh(N)
    S_init = np.full(N, S_init_val)

    # scipy BDF
    print('    scipy BDF...', end=' ', flush=True)
    scipy_res = run_scipy_bdf(eos_np, np_arrays, S_init, t_end)
    if scipy_res:
        print(f'done ({scipy_res["wall_time"]:.1f}s, {scipy_res["nfev"]} fev)')
    else:
        print('FAILED')
        return None

    # JAX Tsit5
    print('    JAX Tsit5...', end=' ', flush=True)
    jax_res = run_jax_tsit5(eos_jax, jax_mesh, S_init, t_end)
    if jax_res:
        print(f'done ({jax_res["wall_time"]:.1f}s, {jax_res["total_steps"]} steps)')
    else:
        print('FAILED')
        return None

    # Compare at common times
    common_times = np.intersect1d(
        np.round(scipy_res['times'], 2),
        np.round(jax_res['times'], 2),
    )

    if len(common_times) == 0:
        # Interpolate JAX onto scipy times
        from scipy.interpolate import interp1d
        for key in ['T_magma', 'T_core', 'Phi_global', 'S_mean', 'S_surf']:
            if len(jax_res['times']) > 1:
                f = interp1d(jax_res['times'], jax_res[key], fill_value='extrapolate')
                jax_res[key + '_interp'] = f(scipy_res['times'])
            else:
                jax_res[key + '_interp'] = np.full_like(scipy_res['times'], jax_res[key][0])

    # Compute parity metrics at final time
    # Use the minimum common end time
    t_common = min(scipy_res['times'][-1], jax_res['times'][-1])

    # Get final values
    idx_sp = np.argmin(np.abs(scipy_res['times'] - t_common))
    idx_jx = np.argmin(np.abs(jax_res['times'] - t_common))

    T_mag_sp = scipy_res['T_magma'][idx_sp]
    T_mag_jx = jax_res['T_magma'][idx_jx]
    Phi_sp = scipy_res['Phi_global'][idx_sp]
    Phi_jx = jax_res['Phi_global'][idx_jx]
    S_mean_sp = scipy_res['S_mean'][idx_sp]
    S_mean_jx = jax_res['S_mean'][idx_jx]

    print(f'    T_magma: scipy={T_mag_sp:.1f} K, jax={T_mag_jx:.1f} K, diff={abs(T_mag_sp-T_mag_jx):.1f} K')
    print(f'    Phi_global: scipy={Phi_sp:.4f}, jax={Phi_jx:.4f}, diff={abs(Phi_sp-Phi_jx):.4f}')
    print(f'    S_mean: scipy={S_mean_sp:.1f}, jax={S_mean_jx:.1f}, diff={abs(S_mean_sp-S_mean_jx):.1f} J/kg/K')

    return {
        'label': label,
        'N': N,
        'S_init': S_init_val,
        't_end': t_end,
        'scipy': scipy_res,
        'jax': jax_res,
        'dT_magma': abs(T_mag_sp - T_mag_jx),
        'dPhi': abs(Phi_sp - Phi_jx),
        'dS_mean': abs(S_mean_sp - S_mean_jx),
    }


def make_plots(all_results):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    n_cases = len(all_results)
    fig, axes = plt.subplots(n_cases, 3, figsize=(15, 4 * n_cases), squeeze=False)

    for i, res in enumerate(all_results):
        sp = res['scipy']
        jx = res['jax']

        # (a) T_magma(t)
        ax = axes[i, 0]
        ax.plot(sp['times'], sp['T_magma'], 'b-', label='scipy BDF', linewidth=2)
        ax.plot(jx['times'], jx['T_magma'], 'r--', label='JAX Tsit5', linewidth=2)
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('T_magma [K]')
        ax.set_title(f'{res["label"]}: T_magma')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # (b) Phi_global(t)
        ax = axes[i, 1]
        ax.plot(sp['times'], sp['Phi_global'], 'b-', label='scipy BDF', linewidth=2)
        ax.plot(jx['times'], jx['Phi_global'], 'r--', label='JAX Tsit5', linewidth=2)
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('Phi_global')
        ax.set_title(f'{res["label"]}: Phi_global')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # (c) S_mean(t)
        ax = axes[i, 2]
        ax.plot(sp['times'], sp['S_mean'], 'b-', label='scipy BDF', linewidth=2)
        ax.plot(jx['times'], jx['S_mean'], 'r--', label='JAX Tsit5', linewidth=2)
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('S_mean [J/kg/K]')
        ax.set_title(f'{res["label"]}: Mean entropy')
        ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'jax_parity_comparison.pdf'
    plt.savefig(fig_path)
    print(f'\nPlots saved to {fig_path}')
    plt.close()

    # Summary table
    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * n_cases))
    ax.axis('off')
    table_data = []
    for res in all_results:
        table_data.append([
            res['label'],
            f'{res["dT_magma"]:.1f} K',
            f'{res["dPhi"]:.4f}',
            f'{res["dS_mean"]:.1f} J/kg/K',
            f'{res["scipy"]["wall_time"]:.1f}s',
            f'{res["jax"]["wall_time"]:.1f}s',
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=['Case', 'dT_magma', 'dPhi', 'dS_mean', 'scipy time', 'JAX time'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title('JAX vs scipy Aragog Parity Summary')
    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'jax_parity_summary.pdf'
    plt.savefig(fig_path)
    print(f'Summary saved to {fig_path}')
    plt.close()


def main():
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.eos import EntropyEOS_JAX

    if not EOS_DIR.exists():
        print(f'EOS tables not found at {EOS_DIR}')
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading EOS tables...')
    eos_jax = EntropyEOS_JAX(EOS_DIR)
    eos_np = EntropyEOS(EOS_DIR)

    # Parity test cases: varying N and S_init
    cases = [
        # (N, S_init, t_end, label)
        (9, 3200.0, 100.0, 'N9_S3200_100yr'),
        (20, 3200.0, 100.0, 'N20_S3200_100yr'),
        (20, 3000.0, 100.0, 'N20_S3000_100yr'),
        (20, 3400.0, 100.0, 'N20_S3400_100yr'),
        (9, 3200.0, 500.0, 'N9_S3200_500yr'),
    ]

    all_results = []
    for N, S_init, t_end, label in cases:
        res = run_case(eos_jax, eos_np, N, S_init, t_end, label)
        if res:
            all_results.append(res)

    if all_results:
        make_plots(all_results)

        # Save CSV summary
        import csv
        csv_path = OUTPUT_DIR / 'parity_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'label', 'N', 'S_init', 't_end', 'dT_magma', 'dPhi', 'dS_mean',
                'scipy_wall_s', 'jax_wall_s',
            ])
            writer.writeheader()
            for res in all_results:
                writer.writerow({
                    'label': res['label'],
                    'N': res['N'],
                    'S_init': res['S_init'],
                    't_end': res['t_end'],
                    'dT_magma': f'{res["dT_magma"]:.2f}',
                    'dPhi': f'{res["dPhi"]:.6f}',
                    'dS_mean': f'{res["dS_mean"]:.2f}',
                    'scipy_wall_s': f'{res["scipy"]["wall_time"]:.2f}',
                    'jax_wall_s': f'{res["jax"]["wall_time"]:.2f}',
                })
        print(f'CSV saved to {csv_path}')

    print('\n=== PARITY ASSESSMENT ===')
    passes = 0
    for res in all_results:
        ok_T = res['dT_magma'] < 50.0
        ok_Phi = res['dPhi'] < 0.05
        ok_S = res['dS_mean'] < 20.0
        status = 'PASS' if (ok_T and ok_Phi and ok_S) else 'FAIL'
        if status == 'PASS':
            passes += 1
        print(f'  {res["label"]}: {status} (dT={res["dT_magma"]:.1f}K, dPhi={res["dPhi"]:.4f}, dS={res["dS_mean"]:.1f})')

    print(f'\n  {passes}/{len(all_results)} cases passed')


if __name__ == '__main__':
    main()
