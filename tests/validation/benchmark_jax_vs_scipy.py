"""Performance benchmark: JAX diffrax vs scipy BDF entropy solver.

Benchmarks wall-clock time for a grey-body cooling problem across
different mesh resolutions (N = 9, 20, 50, 100, 200).

Measures:
- JIT compilation time (first call only)
- Per-step solve time (warm JIT)
- scipy BDF with sparse Jacobian for comparison

Usage:
    python3 tests/validation/benchmark_jax_vs_scipy.py

Output:
    output_files/benchmark_jax_vs_scipy/ (plots + CSV)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np

# Ensure PROTEUS is importable
PROTEUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROTEUS_ROOT / 'aragog' / 'src'))
sys.path.insert(0, str(PROTEUS_ROOT / 'src'))

jax.config.update('jax_enable_x64', True)

SECS_PER_YEAR = 31557600.0
SIGMA_SB = 5.670374419e-8

# EOS directory
EOS_DIR = Path(os.environ.get(
    'ARAGOG_TEST_EOS_DIR',
    str(PROTEUS_ROOT / 'output' / 'coupled_parity' / 'spider' / 'data' / 'spider_eos'),
))

OUTPUT_DIR = PROTEUS_ROOT / 'output_files' / 'benchmark_jax_vs_scipy'


def make_mesh_arrays(N):
    """Build MeshArrays for benchmarks."""
    from aragog.jax.phase import MeshArrays

    R_INNER, R_OUTER = 5.371e6, 6.371e6
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

    return MeshArrays(
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
    ), {
        'r_stag': r_stag, 'r_basic': r_basic, 'P_stag': P_stag,
        'P_basic': P_basic, 'area': area, 'volume': volume,
        'd_dr': d_dr, 'q_mat': q_mat,
    }


def benchmark_jax(eos_jax, N, t_end=100.0, method='tsit5', n_repeats=3):
    """Benchmark JAX solver at resolution N.

    Returns dict with jit_time, warm_time (mean of n_repeats), n_steps.
    """
    from aragog.jax.phase import PhaseParams
    from aragog.jax.solver import BoundaryParams, solve_entropy

    mesh, _ = make_mesh_arrays(N)
    S_init = jnp.full(N, 3200.0)
    heating = jnp.zeros(N)
    params = PhaseParams()
    bc = BoundaryParams(
        outer_bc_type=1, outer_bc_value=0.0,
        emissivity=1.0, T_eq=255.0,
        inner_bc_type=0, inner_bc_value=0.0,
        core_density=10738.0, core_heat_capacity=880.0,
        tfac_core_avg=1.147,
    )

    # Increase max_steps for larger N (stiff problem)
    max_steps = max(100_000, N * 10_000)

    # First call: includes JIT compilation
    t0 = time.perf_counter()
    try:
        result = solve_entropy(
            S_init, 0.0, t_end, eos_jax, params, mesh, bc, heating,
            atol=0.1, rtol=1e-4, max_steps=max_steps, method=method,
        )
        jit_time = time.perf_counter() - t0
        success = result.success
        n_steps = result.n_steps
    except Exception as e:
        jit_time = time.perf_counter() - t0
        print(f'  FAILED: JAX {method} N={N}: {e}')
        return {'jit_time': jit_time, 'warm_time': np.nan, 'warm_std': 0, 'n_steps': 0}

    if not success:
        print(f'  WARNING: JAX {method} N={N} failed after {n_steps} steps')
        return {'jit_time': jit_time, 'warm_time': np.nan, 'warm_std': 0, 'n_steps': n_steps}

    # Warm calls
    warm_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        try:
            result = solve_entropy(
                S_init, 0.0, t_end, eos_jax, params, mesh, bc, heating,
                atol=0.1, rtol=1e-4, max_steps=max_steps, method=method,
            )
            warm_times.append(time.perf_counter() - t0)
        except Exception:
            warm_times.append(np.nan)

    return {
        'jit_time': jit_time,
        'warm_time': np.nanmean(warm_times) if warm_times else np.nan,
        'warm_std': np.nanstd(warm_times) if warm_times else 0,
        'n_steps': n_steps,
    }


def benchmark_scipy(eos_np, N, t_end=100.0, n_repeats=3):
    """Benchmark scipy BDF solver at resolution N."""
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags

    from aragog.eos.entropy_phase import EntropyPhaseEvaluator

    _, np_arrays = make_mesh_arrays(N)
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

    S_init = np.full(N, 3200.0)

    # Sparse Jacobian pattern (pentadiagonal)
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

        T_surf = phase_basic.temperature()[-1]
        heat_flux[-1] = SIGMA_SB * (T_surf**4 - 255.0**4)
        heat_flux[0] = 0.0

        energy_flux = heat_flux * area
        cap = phase_stag.density() * phase_stag.temperature() * volume
        return -np.diff(energy_flux) / cap * SECS_PER_YEAR

    times = []
    n_steps = 0
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        sol = solve_ivp(rhs, (0, t_end), S_init, method='BDF',
                        atol=0.1, rtol=1e-4, jac_sparsity=jac_sparsity)
        times.append(time.perf_counter() - t0)
        if sol.status == 0:
            n_steps = sol.nfev

    return {
        'warm_time': np.mean(times),
        'warm_std': np.std(times),
        'n_steps': n_steps,
    }


def main():
    from aragog.eos.entropy import EntropyEOS
    from aragog.jax.eos import EntropyEOS_JAX

    if not EOS_DIR.exists():
        print(f'EOS tables not found at {EOS_DIR}')
        print('Set ARAGOG_TEST_EOS_DIR or run SPIDER to generate tables.')
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading EOS tables...')
    eos_jax = EntropyEOS_JAX(EOS_DIR)
    eos_np = EntropyEOS(EOS_DIR)

    resolutions = [9, 20, 50, 100, 200]
    t_end = 100.0  # yr
    n_repeats = 3

    results = []
    print(f'\nBenchmark: grey-body cooling, t_end={t_end} yr, {n_repeats} repeats')
    print(f'{"N":>5} {"JAX JIT (s)":>12} {"JAX warm (s)":>14} {"scipy BDF (s)":>14} {"JAX steps":>10} {"scipy nfev":>11} {"speedup":>8}')
    print('-' * 85)

    for N in resolutions:
        print(f'  N={N}...', end=' ', flush=True)

        # JAX Tsit5
        jax_res = benchmark_jax(eos_jax, N, t_end, method='tsit5', n_repeats=n_repeats)
        # scipy BDF
        scipy_res = benchmark_scipy(eos_np, N, t_end, n_repeats=n_repeats)

        jax_warm = jax_res['warm_time']
        scipy_warm = scipy_res['warm_time']
        speedup = scipy_warm / jax_warm if (not np.isnan(jax_warm) and jax_warm > 0) else np.nan

        jax_str = f'{jax_warm:>14.3f}' if not np.isnan(jax_warm) else '          FAIL'
        spd_str = f'{speedup:>8.2f}x' if not np.isnan(speedup) else '    N/A'
        print(f'\r{N:>5} {jax_res["jit_time"]:>12.2f} {jax_str} {scipy_warm:>14.3f} {jax_res["n_steps"]:>10} {scipy_res["n_steps"]:>11} {spd_str}')

        results.append({
            'N': N,
            'jax_jit_s': jax_res['jit_time'],
            'jax_warm_s': jax_res['warm_time'],
            'jax_warm_std': jax_res.get('warm_std', 0),
            'jax_steps': jax_res['n_steps'],
            'scipy_s': scipy_res['warm_time'],
            'scipy_std': scipy_res.get('warm_std', 0),
            'scipy_nfev': scipy_res['n_steps'],
            'speedup': speedup,
        })

    # Save CSV
    csv_path = OUTPUT_DIR / 'benchmark_results.csv'
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {csv_path}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Ns = [r['N'] for r in results]
        jax_warm = [r['jax_warm_s'] for r in results]
        jax_jit = [r['jax_jit_s'] for r in results]
        scipy_t = [r['scipy_s'] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel (a): Wall-clock time vs N
        ax = axes[0]
        ax.plot(Ns, jax_warm, 'o-', label='JAX Tsit5 (warm)', color='C0')
        ax.plot(Ns, jax_jit, 's--', label='JAX Tsit5 (incl. JIT)', color='C0', alpha=0.5)
        ax.plot(Ns, scipy_t, '^-', label='scipy BDF (sparse J)', color='C1')
        ax.set_xlabel('Number of mesh nodes N')
        ax.set_ylabel('Wall-clock time [s]')
        ax.set_title('(a) Solve time vs resolution')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Panel (b): Speedup
        ax = axes[1]
        speedups = [r['speedup'] for r in results]
        ax.bar(range(len(Ns)), speedups, tick_label=[str(n) for n in Ns], color='C2')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of mesh nodes N')
        ax.set_ylabel('scipy / JAX speedup')
        ax.set_title('(b) Relative performance (>1 = JAX faster)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = OUTPUT_DIR / 'benchmark_jax_vs_scipy.pdf'
        plt.savefig(fig_path)
        print(f'Plot saved to {fig_path}')
        plt.close()

    except ImportError:
        print('matplotlib not available, skipping plot')


if __name__ == '__main__':
    main()
