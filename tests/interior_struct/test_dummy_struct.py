"""Tests for the dummy interior structure module (Noack & Lasbleis 2020)."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.unit
class TestNoackScalingLaws:
    """Verify the Noack & Lasbleis (2020) scaling law implementation."""

    def test_earth_like_planet_radius(self):
        """1 M_Earth with X_Fe~0.32 should give R ~ 6370 km."""
        from proteus.interior_struct.dummy import _iron_fractions

        x_cmf, x_fe, x_fem = _iron_fractions(0.325, 'mass', fe_mantle=0.1)
        # Eq. 5: R_p = (7030 - 1840 * X_Fe) * (M/M_E)^0.282
        R_p_km = (7030 - 1840 * x_fe) * 1.0**0.282
        # Earth radius is ~6371 km; scaling law should give within 5%
        assert abs(R_p_km - 6371) / 6371 < 0.05, f'R_p={R_p_km:.0f} km'

    def test_core_radius_earth_like(self):
        """Core radius for 1 M_Earth, X_CMF=0.325 should be ~3480 km."""
        # Eq. 9: R_c = 4850 * X_CMF^0.328 * (M/M_E)^0.266
        R_c_km = 4850 * 0.325**0.328 * 1.0**0.266
        assert abs(R_c_km - 3480) / 3480 < 0.10, f'R_c={R_c_km:.0f} km'

    def test_surface_gravity_earth_like(self):
        """Surface gravity for 1 M_Earth should be ~9.8 m/s^2."""
        from proteus.utils.constants import M_earth, const_G

        x_fe = 0.32
        R_p = (7030 - 1840 * x_fe) * 1.0**0.282 * 1e3  # [m]
        g = const_G * M_earth / R_p**2
        assert abs(g - 9.81) / 9.81 < 0.05, f'g={g:.2f} m/s^2'

    def test_iron_fractions_mass_mode(self):
        """Iron fractions from mass-mode core_frac."""
        from proteus.interior_struct.dummy import _iron_fractions

        x_cmf, x_fe, x_fem = _iron_fractions(0.325, 'mass')
        assert x_cmf == 0.325
        assert 0.0 < x_fem < 0.15  # mantle iron fraction
        assert 0.30 < x_fe < 0.40  # total iron fraction

    def test_iron_fractions_radius_mode(self):
        """Iron fractions from radius-mode core_frac."""
        from proteus.interior_struct.dummy import _iron_fractions

        x_cmf, x_fe, x_fem = _iron_fractions(0.55, 'radius')
        assert 0.01 < x_cmf < 0.80

    def test_solve_dummy_structure_fills_hf_row(self):
        """Full dummy solve fills all required hf_row keys."""
        from proteus.interior_struct.dummy import solve_dummy_structure

        config = SimpleNamespace(
            planet=SimpleNamespace(
                mass_tot=1.0,
                temperature_mode='adiabatic',
                tsurf_init=4000.0,
                tcenter_init=6000.0,
                f_accretion=0.04,
                f_differentiation=0.50,
            ),
            interior_struct=SimpleNamespace(
                core_frac=0.325,
                core_frac_mode='mass',
                core_heatcap='self',
                eos_dir='PALEOS_MgSiO3',
            ),
            interior_energetics=SimpleNamespace(
                num_levels=50,
                module='aragog',
            ),
        )
        hf_row = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'), exist_ok=True)
            solve_dummy_structure(config, hf_row, tmpdir)

        # Check all required keys
        assert hf_row['R_int'] > 0
        assert hf_row['R_core'] > 0
        assert hf_row['R_core'] < hf_row['R_int']
        assert hf_row['M_int'] > 0
        assert hf_row['M_core'] > 0
        assert hf_row['M_core'] < hf_row['M_int']
        assert hf_row['gravity'] > 0
        assert hf_row['core_density'] > 0
        assert hf_row['core_heatcap'] > 0

    def test_solve_dummy_writes_output_files(self):
        """Dummy solve writes zalmoxis_output.dat and temp file."""
        from proteus.interior_struct.dummy import solve_dummy_structure

        config = SimpleNamespace(
            planet=SimpleNamespace(
                mass_tot=1.0,
                temperature_mode='isothermal',
                tsurf_init=3000.0,
                tcenter_init=6000.0,
                f_accretion=0.04,
                f_differentiation=0.50,
            ),
            interior_struct=SimpleNamespace(
                core_frac=0.325,
                core_frac_mode='mass',
                core_heatcap=880.0,
                eos_dir='PALEOS_MgSiO3',
            ),
            interior_energetics=SimpleNamespace(
                num_levels=30,
                module='aragog',
            ),
        )
        hf_row = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            solve_dummy_structure(config, hf_row, tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, 'data', 'zalmoxis_output.dat'))
            assert os.path.isfile(os.path.join(tmpdir, 'data', 'zalmoxis_output_temp.txt'))

    def test_temperature_modes(self):
        """All four temperature modes produce valid profiles."""
        from proteus.interior_struct.dummy import solve_dummy_structure

        for mode in ('isothermal', 'linear', 'adiabatic', 'accretion'):
            config = SimpleNamespace(
                planet=SimpleNamespace(
                    mass_tot=1.0,
                    temperature_mode=mode,
                    tsurf_init=4000.0,
                    tcenter_init=6000.0,
                    f_accretion=0.04,
                    f_differentiation=0.50,
                ),
                interior_struct=SimpleNamespace(
                    core_frac=0.325,
                    core_frac_mode='mass',
                    core_heatcap='self',
                    eos_dir='PALEOS_MgSiO3',
                ),
                interior_energetics=SimpleNamespace(
                    num_levels=20,
                    module='aragog',
                ),
            )
            hf_row = {}
            with tempfile.TemporaryDirectory() as tmpdir:
                solve_dummy_structure(config, hf_row, tmpdir)
                # Read back temperature profile
                data = np.loadtxt(os.path.join(tmpdir, 'data', 'zalmoxis_output.dat'))
                T = data[:, 4]
                assert np.all(T > 0), f'{mode}: negative temperatures'
                assert np.all(np.isfinite(T)), f'{mode}: NaN temperatures'

    def test_mass_scaling(self):
        """Heavier planets should have larger radii."""
        from proteus.interior_struct.dummy import solve_dummy_structure

        radii = []
        for mass in [0.5, 1.0, 2.0]:
            config = SimpleNamespace(
                planet=SimpleNamespace(
                    mass_tot=mass,
                    temperature_mode='isothermal',
                    tsurf_init=3000.0,
                    tcenter_init=6000.0,
                    f_accretion=0.04,
                    f_differentiation=0.50,
                ),
                interior_struct=SimpleNamespace(
                    core_frac=0.325,
                    core_frac_mode='mass',
                    core_heatcap=880.0,
                    eos_dir='PALEOS_MgSiO3',
                ),
                interior_energetics=SimpleNamespace(
                    num_levels=10,
                    module='aragog',
                ),
            )
            hf_row = {}
            with tempfile.TemporaryDirectory() as tmpdir:
                solve_dummy_structure(config, hf_row, tmpdir)
            radii.append(hf_row['R_int'])
        assert radii[0] < radii[1] < radii[2], f'R should increase with M: {radii}'
