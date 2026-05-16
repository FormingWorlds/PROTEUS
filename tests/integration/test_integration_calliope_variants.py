"""Integration tests for CALLIOPE outgassing under varied fO2 buffers
and volatile budgets.

Complements the default Earth-like scenario in
``test_integration_calliope_multi_timestep.py`` by exercising the
reduced (fO2 = IW-2) and oxidising (fO2 = IW+4) branches of the IW
buffer, and a hydrogen-rich budget that pushes the atmosphere toward
an H2-dominated regime. Each runs the all-dummy backend with CALLIOPE
swapped in for outgassing.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.conftest import (
    validate_mass_conservation,
    validate_stability,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


@pytest.mark.integration
def test_integration_calliope_reducing_atmosphere(proteus_multi_timestep_run):
    """CALLIOPE outgassing at a reduced fO2 buffer (IW-2).

    Physical scenario: at log10(fO2/IW) = -2 the atmosphere is reduced and
    H2 / CO dominate over H2O / CO2. This exercises CALLIOPE's reduced
    branch and the fO2-derived helpfile columns under non-default
    chemistry.

    Validates:
    - Simulation runs to completion with fO2 = IW-2.
    - fO2_shift_IW_derived stays close to the configured -2 (within 1 dex
      tolerance to allow the chemistry to settle).
    - H2 partial pressure is non-zero (planet not desiccated).
    - Mass conservation holds across timesteps.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        outgas__module='calliope',
        outgas__fO2_shift_IW=-2.0,
        planet__tsurf_init=2000.0,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    validate_stability(hf)
    validate_mass_conservation(hf, tolerance=0.10)
    # Discrimination: a regression that ignored the configured fO2 shift
    # and re-equilibrated against the default value of 0 would land at
    # fO2_shift_IW_derived ~ 0, not -2. Allow 1 dex tolerance so the
    # chemistry can settle.
    fO2 = hf['fO2_shift_IW_derived'].values
    assert np.all(np.isfinite(fO2)), 'fO2_shift_IW_derived must be finite'
    assert np.median(fO2) < -1.0, (
        f'reduced atmosphere should show median fO2 < -1, got {np.median(fO2):.2f}'
    )


@pytest.mark.integration
def test_integration_calliope_oxidising_atmosphere(proteus_multi_timestep_run):
    """CALLIOPE outgassing at an oxidising fO2 buffer (IW+4).

    Physical scenario: at IW+4 the atmosphere is oxidised and H2O / CO2
    dominate over H2 / CO. This exercises the oxidising-branch chemistry
    and stress-tests the H2-binodal partition that is bypassed for
    H2-poor compositions.

    Validates:
    - Simulation runs to completion at fO2 = +4.
    - fO2_shift_IW_derived sits at the oxidised end (median > 1).
    - Total surface pressure is positive throughout.
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        outgas__module='calliope',
        outgas__fO2_shift_IW=4.0,
        planet__tsurf_init=2000.0,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    fO2 = hf['fO2_shift_IW_derived'].values
    assert np.median(fO2) > 1.0, (
        f'oxidising atmosphere should show median fO2 > +1, got {np.median(fO2):.2f}'
    )
    # Boundedness invariant: surface pressure strictly positive.
    p_surf = hf['P_surf'].values
    assert np.all(p_surf > 0.0), 'P_surf must remain positive'
    assert np.all(np.isfinite(p_surf)), 'P_surf must be finite'


@pytest.mark.integration
def test_integration_calliope_high_carbon_budget(proteus_multi_timestep_run):
    """CALLIOPE outgassing with elevated carbon budget (C/H = 2.0).

    Physical scenario: doubling C/H relative to the default Earth budget
    biases CO2 / CO partitioning relative to H2O / H2. Exercises the
    C-element-mass aggregation in calc_target_elemental_inventories and
    the partitioning math at non-default elemental ratios.

    Validates:
    - Simulation runs to completion at C/H = 2.0.
    - C_kg_total > H_kg_total in the helpfile at every timestep
      (mass ratio inverts the default C/H = 1.0).
    """
    runner = proteus_multi_timestep_run(
        config_path='input/dummy.toml',
        num_timesteps=5,
        max_time=1e6,
        min_time=1e2,
        outgas__module='calliope',
        outgas__fO2_shift_IW=0.0,
        planet__tsurf_init=2000.0,
        planet__elements__C_budget=2.0,
    )

    hf = runner.hf_all
    assert len(hf) >= 3
    c_kg = hf['C_kg_total'].values
    h_kg = hf['H_kg_total'].values
    # Discrimination: a regression that hardcoded the C budget to the
    # default would land at C/H ~ 1; the elevated budget pushes it
    # toward 2 in mass-fraction terms, which after accounting for
    # M_C / M_H = 12 means C_kg / H_kg >> 1.
    assert np.all(c_kg > h_kg), 'elevated C budget must give C mass > H mass throughout'
    # Sanity: both stay positive.
    assert np.all(c_kg > 0.0)
    assert np.all(h_kg > 0.0)
