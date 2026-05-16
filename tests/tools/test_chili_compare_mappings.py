"""Regression tests for the CHILI-vs-PROTEUS comparison script's
column mappings.

The script at ``tests/validation/chili/compare_to_chili.py`` defines a
``CHILI_TO_PROTEUS`` dict that maps CHILI Table 3 column names to
PROTEUS helpfile column names. After PR #659 landed, three of those
mappings were wrong by physically distinct quantities:

- ``T_pot(K)`` -> ``T_magma`` (outgassing temperature, not potential
  temperature)
- ``flux_ASR(W/m2)`` -> ``F_ins`` (raw instellation flux, not absorbed
  stellar radiation; off by ``s0_factor * (1 - albedo)`` ~ 0.34 at
  Earth)
- ``phi(vol_frac)`` -> ``Phi_global`` (mass-weighted melt fraction, not
  volume-weighted; differs by the solid/liquid density ratio in the
  mushy zone)

This file is loaded via importlib because ``tests/validation/chili/``
is a script directory, not a pytest package, and lacks an
``__init__.py``.

See also:
- docs/test_infrastructure.md
- docs/test_categorization.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_compare_to_chili():
    """Load ``tests/validation/chili/compare_to_chili.py`` as a module."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / 'tests' / 'validation' / 'chili' / 'compare_to_chili.py'
    spec = importlib.util.spec_from_file_location('_compare_to_chili', src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_chili_mapping_T_pot_uses_PROTEUS_T_pot_not_T_magma():
    """CHILI ``T_pot(K)`` is the potential temperature; PROTEUS exposes
    that as ``T_pot``. Mapping to ``T_magma`` (the outgassing
    temperature) used to silently produce a several-hundred-K offset
    during the mushy stage when the two diverge."""
    mod = _load_compare_to_chili()
    assert mod.CHILI_TO_PROTEUS['T_pot(K)'] == 'T_pot', (
        'T_pot(K) must map to PROTEUS T_pot, not T_magma'
    )
    # Discrimination guard: an explicit negative pin against the
    # previously-wrong column name catches a regression that reverted
    # to T_magma. The previous mapping diverged from T_pot by several
    # hundred K in the mushy stage; the two names must never coincide.
    assert mod.CHILI_TO_PROTEUS['T_pot(K)'] != 'T_magma'


@pytest.mark.unit
def test_chili_mapping_flux_ASR_uses_derived_F_asr_not_F_ins():
    """CHILI ``flux_ASR(W/m2)`` is absorbed stellar radiation. PROTEUS
    stores raw instellation as ``F_ins``; the absorbed quantity is
    ``F_asr = F_ins * s0_factor * (1 - albedo_pl)`` and is derived in
    ``load_proteus`` rather than stored in the helpfile."""
    mod = _load_compare_to_chili()
    assert mod.CHILI_TO_PROTEUS['flux_ASR(W/m2)'] == 'F_asr', (
        'flux_ASR(W/m2) must map to the derived F_asr column, not F_ins'
    )
    # Discrimination guard: an explicit negative pin against F_ins
    # catches a regression that reverted to the raw-instellation column.
    # The two differ by s0_factor * (1 - albedo_pl), about a factor of
    # 0.34 at Earth-like configurations.
    assert mod.CHILI_TO_PROTEUS['flux_ASR(W/m2)'] != 'F_ins'


@pytest.mark.unit
def test_chili_mapping_phi_vol_frac_uses_PROTEUS_Phi_global_vol():
    """CHILI ``phi(vol_frac)`` is the volume-weighted melt fraction;
    PROTEUS exposes that as ``Phi_global_vol``. ``Phi_global`` is
    mass-weighted and differs by the solid/liquid density ratio in
    the mushy zone (order 5-10% for silicates)."""
    mod = _load_compare_to_chili()
    assert mod.CHILI_TO_PROTEUS['phi(vol_frac)'] == 'Phi_global_vol', (
        'phi(vol_frac) must map to Phi_global_vol, not Phi_global'
    )
    # Discrimination guard: an explicit negative pin against Phi_global
    # catches a regression to the mass-weighted column. The two values
    # diverge by 5-10% in the mushy zone for silicate compositions.
    assert mod.CHILI_TO_PROTEUS['phi(vol_frac)'] != 'Phi_global'


@pytest.mark.unit
def test_chili_mapping_other_columns_unchanged():
    """Sanity guard: the seven non-corrected mappings stay byte-identical
    so the fix doesn't accidentally drift a working column name."""
    mod = _load_compare_to_chili()
    expected = {
        't(yr)': 'Time',
        'T_surf(K)': 'T_surf',
        'flux_OLR(W/m2)': 'F_olr',
        'p_surf(bar)': 'P_surf',
        'p_H2O(bar)': 'H2O_bar',
        'p_CO2(bar)': 'CO2_bar',
        'p_CO(bar)': 'CO_bar',
        'p_H2(bar)': 'H2_bar',
        'p_CH4(bar)': 'CH4_bar',
        'p_O2(bar)': 'O2_bar',
    }
    for ck, expected_pk in expected.items():
        assert mod.CHILI_TO_PROTEUS[ck] == expected_pk, (
            f'{ck!r} must continue mapping to {expected_pk!r}'
        )
    # Coverage guard: every CHILI key in the expected reference set
    # must actually exist in the loaded mapping. A regression that
    # silently dropped a key would let the loop's per-key equality
    # raise a KeyError, but pin the membership explicitly so the
    # failure surfaces as a clean assertion rather than a KeyError
    # traceback.
    for ck in expected:
        assert ck in mod.CHILI_TO_PROTEUS, f'{ck!r} missing from CHILI_TO_PROTEUS'
    # No-drift guard: the three corrected columns (T_pot, flux_ASR,
    # phi(vol_frac)) must still be the corrected values, not the
    # pre-fix values. This catches a regression that reverted any of
    # the three fixes in the SAME commit as a working-column rename.
    assert mod.CHILI_TO_PROTEUS['T_pot(K)'] == 'T_pot'
    assert mod.CHILI_TO_PROTEUS['flux_ASR(W/m2)'] == 'F_asr'
    assert mod.CHILI_TO_PROTEUS['phi(vol_frac)'] == 'Phi_global_vol'
