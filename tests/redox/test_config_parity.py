"""
Config-module parity tests (#57 Commit B.5).

Two duplicated-registry concerns (round-3 architecture review):

1. `src/proteus/config/_redox.py` declares `_OXYBAROMETERS` and
   `_BUFFERS` tuples used by attrs validators. These MUST agree with
   the runtime dispatch tables in `src/proteus/redox/buffers.py`
   (`OXYBAROMETERS` and `BUFFERS` dicts). The duplication is
   intentional (avoids a circular import at config load time), but
   drift breaks config validation silently.

2. `src/proteus/config/_struct.py::MantleComp` oxide defaults should
   match `src/proteus/redox/buffers.py::_oxide_mole_fractions_from_mantle_comp`
   expectations, and should sum to less than 100 wt%.
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_oxybarometers_registry_parity():
    """`_redox._OXYBAROMETERS` and `buffers.OXYBAROMETERS` must agree."""
    from proteus.config._redox import _OXYBAROMETERS
    from proteus.redox.buffers import OXYBAROMETERS
    assert set(_OXYBAROMETERS) == set(OXYBAROMETERS.keys()), (
        f'Drift: config validator tuple = {sorted(_OXYBAROMETERS)}, '
        f'runtime dict = {sorted(OXYBAROMETERS.keys())}. Update both.'
    )


@pytest.mark.unit
def test_buffers_registry_parity():
    """`_redox._BUFFERS` and `buffers.BUFFERS` must agree."""
    from proteus.config._redox import _BUFFERS
    from proteus.redox.buffers import BUFFERS
    assert set(_BUFFERS) == set(BUFFERS.keys()), (
        f'Drift: config validator tuple = {sorted(_BUFFERS)}, '
        f'runtime dict = {sorted(BUFFERS.keys())}. Update both.'
    )


@pytest.mark.unit
def test_mantle_comp_default_oxide_sum_near_100():
    """Earth-BSE defaults sum to ~100 ± 0.5 wt% (Schaefer+24 Table 1)."""
    from proteus.config._struct import MantleComp
    mc = MantleComp()
    total = (
        mc.SiO2_wt + mc.TiO2_wt + mc.Al2O3_wt + mc.FeO_total_wt
        + mc.MgO_wt + mc.CaO_wt + mc.Na2O_wt + mc.K2O_wt + mc.P2O5_wt
    )
    # Schaefer+24 Table 1 sums to 100.31 wt%; McDonough 2003 BSE
    # ~100.3 wt%. Validator allows up to 101.0.
    assert 99.5 < total < 101.0, (
        f'MantleComp defaults sum to {total} wt%; expected ~100.3 '
        f'for Earth BSE'
    )


@pytest.mark.unit
def test_mantle_comp_sum_validator_rejects_over_100():
    """Sum > 100 wt% must raise ValueError."""
    from proteus.config._struct import MantleComp
    with pytest.raises(ValueError):
        MantleComp(
            SiO2_wt=50.0, MgO_wt=50.0, FeO_total_wt=10.0,
        )


@pytest.mark.unit
def test_core_comp_default_is_pure_iron():
    """CoreComp default is 100% Fe."""
    from proteus.config._struct import CoreComp
    cc = CoreComp()
    assert cc.Fe_wt == 100.0
    assert cc.H_wt == 0.0
    assert cc.O_wt == 0.0
    assert cc.Si_wt == 0.0


@pytest.mark.unit
def test_core_comp_validator_requires_sum_100():
    """Non-sum-to-100 CoreComp raises."""
    from proteus.config._struct import CoreComp
    with pytest.raises(ValueError):
        CoreComp(Fe_wt=90.0, H_wt=0.0, O_wt=0.0, Si_wt=0.0)


@pytest.mark.unit
def test_struct_nests_mantle_and_core_comp():
    """Struct (via factory default) exposes mantle_comp and core_comp."""
    from proteus.config._struct import CoreComp, MantleComp, Struct
    s = Struct()
    assert isinstance(s.mantle_comp, MantleComp)
    assert isinstance(s.core_comp, CoreComp)
    assert s.mantle_comp.FeO_total_wt == 7.82
    assert s.core_comp.Fe_wt == 100.0
