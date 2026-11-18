"""CMB boundary-layer law by phase: the 'phase' setting and its transition band."""

from __future__ import annotations

import pytest

from proteus.config._interior import Aragog, AragogCoreModule, Interior


def _interior(ra='phase', loc=0.4, wid=0.2, core_bc='core_module', module='aragog'):
    return Interior(
        module=module,
        rfront_loc=loc,
        rfront_wid=wid,
        aragog=Aragog(core_bc=core_bc, core_module=AragogCoreModule(ra_crit_cmb=ra)),
    )


@pytest.mark.unit
def test_the_default_law_is_the_law_by_phase():
    assert AragogCoreModule().ra_crit_cmb == 'phase'
    for value in ('phase', 'eq16'):
        assert AragogCoreModule(ra_crit_cmb=value).ra_crit_cmb == value
    for bad in ('Phase', 'phase ', 'eq17'):
        with pytest.raises(ValueError, match='ra_crit_cmb'):
            AragogCoreModule(ra_crit_cmb=bad)


@pytest.mark.unit
@pytest.mark.parametrize(('loc', 'wid'), [(0.4, 0.2), (0.5, 0.1), (0.1, 0.2), (0.9, 0.2)])
def test_a_band_inside_zero_to_one_is_accepted(loc, wid):
    ie = _interior(loc=loc, wid=wid)
    assert ie.aragog.core_module.ra_crit_cmb == 'phase'


@pytest.mark.unit
@pytest.mark.parametrize(('loc', 'wid'), [(0.05, 0.2), (0.95, 0.2), (0.3, 0.8)])
def test_a_band_that_leaves_zero_to_one_is_refused(loc, wid):
    """The law by phase blends across [rfront_loc - rfront_wid/2, rfront_loc + rfront_wid/2];
    that band must lie inside [0, 1]."""
    with pytest.raises(ValueError, match='rfront_wid'):
        _interior(loc=loc, wid=wid)


@pytest.mark.unit
@pytest.mark.parametrize(
    'kw',
    [
        dict(ra='eq16'),
        dict(ra=450.0),
        dict(core_bc='energy_balance'),
        dict(module='spider'),
    ],
)
def test_the_band_is_checked_only_where_the_law_by_phase_acts(kw):
    _interior(loc=0.05, wid=0.2, **kw)


@pytest.mark.unit
def test_the_law_by_phase_reaches_aragog_as_phase():
    from types import SimpleNamespace

    from proteus.interior_energetics.aragog_core import core_module_params

    config = SimpleNamespace(
        interior_energetics=SimpleNamespace(
            aragog=SimpleNamespace(core_module=AragogCoreModule(ra_crit_cmb='phase'))
        ),
        params=SimpleNamespace(resume=False),
    )
    params = core_module_params(
        config, {'M_core': 1.93e24, 'P_center': 3.6e11}, SimpleNamespace(), '.'
    )
    assert params['ra_crit_cmb'] == 'phase'
