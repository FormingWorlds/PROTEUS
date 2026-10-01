"""Unit tests for interior-structure config validation (config/_struct.py).

Covers the cross-field guards in ``Struct.__attrs_post_init__`` and the
``Zalmoxis`` update-interval ordering guard. These are config-validation
(utility) tests: they assert the documented error contracts and that a
valid neighbouring configuration is accepted, so the rejection is driven
by the specific invalid field and not a blanket raise.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging

import pytest

from proteus.config._struct import Struct, Zalmoxis

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _spider_kwargs(**overrides):
    """Spider Struct kwargs that pass the earlier guards (radius core-frac,
    numeric core density/heatcap) so a later guard under test is reached."""
    base = {
        'module': 'spider',
        'core_frac_mode': 'radius',
        'core_density': 1.0e4,
        'core_heatcap': 1.0e3,
        'melting_dir': 'Monteux-600',
        'eos_dir': 'WolfBower2018_MgSiO3',
    }
    base.update(overrides)
    return base


class TestZalmoxisUpdateInterval:
    """Zalmoxis structure-refresh interval ordering."""

    def test_min_interval_above_interval_is_rejected(self):
        """`update_min_interval` above `update_interval` would let the floor
        block every refresh before the ceiling can fire, so it is rejected."""
        with pytest.raises(ValueError, match='update_min_interval'):
            Zalmoxis(update_interval=10.0, update_min_interval=20.0)
        # Discrimination: min == interval is the boundary and must be allowed,
        # so the guard is a strict ordering check, not a blanket rejection of
        # a set min_interval.
        z = Zalmoxis(update_interval=10.0, update_min_interval=10.0)
        assert z.update_min_interval == pytest.approx(10.0)

    def test_disabled_refresh_skips_the_ordering_guard(self):
        """With `update_interval = 0` (refresh disabled) the ordering guard is
        inert: any `update_min_interval` is accepted because no refresh fires."""
        z = Zalmoxis(update_interval=0.0, update_min_interval=5.0)
        # Edge case: interval 0 is the off switch; the otherwise-illegal
        # min > interval combination must not raise here.
        assert z.update_interval == pytest.approx(0.0)
        assert z.update_min_interval == pytest.approx(5.0)


class TestStructSpiderGuards:
    """Cross-field guards that constrain the spider structure backend."""

    def test_mass_core_frac_requires_zalmoxis(self):
        """`core_frac_mode = "mass"` is only supported by zalmoxis; pairing it
        with spider raises."""
        with pytest.raises(ValueError, match='core_frac_mode = "mass"'):
            Struct(module='spider', core_frac_mode='mass')
        # Discrimination: the same mass mode with zalmoxis is the valid
        # configuration, so the rejection is the spider pairing, not mass mode.
        s = Struct(module='zalmoxis', core_frac_mode='mass')
        assert s.core_frac_mode == 'mass'

    def test_self_core_density_requires_zalmoxis(self):
        """`core_density = "self"` defers to Zalmoxis' structure solve; with
        spider it raises because spider needs a numeric density."""
        with pytest.raises(ValueError, match='core_density'):
            Struct(**_spider_kwargs(core_density='self'))
        # Discrimination: "self" is valid under zalmoxis (the default module),
        # so the rejection is spider-specific.
        s = Struct(module='zalmoxis', core_density='self')
        assert s.core_density == 'self'

    def test_non_positive_core_heatcap_is_rejected(self):
        """A numeric core heat capacity must be positive; zero or negative
        raises regardless of module."""
        with pytest.raises(ValueError, match='core_heatcap'):
            Struct(module='zalmoxis', core_heatcap=-5.0)
        # Discrimination: a positive value of the same field constructs, so the
        # guard is a sign/positivity check, not a rejection of any numeric value.
        s = Struct(module='zalmoxis', core_heatcap=1.0e3)
        assert s.core_heatcap == pytest.approx(1.0e3)

    def test_spider_requires_melting_dir(self):
        """The spider backend reads melting curves from FWL_DATA, so an unset
        `melting_dir` raises."""
        with pytest.raises(ValueError, match='melting_dir'):
            Struct(**_spider_kwargs(melting_dir=None))
        # Discrimination: with melting_dir provided the spider config is
        # accepted, so the rejection is the missing directory specifically.
        s = Struct(**_spider_kwargs())
        assert s.melting_dir == 'Monteux-600'

    def test_spider_requires_eos_dir(self):
        """The spider backend reads its EOS tables from FWL_DATA, so an unset
        `eos_dir` raises once melting_dir is provided."""
        with pytest.raises(ValueError, match='eos_dir'):
            Struct(**_spider_kwargs(eos_dir=None))
        # Discrimination: with both directories set the spider config is
        # accepted, so the rejection is the missing eos_dir specifically.
        s = Struct(**_spider_kwargs())
        assert s.module == 'spider'
        assert s.eos_dir == 'WolfBower2018_MgSiO3'


class TestZalmoxisVolatileGates:
    """Gates on the dissolved-volatile structure path.

    Phase-aware volatile mixing (`dry_mantle = false`) is now supported:
    the pinned Zalmoxis release evaluates a per-shell volatile profile in
    the mantle density. Binodal-aware miscibility (`global_miscibility`)
    still requires the H2-silicate binodal handoff on the Zalmoxis side
    (Zalmoxis tracker #64), so it must still fail loudly at config load
    instead of silently doing nothing at runtime.
    """

    def test_global_miscibility_is_rejected(self):
        """`global_miscibility = true` still raises: the H2-silicate binodal
        handoff it needs is not yet implemented on the Zalmoxis side."""
        with pytest.raises(ValueError, match='global_miscibility'):
            Struct(module='zalmoxis', zalmoxis=Zalmoxis(global_miscibility=True))
        # Discrimination: the default (miscibility off) constructs, so the
        # rejection is the flag, not the zalmoxis module itself.
        s = Struct(module='zalmoxis')
        assert s.zalmoxis.global_miscibility is False

    def test_wet_mantle_is_accepted(self):
        """`dry_mantle = false` now constructs: the gate was lifted once the
        pinned Zalmoxis release gained per-shell volatile-profile support."""
        s = Struct(module='zalmoxis', zalmoxis=Zalmoxis(dry_mantle=False))
        assert s.zalmoxis.dry_mantle is False
        # Retro-compat: the default remains dry, byte-identical to baseline.
        assert Struct(module='zalmoxis').zalmoxis.dry_mantle is True

    @pytest.mark.parametrize('module', ['spider', 'dummy', 'zalmoxis'])
    def test_global_miscibility_is_rejected_for_every_structure(self, module):
        """`global_miscibility = true` is rejected for every structure module:
        only the zalmoxis structure writes the solvus that the main loop and
        SPIDER read, so under any other module the solvus frame would start
        from the zero-initialised helpfile values."""
        kwargs = _spider_kwargs() if module == 'spider' else {'module': module}
        with pytest.raises(ValueError, match='global_miscibility'):
            Struct(**kwargs, zalmoxis=Zalmoxis(global_miscibility=True))
        # Discrimination: the same structure with the flag off constructs.
        s = Struct(**kwargs, zalmoxis=Zalmoxis(global_miscibility=False))
        assert s.module == module
        assert s.zalmoxis.global_miscibility is False

    def test_spider_module_skips_the_eos_format_check(self):
        """Under spider the zalmoxis EOS strings are inert: an EOS string that
        fails the zalmoxis format check still constructs. The miscibility
        rejection is the one check that applies to every module."""
        s = Struct(**_spider_kwargs(zalmoxis=Zalmoxis(core_eos='no_colon')))
        assert s.zalmoxis.core_eos == 'no_colon'
        # The paired negative: zalmoxis does enforce the format, so acceptance
        # above is the module skipping the check rather than the check being
        # absent.
        with pytest.raises(ValueError, match='core_eos'):
            Struct(module='zalmoxis', zalmoxis=Zalmoxis(core_eos='no_colon'))


class TestZalmoxisMushyZoneWarning:
    """The mushy_zone_factor no-effect warning must fire only for the EOS
    families that actually ignore the factor.

    mushy_zone_factor scales the derived solidus ``T_sol = T_liq * mzf`` for
    the PALEOS family (unified, 2-phase, and the API variants) through
    ``load_zalmoxis_solidus_liquidus_functions``. For WolfBower2018 and
    RTPress100TPa the melting curves come from file and the factor is inert,
    so there the warning is correct and must still fire.
    """

    @staticmethod
    def _warns(caplog, mantle_eos, mzf=0.8, core_eos='Seager2007:iron', ice_layer_eos=None):
        """Construct a zalmoxis Struct and report whether the no-effect
        warning fired for the given layer EOS at the given factor.

        The core defaults to a file-curve EOS so that ``mantle_eos`` alone
        decides the outcome."""
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.config._struct'):
            Struct(
                module='zalmoxis',
                zalmoxis=Zalmoxis(
                    core_eos=core_eos,
                    mantle_eos=mantle_eos,
                    ice_layer_eos=ice_layer_eos,
                    mushy_zone_factor=mzf,
                ),
            )
        return any('has no effect' in r.getMessage() for r in caplog.records)

    def test_warning_silent_for_the_paleos_family(self, caplog):
        """Each PALEOS-family EOS feeds the factor into the derived solidus, so
        the no-effect warning must stay silent for all four."""
        for eos in (
            'PALEOS:MgSiO3',
            'PALEOS-2phase:MgSiO3',
            'PALEOS-API:MgSiO3',
            'PALEOS-API-2phase:MgSiO3',
        ):
            assert not self._warns(caplog, eos), eos

    def test_warning_fires_for_file_curve_eos(self, caplog):
        """WolfBower2018 and RTPress100TPa read melting curves from file, so the
        factor has no effect and the warning must fire."""
        for eos in ('WolfBower2018:MgSiO3', 'RTPress100TPa:MgSiO3'):
            assert self._warns(caplog, eos), eos

    def test_warning_silent_when_only_the_core_is_paleos(self, caplog):
        """A unified PALEOS core density depends on the factor, so a
        WolfBower2018 mantle above it does not make the setting inert."""
        assert not self._warns(caplog, 'WolfBower2018:MgSiO3', core_eos='PALEOS:iron')

    def test_warning_silent_when_only_the_ice_layer_is_paleos(self, caplog):
        """A PALEOS ice layer also honors the factor."""
        assert not self._warns(
            caplog,
            'WolfBower2018:MgSiO3',
            ice_layer_eos='PALEOS:H2O',
        )

    def test_warning_fires_when_no_layer_is_paleos(self, caplog):
        """Paired negative: with no PALEOS layer anywhere, the warning fires."""
        assert self._warns(
            caplog,
            'WolfBower2018:MgSiO3',
            core_eos='Seager2007:iron',
            ice_layer_eos='Seager2007:H2O',
        )

    def test_warning_tracks_the_factor_not_only_the_eos(self, caplog):
        """The warning depends on mushy_zone_factor < 1: at the sharp-boundary
        value 1.0 an ignored setting is not misreported."""
        assert not self._warns(caplog, 'WolfBower2018:MgSiO3', mzf=1.0)
        # Paired positive: the same EOS at 0.8 fires, so silence at 1.0 is the
        # factor guard rather than the EOS being exempt.
        assert self._warns(caplog, 'WolfBower2018:MgSiO3', mzf=0.8)


class TestMeltingDirWithPaleos:
    """melting_dir is not read with Zalmoxis and a PALEOS mantle EOS."""

    @staticmethod
    def _warned(caplog, **kwargs):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.config._struct'):
            Struct(**kwargs)
        return any('is not read' in r.getMessage() for r in caplog.records)

    def test_warning_for_melting_dir_with_paleos_mantle(self, caplog):
        """The PALEOS curves are used, so a configured melting_dir gives one warning."""
        assert self._warned(caplog, module='zalmoxis', melting_dir='Monteux-600')
        assert sum('is not read' in r.getMessage() for r in caplog.records) == 1

    def test_no_warning_where_melting_dir_is_read(self, caplog):
        """Discrimination: a file-curve mantle EOS or an unset melting_dir stays silent."""
        wb = Zalmoxis(mantle_eos='WolfBower2018:MgSiO3')
        assert not self._warned(
            caplog, module='zalmoxis', zalmoxis=wb, melting_dir='Monteux-600'
        )
        assert not self._warned(caplog, module='zalmoxis', melting_dir=None)

    def test_warning_follows_the_mgsio3_component_of_a_mixture(self, caplog):
        """A PALEOS mixture gets the generated set, so melting_dir is not read; a
        mixture with a Wolf and Bower MgSiO3 component reads it."""
        mix = Zalmoxis(mantle_eos='PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1')
        assert self._warned(caplog, module='zalmoxis', zalmoxis=mix, melting_dir='Monteux-600')
        wb = Zalmoxis(mantle_eos='WolfBower2018:MgSiO3:0.9+PALEOS:H2O:0.1')
        assert not self._warned(
            caplog, module='zalmoxis', zalmoxis=wb, melting_dir='Monteux-600'
        )
        single = Zalmoxis(mantle_eos='PALEOS-2phase:MgSiO3')
        assert self._warned(
            caplog, module='zalmoxis', zalmoxis=single, melting_dir='Monteux-600'
        )


@pytest.mark.parametrize(
    'mantle, rejected',
    [
        ('PALEOS:MgSiO3:0.5+WolfBower2018:MgSiO3:0.5', True),
        ('PALEOS:MgSiO3:0.5+PALEOS-2phase:MgSiO3:0.5', True),
        ('PALEOS:MgSiO3:0.5+PALEOS:MgSiO3:0.5', False),
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1', False),
    ],
)
def test_two_mgsio3_sources_are_rejected_at_load(mantle, rejected):
    """A mixture with MgSiO3 components of different keys has no single melting curve,
    so it is rejected, naming both; one key repeated is one material."""
    kwargs = dict(module='zalmoxis', zalmoxis=Zalmoxis(mantle_eos=mantle))
    if rejected:
        with pytest.raises(ValueError, match='MgSiO3 components from different sources') as exc:
            Struct(**kwargs)
        for key in mantle.replace(':0.5', '').split('+'):
            assert key in str(exc.value)
    else:
        Struct(**kwargs)


@pytest.mark.parametrize(
    'mantle, rejected',
    [
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:inf', True),
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:-Infinity', True),
        ('PALEOS:MgSiO3:0.9+PALEOS: H2O', True),
        ('PALEOS:MgSiO3:0.9+H2O:0.1', True),
        ('PALEOS:MgSiO3:0.9+:H2O:0.1', True),
        ('PALEOS:MgSiO3:0.9+PALEOS::0.1', True),
        ('PALEOS:MgSiO3:1.1+PALEOS:H2O:-0.1', True),
        ('PALEOS:MgSiO3:', True),
        ('PALEOS::MgSiO3', True),
        ('PALEOS:MgSiO3:foo', True),
        ('PALEOS:MgSiO3:bar:0.5', True),
        ('PALEOS:MgSiO3:0.5:0.5', True),
        ('PALEOS:MgSiO3: 0.9', True),
        ('PALEOS:nan', True),
        ('PALEOS:MgSiO3:nan', True),
        ('0.5:MgSiO3', True),
        ('PALEOS:MgSiO3:0.9+', True),
        ('PALEOS:MgSiO3:0.9++PALEOS:H2O:0.1', True),
        ('PALEOS:MgSiO3:0.9 + PALEOS:H2O:0.1', False),
        ('PALEOS:MgSiO3:0', True),
        ('PALEOS:MgSiO3:0+PALEOS:H2O:0.1', True),
        ('PALEOS:MgSiO3', False),
    ],
)
def test_an_eos_component_with_a_space_or_a_non_finite_fraction_is_rejected(mantle, rejected):
    """A component that is not '<source>:<material>' with at most one positive finite
    fraction, or that has an inner space, names no registry key or no valid mass fraction for
    Zalmoxis, so it is rejected at load; a zero fraction would leave a mixture member out of
    the structure while the energetics still follow it. Spaces around '+' are fine."""
    kwargs = dict(module='zalmoxis', zalmoxis=Zalmoxis(mantle_eos=mantle))
    if rejected:
        with pytest.raises(ValueError, match='with a positive finite fraction and no spaces'):
            Struct(**kwargs)
    else:
        assert Struct(**kwargs).zalmoxis.mantle_eos == mantle


@pytest.mark.parametrize('layer', ['core_eos', 'ice_layer_eos'])
def test_a_core_or_ice_component_is_checked_like_a_mantle_component(layer):
    """The core and ice layer EOS components follow the same format rule as the mantle."""
    with pytest.raises(ValueError, match=f'`interior_struct.zalmoxis.{layer}` component'):
        Struct(module='zalmoxis', zalmoxis=Zalmoxis(**{layer: 'PALEOS:H2O:foo'}))


def test_zalmoxis_structure_without_its_section_loads():
    """The MgSiO3-source check needs the Zalmoxis section and skips when it is absent."""
    assert Struct(module='zalmoxis', zalmoxis=None).zalmoxis is None
