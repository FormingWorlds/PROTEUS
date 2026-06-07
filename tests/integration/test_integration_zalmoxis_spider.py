"""Integration test: zalmoxis (real interior structure) coupled to
SPIDER (real interior energetics, C code via subprocess).

The (interior_struct=zalmoxis, interior_energetics=spider) pair is
the production interior stack alternate to (zalmoxis, aragog).
Zalmoxis solves the mass-radius problem and writes a SPIDER mesh
file via ``num_spider_nodes > 0`` so SPIDER can read the external
mesh on the next iteration. SPIDER's ``-rho_core`` echo path can
overwrite ``core_density`` after the first call; ``core_heatcap``
flows the same way.

The mirror configuration with ``interior_struct.module='spider'``
(no Zalmoxis structure solve, SPIDER-only structure) is also
exercised here for its cross-validator rejections: ``core_frac_mode
= 'mass'``, ``core_density = 'self'``, and ``core_heatcap = 'self'``
are valid only with ``module='zalmoxis'``. ``melting_dir`` and
``eos_dir`` are required when ``module='spider'`` but optional when
``module='zalmoxis'`` (Zalmoxis derives its EOS from its own
config).

Integration-tier scope:

- The (zalmoxis, spider) pair round-trips through Config without
  raising. Both modules end up on their documented backends.
- ``Spider.solver_type`` enum is pinned as ``{'adams', 'bdf'}``
  with the documented default ``'bdf'`` round-tripping.
- ``Spider.matprop_smooth_width`` uses ``(gt(0), lt(1))`` at the
  field layer; both endpoints reject; small positive and just-
  below-one round-trip.
- ``Spider.tolerance_struct`` uses ``gt(0)`` with both-edge
  selectivity at the minimum meaningful positive value; default 1e2
  round-trips.
- ``valid_spider`` rejects an Interior config with all four
  transport terms disabled; mirror of ``valid_aragog`` exercised in
  the zalmoxis_aragog pair.
- ``interior_struct.module='spider'`` cross-rejections in
  ``Struct.__attrs_post_init__``:
  - ``core_frac_mode='mass'`` rejects (SPIDER only supports radius
    mode).
  - ``core_density='self'`` and ``core_heatcap='self'`` both reject
    (the self-consistent EOS path is Zalmoxis-only).
  - ``melting_dir=None`` rejects with a message naming the
    ``FWL_DATA/interior_lookup_tables/Melting_curves`` path.
  - ``eos_dir=None`` rejects with a message naming the
    ``FWL_DATA/interior_lookup_tables/EOS/dynamic`` path.
- ``valid_zalmoxis`` early-returns when
  ``interior_struct.module='spider'``: the EOS-format and 2-layer
  cross-rules do not fire on a SPIDER-mode config even if the
  Zalmoxis sub-fields would otherwise reject. Field-level
  validators on Zalmoxis (mushy_zone_factor range, gt(0)
  tolerances) STILL fire because they live on the field, not the
  cross-validator.
- ``Zalmoxis.lookup_nP`` / ``lookup_nS`` validators
  (``ge(100)``/``ge(50)``) gate the SPIDER P-S table resolution
  Zalmoxis generates from PALEOS. Edges pinned both ways.
- The wrapper merge contract registers the Zalmoxis -> SPIDER
  hand-off columns (``R_int``, ``M_int``, ``R_core``, ``P_cmb``,
  ``core_density``, ``core_heatcap``, ``T_cmb_initial``, ``F_cmb``).

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs():
    """Base Config kwargs for the (zalmoxis, spider) combination.

    ``atmos_clim='dummy'`` avoids the ``janus_escape_atmosphere``
    cross-validator. The fO2 buffer goes through ``ic_chemistry`` so
    no outgas backend beyond dummy is implicitly required.
    """
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._outgas import Outgas
    from proteus.config._planet import Elements, Planet
    from proteus.config._star import Star, StarDummy

    return dict(
        atmos_clim=AtmosClim(module='dummy', rayleigh=False),
        escape=Escape(module='dummy'),
        outgas=Outgas(module='dummy'),
        star=Star(module='dummy', dummy=StarDummy(calculate_radius=True)),
        planet=Planet(mass_tot=1.0, elements=Elements(O_mode='ic_chemistry')),
    )


def _spider_struct_kwargs():
    """Auxiliary fields a Struct(module='spider') needs to pass its
    ``__attrs_post_init__``: ``core_frac_mode='radius'`` (Spider
    rejects 'mass'), numeric ``core_density``/``core_heatcap``
    (Spider rejects 'self'), and the FWL_DATA-relative lookup
    folders.
    """
    return dict(
        core_frac_mode='radius',
        core_density=5500.0,
        core_heatcap=880.0,
        melting_dir='Monteux-600',
        eos_dir='WolfBower2018_MgSiO3',
    )


# ---------------------------------------------------------------------------
# (zalmoxis, spider) pair round-trip.
# ---------------------------------------------------------------------------


def test_zalmoxis_spider_round_trips_through_config():
    """The (interior_struct=zalmoxis, interior_energetics=spider) pair
    round-trips through Config without raising. Zalmoxis stays on its
    documented defaults (newton outer solver, PALEOS EOS); SPIDER
    lands on its documented default solver type ('bdf').
    """
    from proteus.config import Config
    from proteus.config._interior import Interior
    from proteus.config._struct import Struct

    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        interior_energetics=Interior(module='spider'),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.module == 'zalmoxis'
    assert cfg.interior_energetics.module == 'spider'
    assert cfg.interior_energetics.spider.solver_type == 'bdf'
    assert cfg.interior_struct.zalmoxis.outer_solver == 'newton'


# ---------------------------------------------------------------------------
# Spider field validators.
# ---------------------------------------------------------------------------


def test_spider_solver_type_enum_pinned_as_set_under_zalmoxis_pair():
    """Pin ``Spider.solver_type`` as ``{'adams', 'bdf'}``. The
    documented default 'bdf' round-trips.

    Set equality catches a regression that silently added a third
    SUNDIALS integrator while keeping the documented values
    accepted.
    """
    import attrs

    from proteus.config._interior import Spider

    allowed = attrs.fields(Spider).solver_type.validator.options
    assert set(allowed) == {'adams', 'bdf'}, (
        f'Spider.solver_type enum drifted from documented set: {allowed}'
    )
    for known in ('adams', 'bdf'):
        s = Spider(solver_type=known)
        assert s.solver_type == known
    with pytest.raises(ValueError, match=r'(?i)solver_type'):
        Spider(solver_type='rk4')
    assert Spider().solver_type == 'bdf'


def test_spider_matprop_smooth_width_open_interval_selectivity():
    """``Spider.matprop_smooth_width`` uses ``(gt(0), lt(1))``: both
    endpoints reject, small positive and just-below-one round-trip.
    Default 1e-2 round-trips.

    Edge: pin both ends of the open interval (catches both ``ge(0)``
    and ``le(1)`` regressions).
    """
    from proteus.config._interior import Spider

    for bad_val in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r'(?i)matprop_smooth_width'):
            Spider(matprop_smooth_width=bad_val)
    s = Spider(matprop_smooth_width=1e-12)
    assert s.matprop_smooth_width == pytest.approx(1e-12, rel=1e-12)
    s_hi = Spider(matprop_smooth_width=0.999)
    assert s_hi.matprop_smooth_width == pytest.approx(0.999, rel=1e-12)
    assert Spider().matprop_smooth_width == pytest.approx(1e-2, rel=1e-12)


def test_spider_tolerance_struct_gt0_with_selectivity():
    """``Spider.tolerance_struct`` uses ``gt(0)``. ``=0`` and
    negative reject; small positive round-trips; the documented
    default 1e2 round-trips.

    The same field exists on Aragog with identical validator; pinning
    here documents the SPIDER side of the matched-pair contract.
    """
    from proteus.config._interior import Spider

    for bad_val in (0.0, -1.0):
        with pytest.raises(ValueError, match=r'(?i)tolerance_struct'):
            Spider(tolerance_struct=bad_val)
    s = Spider(tolerance_struct=1e-12)
    assert s.tolerance_struct == pytest.approx(1e-12, rel=1e-12)
    assert Spider().tolerance_struct == pytest.approx(1e2, rel=1e-12)


def test_spider_requires_at_least_one_energy_transport_term_under_zalmoxis():
    """``valid_spider`` rejects an Interior config with all four
    ``trans_*`` flags False. Mirror of ``valid_aragog`` exercised in
    the zalmoxis_aragog pair: SPIDER has the same transport-term
    requirement.

    Edge: round-trip a single-term config (only convection) and
    reject the all-off case.
    """
    from proteus.config import Config
    from proteus.config._interior import Interior
    from proteus.config._struct import Struct

    with pytest.raises(ValueError, match=r'(?i)transport'):
        Config(
            interior_struct=Struct(module='zalmoxis'),
            interior_energetics=Interior(
                module='spider',
                trans_conduction=False,
                trans_convection=False,
                trans_mixing=False,
                trans_grav_sep=False,
            ),
            **_base_config_kwargs(),
        )
    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        interior_energetics=Interior(
            module='spider',
            trans_conduction=False,
            trans_convection=True,
            trans_mixing=False,
            trans_grav_sep=False,
        ),
        **_base_config_kwargs(),
    )
    assert cfg.interior_energetics.trans_convection is True
    assert cfg.interior_energetics.trans_conduction is False


# ---------------------------------------------------------------------------
# interior_struct.module='spider' cross-rejections.
# ---------------------------------------------------------------------------


def test_struct_spider_rejects_core_frac_mode_mass():
    """``Struct.__attrs_post_init__`` rejects
    ``core_frac_mode='mass'`` when ``module='spider'``. SPIDER only
    supports radius-based core fractions.

    Edge: confirm the SPIDER+radius mirror round-trips so the
    rejection is selective.
    """
    from proteus.config._struct import Struct

    with pytest.raises(ValueError, match=r'(?i)core_frac_mode'):
        Struct(
            module='spider',
            core_frac_mode='mass',
            core_density=5500.0,
            core_heatcap=880.0,
            melting_dir='Monteux-600',
            eos_dir='WolfBower2018_MgSiO3',
        )
    s = Struct(module='spider', **_spider_struct_kwargs())
    assert s.module == 'spider'
    assert s.core_frac_mode == 'radius'


def test_struct_spider_rejects_self_sentinels_on_core_density_and_heatcap():
    """``Struct.__attrs_post_init__`` rejects ``core_density='self'``
    and ``core_heatcap='self'`` when ``module='spider'``. The
    self-consistent EOS path is delegated to Zalmoxis; SPIDER
    requires explicit numeric values.

    Discrimination: both fields tested separately. The all-numeric
    SPIDER config round-trips so the rejection is selective.
    """
    from proteus.config._struct import Struct

    # core_density='self' rejects with spider.
    with pytest.raises(ValueError, match=r'(?i)core_density'):
        Struct(
            module='spider',
            core_frac_mode='radius',
            core_density='self',
            core_heatcap=880.0,
            melting_dir='Monteux-600',
            eos_dir='WolfBower2018_MgSiO3',
        )
    # core_heatcap='self' rejects with spider.
    with pytest.raises(ValueError, match=r'(?i)core_heatcap'):
        Struct(
            module='spider',
            core_frac_mode='radius',
            core_density=5500.0,
            core_heatcap='self',
            melting_dir='Monteux-600',
            eos_dir='WolfBower2018_MgSiO3',
        )
    # All-numeric round-trips.
    s = Struct(module='spider', **_spider_struct_kwargs())
    assert s.core_density == pytest.approx(5500.0, rel=1e-12)
    assert s.core_heatcap == pytest.approx(880.0, rel=1e-12)


def test_struct_spider_requires_melting_dir_and_eos_dir():
    """``Struct.__attrs_post_init__`` rejects ``module='spider'``
    without explicit ``melting_dir`` and ``eos_dir`` folder names.
    The error messages name the ``FWL_DATA/interior_lookup_tables``
    sub-paths so the user knows where to find a valid value.

    Mirror: ``module='zalmoxis'`` round-trips with ``melting_dir =
    None`` and ``eos_dir = None`` because Zalmoxis derives its EOS
    from its own config.
    """
    from proteus.config._struct import Struct

    # SPIDER rejects None on either folder.
    with pytest.raises(ValueError, match=r'(?i)melting_dir'):
        Struct(
            module='spider',
            core_frac_mode='radius',
            core_density=5500.0,
            core_heatcap=880.0,
            eos_dir='WolfBower2018_MgSiO3',
        )
    with pytest.raises(ValueError, match=r'(?i)eos_dir'):
        Struct(
            module='spider',
            core_frac_mode='radius',
            core_density=5500.0,
            core_heatcap=880.0,
            melting_dir='Monteux-600',
        )
    # Zalmoxis round-trips with both folders None.
    s = Struct(module='zalmoxis')
    assert s.melting_dir is None
    assert s.eos_dir is None


# ---------------------------------------------------------------------------
# valid_zalmoxis early-return when interior_struct.module='spider'.
# ---------------------------------------------------------------------------


def test_valid_zalmoxis_does_not_fire_when_module_is_spider():
    """``valid_zalmoxis`` short-circuits via ``if instance.module ==
    'spider': return`` at the top of the function. Cross-validator
    rules on ``mantle_mass_fraction``, EOS format, and 2-layer
    geometry therefore do NOT fire on a SPIDER-mode Struct.

    Discrimination: construct a Struct that would fail the 2-layer
    cross-validator under ``module='zalmoxis'`` (non-zero
    mantle_mass_fraction with a non-T-dep mantle) BUT lift to
    ``module='spider'`` with the spider auxiliaries. The Struct
    constructs successfully; the Zalmoxis sub-fields are stashed
    but never validated by valid_zalmoxis.
    """
    from proteus.config._struct import Struct, Zalmoxis

    # Build the SPIDER struct first so we know the auxiliaries pass.
    s_ok = Struct(module='spider', **_spider_struct_kwargs())
    assert s_ok.module == 'spider'

    # The same Zalmoxis sub-config that would fail valid_zalmoxis
    # under module='zalmoxis' (mantle_mass_fraction != 0 with a
    # non-T-dep PALEOS:MgSiO3 mantle) is accepted under
    # module='spider' because the cross-validator early-returns.
    s_spider = Struct(
        module='spider',
        zalmoxis=Zalmoxis(
            mantle_eos='PALEOS:MgSiO3',
            mantle_mass_fraction=0.4,
        ),
        **_spider_struct_kwargs(),
    )
    assert s_spider.module == 'spider'
    assert s_spider.zalmoxis.mantle_mass_fraction == pytest.approx(0.4, rel=1e-12)


def test_field_level_zalmoxis_validators_fire_independently_of_module():
    """Field-level validators on the Zalmoxis sub-config fire on the
    Zalmoxis instance itself, regardless of the parent
    ``Struct.module`` value. ``mushy_zone_factor`` rejects values
    outside ``[0.7, 1.0]`` even when the Struct it lives on has
    ``module='spider'``.

    Discrimination: the field-level rejection happens at Zalmoxis
    construction time, before the Struct sees the value. Cross-
    validators (``valid_zalmoxis``) need not fire for the field
    validators to act.
    """
    from proteus.config._struct import Zalmoxis

    with pytest.raises(ValueError, match=r'(?i)mushy_zone_factor'):
        Zalmoxis(mushy_zone_factor=0.5)
    # Adjacent-valid: the lower bound 0.7 must round-trip.
    z = Zalmoxis(mushy_zone_factor=0.7)
    assert z.mushy_zone_factor == pytest.approx(0.7, rel=1e-12)


# ---------------------------------------------------------------------------
# SPIDER P-S table resolution (Zalmoxis-generated).
# ---------------------------------------------------------------------------


def test_zalmoxis_lookup_resolutions_for_spider_table_generation():
    """When ``interior_struct.module='zalmoxis'`` and
    ``interior_energetics.module='spider'``, Zalmoxis generates the
    P-S lookup tables that SPIDER reads at startup. ``lookup_nP`` and
    ``lookup_nS`` validators set the floor on these resolutions
    (``ge(100)``, ``ge(50)``) so a misconfigured run does not feed
    SPIDER a coarse table that would silently distort phase
    boundaries.

    Edge: pin both edges of each floor.
    """
    from proteus.config._struct import Zalmoxis

    # lookup_nP floor.
    assert Zalmoxis(lookup_nP=100).lookup_nP == 100
    with pytest.raises(ValueError, match=r'(?i)lookup_nP'):
        Zalmoxis(lookup_nP=99)
    # lookup_nS floor.
    assert Zalmoxis(lookup_nS=50).lookup_nS == 50
    with pytest.raises(ValueError, match=r'(?i)lookup_nS'):
        Zalmoxis(lookup_nS=49)
    # Defaults match the production paper-pair values.
    default = Zalmoxis()
    assert default.lookup_nP == 1350
    assert default.lookup_nS == 280


# ---------------------------------------------------------------------------
# Wrapper-merge contract: Zalmoxis -> SPIDER hand-off + SPIDER outputs.
# ---------------------------------------------------------------------------


def test_zalmoxis_spider_helpfile_keys_register_structure_and_cmb_handoff():
    """The Zalmoxis -> SPIDER hand-off populates structure-side
    columns in ``hf_row`` that SPIDER reads at the next iteration.
    SPIDER writes back the CMB-side energy state.

    Required structure-side columns (Zalmoxis): ``R_int``, ``M_int``,
    ``R_core``, ``M_core``, ``M_mantle``, ``M_planet``, ``P_center``,
    ``P_cmb``, ``core_density``, ``core_heatcap``.
    Required CMB-side columns (SPIDER writes / loop reads back):
    ``T_cmb``, ``T_cmb_initial``, ``T_magma``, ``F_cmb``, ``F_int``.
    Required melt-state columns: ``Phi_global``, ``Phi_global_vol``,
    ``M_mantle_solid``, ``M_mantle_liquid``, ``RF_depth``.

    Discrimination: every key checked separately so a regression
    that dropped any one fails the per-key loop. ``ZeroHelpfileRow``
    seeds each as a float zero.
    """
    from proteus.utils.coupler import GetHelpfileKeys, ZeroHelpfileRow

    keys = GetHelpfileKeys()
    structure_keys = (
        'R_int',
        'M_int',
        'R_core',
        'M_core',
        'M_mantle',
        'M_planet',
        'P_center',
        'P_cmb',
        'core_density',
        'core_heatcap',
    )
    cmb_keys = ('T_cmb', 'T_cmb_initial', 'T_magma', 'F_cmb', 'F_int')
    melt_keys = (
        'Phi_global',
        'Phi_global_vol',
        'M_mantle_solid',
        'M_mantle_liquid',
        'RF_depth',
    )
    for key in structure_keys + cmb_keys + melt_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in structure_keys + cmb_keys + melt_keys:
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
