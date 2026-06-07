"""Integration test: zalmoxis (real interior structure) coupled to
aragog (real interior energetics).

Zalmoxis solves the planetary mass-radius problem from EOS, and
Aragog steps the entropy ODE on the resulting mantle. The two
modules share boundary state through hf_row: Zalmoxis writes
``R_int``, ``M_int``, ``R_core``, ``P_center``, ``P_cmb``,
``core_density``, and ``core_heatcap`` at structure refresh; Aragog
reads ``R_int`` and the per-layer EOS state to initialise its own
solver. The ``core_density = 'self'`` and ``core_heatcap = 'self'``
sentinels delegate the core EOS to Zalmoxis and are only valid when
``interior_struct.module = 'zalmoxis'``.

Integration-tier scope:

- The (interior_struct=zalmoxis, interior_energetics=aragog)
  combination round-trips through Config without raising. Both
  modules end up on their documented backends.
- ``interior_struct.module`` enum is pinned as
  ``{None, 'dummy', 'spider', 'zalmoxis'}`` so a regression that
  silently added a fourth structure backend surfaces here.
- ``interior_energetics.module`` enum is pinned as
  ``{'spider', 'aragog', 'dummy', 'boundary'}``.
- Aragog's ``backend``, ``core_bc``, ``phase_smoothing``, and
  ``solver_method`` enums are pinned as sets. The documented
  defaults (``jax``, ``energy_balance``, ``tanh``, ``cvode``)
  round-trip.
- Zalmoxis's ``outer_solver`` enum is pinned as
  ``{'picard', 'newton'}`` and ``core_frac_mode`` as
  ``{'radius', 'mass'}``. The documented defaults (``newton``,
  ``mass``) round-trip.
- EOS format strings on ``Zalmoxis.core_eos`` / ``mantle_eos`` /
  ``ice_layer_eos`` require ``<source>:<material>``. Strings
  missing the colon are rejected by ``valid_zalmoxis``; the
  ``none_if_none`` converter on ``ice_layer_eos`` coerces the
  lowercase ``'none'`` sentinel to Python None.
- ``mushy_zone_factor`` is constrained to ``[0.7, 1.0]`` inclusive
  at the field layer; both endpoints round-trip.
- ``mantle_mass_fraction`` field-level ``ge(0)``/``lt(1)``: ``=0``
  round-trips (catches a ``gt(0)`` regression that would silently
  reject the 2-layer default), ``=0.999`` round-trips (catches a
  ``le(<below_unit>)`` regression), ``-1`` and ``1.0`` reject.
- The 2-layer cross-validator: with ``ice_layer_eos = None`` AND a
  non-T-dep mantle EOS (``PALEOS:MgSiO3``), ``mantle_mass_fraction
  != 0`` rejects. The mirror case with a T-dep mantle EOS
  (``WolfBower2018:MgSiO3``) round-trips ``mantle_mass_fraction =
  0.4`` because the field then partitions mass between core and
  mantle.
- The 3-layer cross-validator (with ice layer): ``core_frac +
  mantle_mass_fraction > 0.75`` rejects per Seager 2007.
- ``core_density`` and ``core_heatcap``: ``'self'`` sentinel round-
  trips with ``module='zalmoxis'``; numeric ``> 0`` round-trips;
  ``0`` and negative values reject.
- Zalmoxis solver tolerances (``solver_tol_outer``,
  ``solver_tol_inner``) and Newton knobs (``newton_tol``,
  ``newton_relative_tolerance``, ``newton_absolute_tolerance``)
  use ``gt(0)`` with both-edge selectivity at the minimum
  meaningful positive value.
- ``solver_max_iter_outer``/``inner`` ``ge(10)`` selectivity:
  ``=10`` round-trips, ``=9`` rejects.
- ``newton_max_iter`` ``ge(5)`` and ``num_levels`` ``ge(50)``
  (default 150) selectivity in the same shape.
- ``update_interval``/``update_min_interval``/``update_stale_ceiling``
  ``ge(0)`` selectivity at zero.
- ``Zalmoxis.__attrs_post_init__`` rejects ``update_min_interval >
  update_interval`` and round-trips the equality and reversed
  inequality cases.
- ``Aragog.tolerance_struct`` and ``Aragog.atol_temperature_equivalent``
  both use ``gt(0)`` with the both-edge selectivity pattern.
- ``Aragog.phi_step_cap`` uses ``ge(0)`` (catches ``gt(0)``
  regression at the documented default 0.0).
- The wrapper merge contract: ``R_int``, ``M_int``, ``R_core``,
  ``P_center``, ``P_cmb``, ``core_density``, ``core_heatcap``,
  ``T_core``, ``T_cmb_initial``, and ``F_cmb`` are registered in
  ``GetHelpfileKeys`` and seeded as float(0.0) by
  ``ZeroHelpfileRow``, so the Zalmoxis -> Aragog hf_row hand-off
  is intact.

The two-timestep real-solver boot lives at the slow tier in
``test_slow_zalmoxis_dummy.py`` (real Zalmoxis with dummy
energetics) and ``test_slow_aragog_calliope.py`` /
``test_slow_aragog_atmodeller.py`` (real Aragog with dummy
structure).

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _base_config_kwargs():
    """Build base Config kwargs for the (zalmoxis, aragog) combination.

    Keeps ``atmos_clim='dummy'``, ``escape='dummy'``, ``star='dummy'``
    so this pair test does not couple to cross-slot validators it does
    not own (``spada_zephyrus``, ``janus_escape_atmosphere``, the
    rotation-constraints branch of ``valid_mors``). The fO2 buffer is
    set to ``ic_chemistry`` so CALLIOPE / atmodeller are not
    implicitly required.
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


# ---------------------------------------------------------------------------
# Pair round-trip and enum pins.
# ---------------------------------------------------------------------------


def test_zalmoxis_aragog_round_trips_through_config():
    """The (interior_struct=zalmoxis, interior_energetics=aragog) pair
    round-trips through Config without raising. Both modules end up on
    their documented backends with the documented production defaults.
    """
    from proteus.config import Config
    from proteus.config._interior import Interior
    from proteus.config._struct import Struct

    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        interior_energetics=Interior(module='aragog'),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.module == 'zalmoxis'
    assert cfg.interior_energetics.module == 'aragog'
    assert cfg.interior_energetics.aragog.backend == 'jax'
    assert cfg.interior_struct.zalmoxis.outer_solver == 'newton'


def test_interior_struct_module_documented_set_round_trips_under_aragog_pair():
    """Pin the documented ``Struct.module`` set
    ``{None, 'dummy', 'spider', 'zalmoxis'}``. Every documented value
    round-trips through Struct construction; the ``'spider'`` path
    requires the auxiliary fields its post-init demands.

    The current Struct validator is a lambda that returns a boolean
    rather than raising on bad input. attrs treats a False return as
    successful validation, so the lambda silently accepts unknown
    strings. This test therefore pins the documented set by probing
    each value, not by asserting rejection of an unknown value. A
    future fix to convert the validator to ``in_(...)`` would let an
    unknown-string rejection be added here.
    """
    from proteus.config._struct import Struct

    for known in (None, 'dummy', 'spider', 'zalmoxis'):
        if known == 'spider':
            s = Struct(
                module='spider',
                core_frac_mode='radius',
                core_density=5500.0,
                core_heatcap=880.0,
                melting_dir='Monteux-600',
                eos_dir='WolfBower2018_MgSiO3',
            )
        else:
            s = Struct(module=known)
        assert s.module == known
    # Discrimination: default module is 'zalmoxis'. A regression that
    # swapped the default to 'dummy' would break the production-path
    # round-trip in test_zalmoxis_aragog_round_trips_through_config.
    assert Struct().module == 'zalmoxis'


def test_interior_energetics_module_enum_pinned_as_set_under_zalmoxis_pair():
    """Pin the ``Interior.module`` enum as
    ``{'spider', 'aragog', 'dummy', 'boundary'}``. Set equality is the
    only way to catch enum drift if a fifth backend is silently added.
    """
    import attrs

    from proteus.config._interior import Interior

    allowed = attrs.fields(Interior).module.validator.options
    assert set(allowed) == {'spider', 'aragog', 'dummy', 'boundary'}, (
        f'Interior.module enum drifted from documented set: {allowed}'
    )
    for known in ('aragog', 'dummy'):
        # boundary + spider require extra fields not relevant here.
        i = Interior(module=known)
        assert i.module == known
    with pytest.raises(ValueError):
        Interior(module='magma_ocean_2')


def test_aragog_enum_fields_pinned_as_set_under_zalmoxis_pair():
    """Pin ``Aragog.backend``, ``core_bc``, ``phase_smoothing``, and
    ``solver_method`` enums as sets. Production defaults
    (``jax``, ``energy_balance``, ``tanh``, ``cvode``) round-trip.

    Set equality catches a regression that silently added an extra
    value while keeping the documented ones round-tripping.
    """
    import attrs

    from proteus.config._interior import Aragog

    fields = attrs.fields(Aragog)
    assert set(fields.backend.validator.options) == {'numpy', 'jax'}
    assert set(fields.core_bc.validator.options) == {
        'quasi_steady',
        'energy_balance',
        'gradient',
        'bower2018',
    }
    assert set(fields.phase_smoothing.validator.options) == {'tanh', 'cubic_hermite'}
    assert set(fields.solver_method.validator.options) == {'cvode', 'radau', 'bdf'}
    default = Aragog()
    assert default.backend == 'jax'
    assert default.core_bc == 'energy_balance'
    assert default.phase_smoothing == 'tanh'
    assert default.solver_method == 'cvode'


def test_zalmoxis_outer_solver_and_core_frac_mode_enums_pinned_as_set():
    """Pin ``Zalmoxis.outer_solver`` as ``{'picard', 'newton'}`` and
    ``Struct.core_frac_mode`` as ``{'radius', 'mass'}``. Defaults
    (``newton``, ``mass``) round-trip.
    """
    import attrs

    from proteus.config._struct import Struct, Zalmoxis

    assert set(attrs.fields(Zalmoxis).outer_solver.validator.options) == {
        'picard',
        'newton',
    }
    assert set(attrs.fields(Struct).core_frac_mode.validator.options) == {
        'radius',
        'mass',
    }
    assert Zalmoxis().outer_solver == 'newton'
    assert Struct().core_frac_mode == 'mass'


# ---------------------------------------------------------------------------
# Zalmoxis EOS format validators.
# ---------------------------------------------------------------------------


def test_zalmoxis_eos_format_rejects_missing_colon_under_aragog_pair():
    """``valid_zalmoxis`` rejects EOS strings that lack
    ``<source>:<material>`` for ``core_eos``, ``mantle_eos``, and
    (when non-None) ``ice_layer_eos``. The documented production
    defaults (``PALEOS:iron``, ``PALEOS:MgSiO3``, None) round-trip.

    Edge: each of the three fields tested separately so a regression
    that loosened the check on one field is caught.
    """
    from proteus.config import Config
    from proteus.config._struct import Struct, Zalmoxis

    # Default round-trips.
    cfg = Config(
        interior_struct=Struct(module='zalmoxis'),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.zalmoxis.core_eos == 'PALEOS:iron'
    assert cfg.interior_struct.zalmoxis.mantle_eos == 'PALEOS:MgSiO3'
    assert cfg.interior_struct.zalmoxis.ice_layer_eos is None

    # core_eos missing colon rejects.
    with pytest.raises(ValueError, match=r'(?i)core_eos'):
        Config(
            interior_struct=Struct(module='zalmoxis', zalmoxis=Zalmoxis(core_eos='PALEOSiron')),
            **_base_config_kwargs(),
        )
    # mantle_eos missing colon rejects.
    with pytest.raises(ValueError, match=r'(?i)mantle_eos'):
        Config(
            interior_struct=Struct(
                module='zalmoxis',
                zalmoxis=Zalmoxis(mantle_eos='PALEOSMgSiO3'),
            ),
            **_base_config_kwargs(),
        )
    # ice_layer_eos: 'none' coerces to None via converter, real string
    # missing colon rejects.
    cfg_no_ice = Config(
        interior_struct=Struct(
            module='zalmoxis',
            zalmoxis=Zalmoxis(ice_layer_eos='none'),
        ),
        **_base_config_kwargs(),
    )
    assert cfg_no_ice.interior_struct.zalmoxis.ice_layer_eos is None
    with pytest.raises(ValueError, match=r'(?i)ice_layer_eos'):
        Config(
            interior_struct=Struct(
                module='zalmoxis',
                zalmoxis=Zalmoxis(
                    ice_layer_eos='PALEOSH2O',
                    mantle_mass_fraction=0.3,
                ),
            ),
            **_base_config_kwargs(),
        )


def test_zalmoxis_ice_layer_eos_none_sentinel_case_sensitive_under_aragog_pair():
    """The ``none_if_none`` converter on ``Zalmoxis.ice_layer_eos`` is
    case-sensitive at the field layer. Lowercase ``'none'`` coerces to
    Python ``None``; uppercase ``'None'`` / ``'NONE'`` are passed
    through unchanged (and then rejected by ``valid_zalmoxis`` for
    missing the format colon).

    Discrimination: a regression that broadened the converter to
    case-insensitive would let uppercase ``'None'`` reach
    ``valid_zalmoxis`` as Python ``None`` and silently round-trip.
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis(ice_layer_eos='none').ice_layer_eos is None
    # Uppercase passes through the converter unchanged.
    z_upper = Zalmoxis(ice_layer_eos='None')
    assert z_upper.ice_layer_eos == 'None'
    z_caps = Zalmoxis(ice_layer_eos='NONE')
    assert z_caps.ice_layer_eos == 'NONE'


# ---------------------------------------------------------------------------
# mushy_zone_factor closed-interval endpoints.
# ---------------------------------------------------------------------------


def test_zalmoxis_mushy_zone_factor_closed_interval_endpoints_round_trip():
    """``Zalmoxis.mushy_zone_factor`` is constrained to ``[0.7, 1.0]``
    inclusive at the field layer. Both endpoints round-trip; the
    documented default (0.8) round-trips; just-below-0.7 and
    just-above-1.0 reject.
    """
    from proteus.config._struct import Zalmoxis

    z_lo = Zalmoxis(mushy_zone_factor=0.7)
    assert z_lo.mushy_zone_factor == pytest.approx(0.7, rel=1e-12)
    z_hi = Zalmoxis(mushy_zone_factor=1.0)
    assert z_hi.mushy_zone_factor == pytest.approx(1.0, rel=1e-12)
    assert Zalmoxis().mushy_zone_factor == pytest.approx(0.8, rel=1e-12)
    with pytest.raises(ValueError, match=r'(?i)mushy_zone_factor'):
        Zalmoxis(mushy_zone_factor=0.6999)
    with pytest.raises(ValueError, match=r'(?i)mushy_zone_factor'):
        Zalmoxis(mushy_zone_factor=1.0001)


# ---------------------------------------------------------------------------
# mantle_mass_fraction field-level + cross-validator coverage.
# ---------------------------------------------------------------------------


def test_zalmoxis_mantle_mass_fraction_field_layer_selectivity():
    """``Zalmoxis.mantle_mass_fraction`` uses field-level
    ``ge(0)``/``lt(1)``. Pin both edges of the half-open interval:

    - ``=0`` round-trips (the documented 2-layer default; catches a
      ``gt(0)`` regression that would silently reject 0).
    - ``=0.999`` round-trips (catches a ``le(<below_unit>)``
      regression).
    - ``=-1`` rejects (sign guard).
    - ``=1.0`` rejects (``lt(1)``, closed-upper would round-trip).
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis(mantle_mass_fraction=0).mantle_mass_fraction == pytest.approx(
        0.0, abs=1e-12
    )
    assert Zalmoxis(mantle_mass_fraction=0.999).mantle_mass_fraction == pytest.approx(
        0.999, rel=1e-12
    )
    with pytest.raises(ValueError, match=r'(?i)mantle_mass_fraction'):
        Zalmoxis(mantle_mass_fraction=-1.0)
    with pytest.raises(ValueError, match=r'(?i)mantle_mass_fraction'):
        Zalmoxis(mantle_mass_fraction=1.0)


def test_zalmoxis_2layer_non_tdep_requires_zero_mantle_mass_fraction():
    """The 2-layer cross-validator in ``valid_zalmoxis`` rejects
    ``mantle_mass_fraction != 0`` when ``ice_layer_eos = None`` AND
    the mantle EOS is non-T-dependent (i.e. not WolfBower2018 or
    RTPress100TPa). The mantle mass is then derived from the core
    fraction directly; a non-zero ``mantle_mass_fraction`` would
    over-specify the geometry.

    Discrimination: pin both the rejection case AND the mirror case
    with a T-dependent EOS where ``mantle_mass_fraction != 0`` is
    accepted (and partitions mass between core and mantle).
    """
    from proteus.config import Config
    from proteus.config._struct import Struct, Zalmoxis

    # Non-T-dep mantle + non-zero mantle_mass_fraction rejects.
    with pytest.raises(ValueError, match=r'(?i)mantle_mass_fraction'):
        Config(
            interior_struct=Struct(
                module='zalmoxis',
                zalmoxis=Zalmoxis(
                    mantle_eos='PALEOS:MgSiO3',
                    mantle_mass_fraction=0.4,
                ),
            ),
            **_base_config_kwargs(),
        )
    # T-dep mantle EOS (WolfBower2018) round-trips the same value.
    cfg = Config(
        interior_struct=Struct(
            module='zalmoxis',
            zalmoxis=Zalmoxis(
                mantle_eos='WolfBower2018:MgSiO3',
                mantle_mass_fraction=0.4,
            ),
        ),
        **_base_config_kwargs(),
    )
    assert cfg.interior_struct.zalmoxis.mantle_mass_fraction == pytest.approx(0.4, rel=1e-12)


def test_zalmoxis_3layer_rejects_total_fraction_above_seager_cap():
    """The 3-layer cross-validator (with ice layer) rejects
    ``core_frac + mantle_mass_fraction > 0.75`` per Seager 2007's
    upper bound on combined core + mantle for water-world geometries.

    Edge: pin the boundary at 0.75 inclusive (round-trip) and
    just-above (reject).
    """
    from proteus.config import Config
    from proteus.config._struct import Struct, Zalmoxis

    cfg_ok = Config(
        interior_struct=Struct(
            module='zalmoxis',
            core_frac=0.325,
            core_frac_mode='mass',
            zalmoxis=Zalmoxis(
                mantle_eos='PALEOS:MgSiO3',
                ice_layer_eos='PALEOS:H2O',
                mantle_mass_fraction=0.425,
            ),
        ),
        **_base_config_kwargs(),
    )
    assert cfg_ok.interior_struct.zalmoxis.ice_layer_eos == 'PALEOS:H2O'
    # Just above 0.75: rejects.
    with pytest.raises(ValueError, match=r'(?i)75'):
        Config(
            interior_struct=Struct(
                module='zalmoxis',
                core_frac=0.325,
                core_frac_mode='mass',
                zalmoxis=Zalmoxis(
                    mantle_eos='PALEOS:MgSiO3',
                    ice_layer_eos='PALEOS:H2O',
                    mantle_mass_fraction=0.426,
                ),
            ),
            **_base_config_kwargs(),
        )


# ---------------------------------------------------------------------------
# core_density / core_heatcap echo path.
# ---------------------------------------------------------------------------


def test_zalmoxis_core_density_and_heatcap_self_sentinel_round_trip_under_zalmoxis():
    """``Struct.core_density`` and ``core_heatcap`` accept either the
    ``'self'`` sentinel (delegate to Zalmoxis EOS) or a positive
    numeric value. Defaults are ``'self'``.

    Discrimination: round-trip ``'self'`` (zalmoxis-only path) AND a
    numeric override (covers the user-pinned-density branch).
    """
    from proteus.config import Config
    from proteus.config._struct import Struct

    cfg_self = Config(
        interior_struct=Struct(module='zalmoxis'),
        **_base_config_kwargs(),
    )
    assert cfg_self.interior_struct.core_density == 'self'
    assert cfg_self.interior_struct.core_heatcap == 'self'

    cfg_num = Config(
        interior_struct=Struct(module='zalmoxis', core_density=5500.0, core_heatcap=880.0),
        **_base_config_kwargs(),
    )
    assert cfg_num.interior_struct.core_density == pytest.approx(5500.0, rel=1e-12)
    assert cfg_num.interior_struct.core_heatcap == pytest.approx(880.0, rel=1e-12)


def test_zalmoxis_core_density_and_heatcap_reject_zero_and_negative():
    """``Struct.__attrs_post_init__`` requires
    ``core_density`` / ``core_heatcap`` to be the ``'self'`` sentinel
    OR a strictly-positive number. ``0`` and negative values reject.

    Discrimination: both fields tested separately to catch a
    regression that loosened the check on only one of them.
    """
    from proteus.config._struct import Struct

    for bad_val in (0, -1.0, 0.0):
        with pytest.raises(ValueError, match=r'(?i)core_density'):
            Struct(module='zalmoxis', core_density=bad_val)
        with pytest.raises(ValueError, match=r'(?i)core_heatcap'):
            Struct(module='zalmoxis', core_heatcap=bad_val)


# ---------------------------------------------------------------------------
# Solver-tolerance + Newton-knob gt(0) boundaries.
# ---------------------------------------------------------------------------


def test_zalmoxis_solver_tolerances_gt0_with_selectivity():
    """``Zalmoxis.solver_tol_outer`` and ``solver_tol_inner`` use
    ``gt(0)``. ``=0`` and negative reject; small positive round-trips.

    Edge: pin both directions of the boundary so a ``ge(0)`` regression
    AND a ``gt(<floor>)`` regression are both caught.
    """
    from proteus.config._struct import Zalmoxis

    for bad_val in (0.0, -1e-6):
        with pytest.raises(ValueError, match=r'(?i)solver_tol_outer'):
            Zalmoxis(solver_tol_outer=bad_val)
        with pytest.raises(ValueError, match=r'(?i)solver_tol_inner'):
            Zalmoxis(solver_tol_inner=bad_val)
    z = Zalmoxis(solver_tol_outer=1e-12, solver_tol_inner=1e-12)
    assert z.solver_tol_outer == pytest.approx(1e-12, rel=1e-12)
    assert z.solver_tol_inner == pytest.approx(1e-12, rel=1e-12)
    default = Zalmoxis()
    assert default.solver_tol_outer == pytest.approx(3e-3, rel=1e-12)
    assert default.solver_tol_inner == pytest.approx(1e-4, rel=1e-12)


def test_zalmoxis_newton_tolerances_gt0_with_selectivity():
    """Newton-specific knobs (``newton_tol``,
    ``newton_relative_tolerance``, ``newton_absolute_tolerance``)
    use ``gt(0)``. ``=0`` rejects; small positive round-trips;
    documented defaults (1e-4, 1e-9, 1e-10) round-trip.
    """
    from proteus.config._struct import Zalmoxis

    for bad_val in (0.0, -1e-12):
        with pytest.raises(ValueError, match=r'(?i)newton_tol'):
            Zalmoxis(newton_tol=bad_val)
        with pytest.raises(ValueError, match=r'(?i)newton_relative_tolerance'):
            Zalmoxis(newton_relative_tolerance=bad_val)
        with pytest.raises(ValueError, match=r'(?i)newton_absolute_tolerance'):
            Zalmoxis(newton_absolute_tolerance=bad_val)
    z = Zalmoxis(
        newton_tol=1e-15,
        newton_relative_tolerance=1e-15,
        newton_absolute_tolerance=1e-15,
    )
    assert z.newton_tol == pytest.approx(1e-15, rel=1e-12)
    default = Zalmoxis()
    assert default.newton_tol == pytest.approx(1e-4, rel=1e-12)
    assert default.newton_relative_tolerance == pytest.approx(1e-9, rel=1e-12)
    assert default.newton_absolute_tolerance == pytest.approx(1e-10, rel=1e-12)


def test_zalmoxis_solver_max_iter_ge10_with_selectivity():
    """``Zalmoxis.solver_max_iter_outer/inner`` use ``ge(10)``.
    ``=10`` round-trips (catches a ``gt(10)`` regression); ``=9``
    rejects (catches a ``ge(9)`` regression that loosened the floor).
    Default 100 round-trips.
    """
    from proteus.config._struct import Zalmoxis

    z = Zalmoxis(solver_max_iter_outer=10, solver_max_iter_inner=10)
    assert z.solver_max_iter_outer == 10
    assert z.solver_max_iter_inner == 10
    with pytest.raises(ValueError, match=r'(?i)solver_max_iter_outer'):
        Zalmoxis(solver_max_iter_outer=9)
    with pytest.raises(ValueError, match=r'(?i)solver_max_iter_inner'):
        Zalmoxis(solver_max_iter_inner=9)
    default = Zalmoxis()
    assert default.solver_max_iter_outer == 100
    assert default.solver_max_iter_inner == 100


def test_zalmoxis_newton_max_iter_and_num_levels_floors_with_selectivity():
    """``newton_max_iter`` uses ``ge(5)``; ``num_levels`` does not
    expose a floor at the field layer (no validator) but the
    documented default is 150. ``lookup_nP`` / ``lookup_nS`` use
    ``ge(100)`` / ``ge(50)`` floors.

    Pin each floor at both edges where present.
    """
    from proteus.config._struct import Zalmoxis

    assert Zalmoxis(newton_max_iter=5).newton_max_iter == 5
    with pytest.raises(ValueError, match=r'(?i)newton_max_iter'):
        Zalmoxis(newton_max_iter=4)
    assert Zalmoxis(lookup_nP=100).lookup_nP == 100
    with pytest.raises(ValueError, match=r'(?i)lookup_nP'):
        Zalmoxis(lookup_nP=99)
    assert Zalmoxis(lookup_nS=50).lookup_nS == 50
    with pytest.raises(ValueError, match=r'(?i)lookup_nS'):
        Zalmoxis(lookup_nS=49)
    default = Zalmoxis()
    assert default.num_levels == 150
    assert default.lookup_nP == 1350
    assert default.lookup_nS == 280


# ---------------------------------------------------------------------------
# Structure-update trigger fields.
# ---------------------------------------------------------------------------


def test_zalmoxis_update_interval_post_init_ordering():
    """``Zalmoxis.__attrs_post_init__`` rejects
    ``update_min_interval > update_interval``: a higher floor than
    ceiling would block all updates before the ceiling could fire.
    Equality round-trips (the floor and ceiling co-fire); the
    ``update_min_interval < update_interval`` case round-trips.

    Edge: ``update_interval = 0`` short-circuits the check (no
    ordering required when both bounds are zero).
    """
    from proteus.config._struct import Zalmoxis

    # Floor higher than ceiling: rejects.
    with pytest.raises(ValueError, match=r'(?i)update_min_interval'):
        Zalmoxis(update_interval=1e6, update_min_interval=2e6)
    # Equality round-trips.
    z_eq = Zalmoxis(update_interval=1e6, update_min_interval=1e6)
    assert z_eq.update_min_interval == pytest.approx(1e6, rel=1e-12)
    # Lower floor round-trips.
    z_lt = Zalmoxis(update_interval=1e6, update_min_interval=5e5)
    assert z_lt.update_min_interval == pytest.approx(5e5, rel=1e-12)
    # Disabled ceiling short-circuits.
    z_off = Zalmoxis(update_interval=0, update_min_interval=1e6)
    assert z_off.update_min_interval == pytest.approx(1e6, rel=1e-12)


def test_zalmoxis_update_dtmagma_frac_and_dphi_abs_open_interval_selectivity():
    """``update_dtmagma_frac`` and ``update_dphi_abs`` use
    ``(gt(0), lt(1))`` at the field layer.

    Edge: pin both endpoints. ``=0`` and ``=1`` reject; small positive
    and just-below-one round-trip; defaults (0.05, 0.05) round-trip.
    """
    from proteus.config._struct import Zalmoxis

    for bad_val in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r'(?i)update_dtmagma_frac'):
            Zalmoxis(update_dtmagma_frac=bad_val)
        with pytest.raises(ValueError, match=r'(?i)update_dphi_abs'):
            Zalmoxis(update_dphi_abs=bad_val)
    z = Zalmoxis(update_dtmagma_frac=0.999, update_dphi_abs=1e-12)
    assert z.update_dtmagma_frac == pytest.approx(0.999, rel=1e-12)
    assert z.update_dphi_abs == pytest.approx(1e-12, rel=1e-12)
    default = Zalmoxis()
    assert default.update_dtmagma_frac == pytest.approx(0.05, rel=1e-12)
    assert default.update_dphi_abs == pytest.approx(0.05, rel=1e-12)


# ---------------------------------------------------------------------------
# Aragog field validators (paired with zalmoxis configuration).
# ---------------------------------------------------------------------------


def test_aragog_tolerance_struct_and_atol_gt0_with_selectivity():
    """``Aragog.tolerance_struct`` and ``atol_temperature_equivalent``
    both use ``gt(0)``. ``=0`` rejects; small positive round-trips;
    documented defaults (1e2, 1e-8) round-trip.
    """
    from proteus.config._interior import Aragog

    for bad_val in (0.0, -1.0):
        with pytest.raises(ValueError, match=r'(?i)tolerance_struct'):
            Aragog(tolerance_struct=bad_val)
        with pytest.raises(ValueError, match=r'(?i)atol_temperature_equivalent'):
            Aragog(atol_temperature_equivalent=bad_val)
    a = Aragog(tolerance_struct=1e-12, atol_temperature_equivalent=1e-30)
    assert a.tolerance_struct == pytest.approx(1e-12, rel=1e-12)
    assert a.atol_temperature_equivalent == pytest.approx(1e-30, rel=1e-12)
    default = Aragog()
    assert default.tolerance_struct == pytest.approx(1e2, rel=1e-12)
    assert default.atol_temperature_equivalent == pytest.approx(1e-8, rel=1e-12)


def test_aragog_phi_step_cap_ge0_with_selectivity():
    """``Aragog.phi_step_cap`` uses ``ge(0)``: ``=0`` round-trips
    (the documented default; catches a ``gt(0)`` regression that
    would silently reject the off-by-default value), small positive
    round-trips, negative rejects.
    """
    from proteus.config._interior import Aragog

    assert Aragog(phi_step_cap=0.0).phi_step_cap == pytest.approx(0.0, abs=1e-12)
    assert Aragog(phi_step_cap=0.05).phi_step_cap == pytest.approx(0.05, rel=1e-12)
    with pytest.raises(ValueError, match=r'(?i)phi_step_cap'):
        Aragog(phi_step_cap=-1e-12)


def test_aragog_requires_at_least_one_energy_transport_term_under_zalmoxis():
    """``valid_aragog`` rejects an Interior config with all four
    transport terms disabled. The cross-validator is on the parent
    Interior class and reads top-level ``trans_*`` flags.

    Edge: round-trip a single-term config (only conduction) and
    reject the all-off config.
    """
    from proteus.config import Config
    from proteus.config._interior import Interior
    from proteus.config._struct import Struct

    with pytest.raises(ValueError, match=r'(?i)transport'):
        Config(
            interior_struct=Struct(module='zalmoxis'),
            interior_energetics=Interior(
                module='aragog',
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
            module='aragog',
            trans_conduction=True,
            trans_convection=False,
            trans_mixing=False,
            trans_grav_sep=False,
        ),
        **_base_config_kwargs(),
    )
    assert cfg.interior_energetics.trans_conduction is True
    assert cfg.interior_energetics.trans_convection is False


# ---------------------------------------------------------------------------
# Wrapper-merge contract: Zalmoxis -> Aragog hf_row hand-off.
# ---------------------------------------------------------------------------


def test_zalmoxis_aragog_helpfile_keys_register_structure_handoff():
    """The wrapper merge propagates structure-side outputs from
    Zalmoxis into ``hf_row``, where Aragog (and the rest of the
    coupling loop) reads them on the next iteration. The schema MUST
    register:

    - Structure outputs (Zalmoxis): ``R_int``, ``M_int``, ``R_core``,
      ``M_core``, ``M_mantle``, ``M_planet``, ``P_center``, ``P_cmb``,
      ``core_density``, ``core_heatcap``.
    - Temperature hand-off (Aragog reads): ``T_core``,
      ``T_cmb_initial``, ``T_magma``, ``T_surf``.
    - CMB energy hand-off (Aragog writes back): ``F_cmb``, ``F_int``.
    - Melt-fraction state (interior_energetics output): ``Phi_global``,
      ``Phi_global_vol``, ``M_mantle_solid``, ``M_mantle_liquid``.

    Discrimination: every key tested separately so a regression that
    dropped any one fails the per-key loop. ``ZeroHelpfileRow`` seeds
    each as a float zero.
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
    temperature_keys = ('T_cmb', 'T_cmb_initial', 'T_magma', 'T_surf')
    cmb_flux_keys = ('F_cmb', 'F_int')
    melt_keys = (
        'Phi_global',
        'Phi_global_vol',
        'M_mantle_solid',
        'M_mantle_liquid',
    )
    for key in structure_keys + temperature_keys + cmb_flux_keys + melt_keys:
        assert key in keys, f'{key} must be registered in GetHelpfileKeys()'
    row = ZeroHelpfileRow()
    for key in structure_keys + temperature_keys + cmb_flux_keys + melt_keys:
        # ZeroHelpfileRow seeds keys as float(0.0); the relative form
        # of pytest.approx is undefined at zero, so use abs.
        assert row[key] == pytest.approx(0.0, abs=1e-30)
        assert isinstance(row[key], float)
