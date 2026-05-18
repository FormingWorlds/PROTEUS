"""PROTEUS-side tests for the Zalmoxis ``outer_solver`` config knob.

Covers two angles:

(1) Config schema: ``proteus.config.interior_struct.zalmoxis.outer_solver``
    accepts only 'picard' or 'newton' and defaults to 'newton'. 'picard'
    remains available for explicit opt-in.

(2) Plumbing: ``load_zalmoxis_configuration`` propagates the knob into
    the ``config_params`` dict that ``zalmoxis.solver.main`` consumes,
    and auto-tightens integrator tolerances when 'newton' is selected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from proteus.config._struct import Zalmoxis

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------


class TestOuterSolverSchema:
    """Config-class validation of outer_solver and Newton-specific knobs."""

    def test_default_outer_solver_is_newton(self):
        """Default ``outer_solver`` is 'newton'. PROTEUS' plumbing also
        tightens integrator tolerances when 'newton' is selected, so the
        default path is end-to-end consistent (see
        ``TestNewtonPathTightensIntegratorTolerances``).
        """
        z = Zalmoxis()
        assert z.outer_solver == 'newton'
        # Discriminator: 'newton' is the only accepted default; a
        # regression that fell back to the legacy 'picard' default
        # would break the end-to-end consistency with the
        # tighten-integrator-tolerance plumbing and would land here.
        assert z.outer_solver != 'picard'

    def test_explicit_picard_accepted(self):
        """An explicit ``outer_solver='picard'`` is accepted by the
        validator; pairs with the default-Newton test above.
        """
        z = Zalmoxis(outer_solver='picard')
        assert z.outer_solver == 'picard'
        # Discriminator: the validator must not silently coerce
        # 'picard' to the default 'newton'. A regression that flipped
        # the default-vs-explicit handling would land here.
        assert z.outer_solver != 'newton'

    def test_explicit_newton_accepted(self):
        """An explicit ``outer_solver='newton'`` is accepted; round-trips
        through the validator and lands on the attribute as-is.
        """
        z = Zalmoxis(outer_solver='newton')
        assert z.outer_solver == 'newton'
        # Discriminator: case-folding or whitespace stripping by the
        # validator could silently rewrite 'newton' to a different
        # casing. Pin the lowercase, exact form.
        assert z.outer_solver == z.outer_solver.lower().strip()

    @pytest.mark.parametrize(
        'bad_value',
        ['Newton', 'PICARD', '', 'broyden', 'levenberg', 'newton ', None, 42],
    )
    def test_invalid_outer_solver_rejected(self, bad_value):
        """attrs in_() validator catches typos, case, and wrong types."""
        with pytest.raises(ValueError):
            Zalmoxis(outer_solver=bad_value)
        # Positive control: the immediately-adjacent good values must
        # both still construct. This rules out a regression that
        # broadened the rejection to also block the legitimate
        # 'newton' and 'picard' tokens.
        assert Zalmoxis(outer_solver='newton').outer_solver == 'newton'
        assert Zalmoxis(outer_solver='picard').outer_solver == 'picard'

    def test_newton_max_iter_default(self):
        """``newton_max_iter`` defaults to 30; tested separately from the
        validator-boundary test below.
        """
        z = Zalmoxis()
        assert z.newton_max_iter == 30
        # Bounded discriminator: 30 must lie strictly above the
        # validator's ge(5) lower bound (otherwise the default sits
        # at the precondition boundary and any noise pushes it
        # below) and well below an absurd ceiling.
        assert 5 < z.newton_max_iter < 1000

    def test_newton_max_iter_minimum_5(self):
        """ge(5) validator: too few iters can't converge anything useful."""
        with pytest.raises(ValueError):
            Zalmoxis(newton_max_iter=4)
        # Boundary: 5 must accept.
        z = Zalmoxis(newton_max_iter=5)
        assert z.newton_max_iter == 5

    def test_newton_tol_default_and_validation(self):
        """``newton_tol`` defaults to 1e-4 and rejects non-positive
        values (zero or negative), so a misconfigured tolerance fails
        loudly at attrs validation time.
        """
        z = Zalmoxis()
        assert z.newton_tol == pytest.approx(1.0e-4, rel=1e-12)
        # gt(0) validator
        with pytest.raises(ValueError):
            Zalmoxis(newton_tol=0)
        with pytest.raises(ValueError):
            Zalmoxis(newton_tol=-1.0e-3)

    def test_newton_integrator_tolerance_defaults(self):
        """Default integrator tolerances are 1e-9 (relative) and 1e-10
        (absolute), which satisfy Zalmoxis' Newton-path precondition
        ``relative_tolerance <= 1e-7``.
        """
        z = Zalmoxis()
        assert z.newton_relative_tolerance == pytest.approx(1.0e-9, rel=1e-12)
        assert z.newton_absolute_tolerance == pytest.approx(1.0e-10, rel=1e-12)


# ----------------------------------------------------------------------
# Plumbing: load_zalmoxis_configuration -> config_params dict
# ----------------------------------------------------------------------


def _make_mock_config(
    outer_solver='picard',
    newton_max_iter=30,
    newton_tol=1.0e-4,
    newton_rel_tol=1.0e-9,
    newton_abs_tol=1.0e-10,
):
    """Build a mock proteus Config that exercises the Zalmoxis-config builder."""
    config = MagicMock()
    config.planet.mass_tot = 1.0  # 1 M_earth
    config.planet.tsurf_init = 1500.0
    config.planet.tcmb_init = 4000.0
    config.planet.tcenter_init = 5000.0
    config.planet.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.core_eos = 'PALEOS:iron'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS-2phase:MgSiO3'
    config.interior_struct.zalmoxis.ice_layer_eos = None
    config.interior_struct.zalmoxis.mushy_zone_factor = 0.8
    config.interior_struct.zalmoxis.mantle_mass_fraction = 0.0
    config.interior_struct.zalmoxis.num_levels = 100
    config.interior_struct.zalmoxis.solver_tol_outer = 3.0e-3
    config.interior_struct.zalmoxis.solver_tol_inner = 1.0e-4
    config.interior_struct.zalmoxis.solver_max_iter_outer = 100
    config.interior_struct.zalmoxis.solver_max_iter_inner = 100
    config.interior_struct.zalmoxis.use_jax = False
    config.interior_struct.zalmoxis.use_anderson = False
    config.interior_struct.zalmoxis.dry_mantle = True
    config.interior_struct.zalmoxis.outer_solver = outer_solver
    config.interior_struct.zalmoxis.newton_max_iter = newton_max_iter
    config.interior_struct.zalmoxis.newton_tol = newton_tol
    config.interior_struct.zalmoxis.newton_relative_tolerance = newton_rel_tol
    config.interior_struct.zalmoxis.newton_absolute_tolerance = newton_abs_tol
    config.interior_struct.zalmoxis.global_miscibility = False
    return config


def _stub_get_target_surface_pressure(monkeypatch, value=1.0e5):
    """Patch the surface-pressure helper so the builder doesn't probe outgassing."""
    import proteus.interior_struct.zalmoxis as _mod

    monkeypatch.setattr(_mod, '_get_target_surface_pressure', lambda *a, **kw: value)


def _make_hf_row():
    """Build a minimal hf_row that satisfies load_zalmoxis_configuration's
    volatile-element loop (M_volatiles = sum of *_kg_total over element_list).
    """
    return {
        'M_volatiles': 0.0,
        'H_kg_total': 0.0,
        'C_kg_total': 0.0,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
    }


class TestPicardPathDoesNotChangeIntegratorTolerances:
    """``outer_solver='picard'`` must not pass ``relative_tolerance`` or
    ``absolute_tolerance`` in ``config_params``.

    Contract: a Picard run must build a ``config_params`` dict that omits
    the integrator-tolerance keys so Zalmoxis falls back to its
    mass-adaptive defaults (1e-5 / 1e-6). The 'outer_solver' key itself
    is present (=='picard'); only the tolerance overrides must be absent.
    """

    def test_picard_omits_integrator_tolerance_keys(self, monkeypatch):
        """When ``outer_solver='picard'``, ``load_zalmoxis_configuration``
        sets the ``outer_solver`` key but does NOT inject
        ``relative_tolerance`` / ``absolute_tolerance``, so Zalmoxis
        falls back to its mass-adaptive defaults.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='picard')
        _stub_get_target_surface_pressure(monkeypatch)
        hf_row = _make_hf_row()

        cp = load_zalmoxis_configuration(config, hf_row)
        assert cp['outer_solver'] == 'picard'
        assert 'relative_tolerance' not in cp, (
            'Picard path must not override integrator tols; got '
            f'relative_tolerance={cp.get("relative_tolerance")}'
        )
        assert 'absolute_tolerance' not in cp


class TestNewtonPathTightensIntegratorTolerances:
    """outer_solver='newton' MUST also pass tightened integrator tols.

    Without these, the in-Zalmoxis Newton fails ValueError at entry
    (precondition: relative_tolerance <= 1e-7).
    """

    def test_newton_passes_relative_tolerance(self, monkeypatch):
        """When ``outer_solver='newton'``, ``load_zalmoxis_configuration``
        injects the tightened integrator tolerances (1e-9 relative,
        1e-10 absolute) plus the Newton-specific iteration cap and tol,
        so the Newton precondition is satisfied automatically.
        """
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='newton')
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['outer_solver'] == 'newton'
        assert cp['relative_tolerance'] == pytest.approx(1.0e-9, rel=1e-12)
        assert cp['absolute_tolerance'] == pytest.approx(1.0e-10, rel=1e-12)
        assert cp['newton_max_iter'] == 30
        assert cp['newton_tol'] == pytest.approx(1.0e-4, rel=1e-12)

    def test_newton_propagates_custom_knobs(self, monkeypatch):
        """Non-default Newton knobs flow through unchanged."""
        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(
            outer_solver='newton',
            newton_max_iter=50,
            newton_tol=5.0e-5,
            newton_rel_tol=1.0e-10,
            newton_abs_tol=1.0e-11,
        )
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['newton_max_iter'] == 50
        assert cp['newton_tol'] == pytest.approx(5.0e-5, rel=1e-12)
        assert cp['relative_tolerance'] == pytest.approx(1.0e-10, rel=1e-12)
        assert cp['absolute_tolerance'] == pytest.approx(1.0e-11, rel=1e-12)

    def test_newton_tols_satisfy_zalmoxis_precondition(self, monkeypatch):
        """The default tols must satisfy Zalmoxis' Newton precondition.

        Discriminating: zalmoxis.solver._NEWTON_REQUIRED_REL_TOL is 1e-7;
        if PROTEUS' default newton_relative_tolerance ever drifts above
        this, Newton will ValueError at entry. This test pins the
        invariant.
        """
        from zalmoxis.solver import _NEWTON_REQUIRED_REL_TOL

        from proteus.interior_struct.zalmoxis import load_zalmoxis_configuration

        config = _make_mock_config(outer_solver='newton')
        _stub_get_target_surface_pressure(monkeypatch)
        cp = load_zalmoxis_configuration(config, _make_hf_row())

        assert cp['relative_tolerance'] <= _NEWTON_REQUIRED_REL_TOL, (
            f"PROTEUS' default newton_relative_tolerance "
            f'({cp["relative_tolerance"]:.0e}) must be <= '
            f'_NEWTON_REQUIRED_REL_TOL ({_NEWTON_REQUIRED_REL_TOL:.0e}); '
            'otherwise Newton ValueErrors on entry.'
        )
        # Sign / positivity guard: an integrator tolerance must be a
        # strictly positive real. A regression that set the default
        # to 0 or a negative number would still pass the
        # less-than-or-equal check above (0 <= 1e-7) but would crash
        # the SciPy integrator at entry.
        assert cp['relative_tolerance'] > 0.0
        # Scale guard: the default must also be well above the
        # smallest representable double (~5e-324). The Newton
        # precondition is 1e-7; the Zalmoxis-default is 1e-9. A
        # regression that bumped the default below 1e-15 would
        # produce silent integrator stalls instead of useful steps.
        assert cp['relative_tolerance'] >= 1.0e-15
