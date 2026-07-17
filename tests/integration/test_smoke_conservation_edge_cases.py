"""
Conservation invariants at hand-picked corners of the physical parameter space.

Companion to ``test_smoke_hypothesis.py``: that file fuzzes the same invariants
across the whole (a, e, Teff, M_planet) box and costs slow-tier time, while
these five cases pin named corners at smoke cost. They live in their own file
because a file carries one tier, so that the tier filters select every test
exactly once.

The cases are non-overlapping with the handwritten smoke tests in
``test_smoke_modules.py``, which all use the ``dummy.toml`` defaults.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from _smoke_invariants import assert_smoke_conservation_invariants
from helpers import PROTEUS_ROOT

from proteus import Proteus

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

# Each tuple sits at a DIFFERENT corner of the parameter space than the
# handwritten tests in test_smoke_modules.py.
EDGE_CASES = [
    # Ultra-hot inner edge (TOI-561 b regime)
    (0.01, 0.0, 5800.0, 1.0),
    # Eccentric near-Earth orbit
    (1.0, 0.4, 5800.0, 1.0),
    # M dwarf habitable zone, super-Earth
    (0.05, 0.0, 2566.0, 3.0),
    # F dwarf wide orbit, sub-Earth
    (1.5, 0.1, 7000.0, 0.5),
    # Circular orbit (e = 0 exactly) catches eccentric-anomaly sign bugs
    (0.5, 0.0, 5800.0, 1.0),
]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('semimajoraxis,eccentricity,teff,m_planet_mearth', EDGE_CASES)
def test_smoke_conservation_invariants_named_edge_cases(
    semimajoraxis, eccentricity, teff, m_planet_mearth
):
    """Run the conservation-invariant check at hand-picked corners of the
    physical parameter space.

    Pinning the corners as parametrised runs means a regression at, for
    example, exactly circular orbits fails a named case rather than waiting
    for the fuzzer to sample that corner.
    """
    unique_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=PROTEUS_ROOT / 'input' / 'dummy.toml')
        runner.config.params.out.path = str(Path(tmpdir) / f'edge_smoke_{unique_id}')
        runner.init_directories()

        runner.config.orbit.semimajoraxis = semimajoraxis
        runner.config.orbit.eccentricity = eccentricity
        runner.config.star.dummy.Teff = teff
        runner.config.planet.mass_tot = m_planet_mearth

        runner.config.planet.tsurf_init = 2000.0
        runner.config.params.stop.time.minimum = 1e2
        runner.config.params.stop.time.maximum = 1e3
        runner.config.params.out.plot_mod = None
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = 'none'

        runner.start(resume=False, offline=True)

        # Discriminating pre-checks: the run produced helpfile rows AND the
        # eccentricity override survived to the final config. A regression that
        # exited the loop early, or one that lost the override at IC, would
        # have let the conservation helper return early and pass vacuously.
        assert len(runner.hf_all) > 0
        assert runner.config.orbit.eccentricity == pytest.approx(eccentricity, abs=1e-12)

        assert_smoke_conservation_invariants(runner.hf_all)
