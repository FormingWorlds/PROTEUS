# Functions used to run the Morrigan giant-impact module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.accretion.common import TIMELINE_COLUMNS, ImpactEvent, validate_timeline
from proteus.utils.constants import AU, M_earth

if TYPE_CHECKING:
    from collections.abc import Sequence

    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

try:
    import morrigan  # type: ignore
except ModuleNotFoundError:  # optional dependency
    morrigan = None

# Entry point Morrigan must expose for PROTEUS to drive it.
MORRIGAN_ENTRY_POINT = 'run_system'

# Timeline fields that identify bodies rather than measure them, so they are
# read as integers while every other field is a physical quantity.
_ID_COLUMNS = ('id_target', 'id_impactor')

INSTALL_HINT = (
    "accretion.module = 'morrigan' requires the morrigan package. "
    'Install it with: pip install "fwl-proteus[morrigan]" '
    '(or bash tools/get_morrigan.sh for an editable checkout).'
)


def require_morrigan():
    """Return the Morrigan package, or explain how to install it.

    Returns
    -------
    module
        The imported ``morrigan`` package.

    Raises
    ------
    ImportError
        If the package is not installed, or is installed but does not
        expose the entry point PROTEUS drives it through.
    """
    if morrigan is None:
        raise ImportError(INSTALL_HINT)

    if not hasattr(morrigan, MORRIGAN_ENTRY_POINT):
        raise ImportError(
            f'The installed morrigan package does not expose '
            f'{MORRIGAN_ENTRY_POINT}(), which PROTEUS uses to run a system. '
            'Update morrigan to a version that provides it.'
        )

    return morrigan


def select_planet(survivors: Sequence[dict], config: Config) -> dict:
    """Choose which surviving body's impact history PROTEUS follows.

    A dynamical run leaves several survivors; PROTEUS simulates one. Each
    survivor record carries ``id``, ``mass_initial`` and ``a_initial``
    (its state at the start of the dynamical run), and ``mass_final`` and
    ``a_final`` (its state at the end), in SI units.

    The selectors are: ``match_config``, which picks the survivor whose
    starting mass and orbit are closest to the configured planet, so a
    borrowed history belongs to a body resembling the one being
    simulated; ``mass``, the most massive survivor; ``semimajoraxis``,
    the survivor whose final orbit is nearest a target in AU; and ``id``,
    an explicitly named body.

    Parameters
    ----------
    survivors : sequence of dict
        Surviving bodies from the dynamical run.
    config : Config
        Model configuration.

    Returns
    -------
    survivor : dict
        The selected record.

    Raises
    ------
    ValueError
        If there are no survivors, or if the 'id' selector names a body
        that did not survive.
    """
    if not survivors:
        raise ValueError(
            'The dynamical run left no surviving bodies, so there is no impact '
            'history to follow. Check the accretion.morrigan settings.'
        )

    mor = config.accretion.morrigan

    match mor.selector:
        case 'mass':
            chosen = max(survivors, key=lambda s: s['mass_final'])

        case 'semimajoraxis':
            target = float(mor.selector_value) * AU
            chosen = min(survivors, key=lambda s: abs(s['a_final'] - target))

        case 'id':
            wanted = int(mor.selector_value)
            matches = [s for s in survivors if int(s['id']) == wanted]
            if not matches:
                available = sorted(int(s['id']) for s in survivors)
                raise ValueError(
                    f'accretion.morrigan.selector_value = {wanted} names a body that '
                    f'did not survive. Surviving ids: {available}'
                )
            chosen = matches[0]

        case _:  # 'match_config'
            # Compare in relative terms so mass and orbit contribute
            # comparably; an absolute distance in SI would be dominated by
            # whichever quantity happens to carry the larger exponent.
            target_mass = config.planet.mass_tot * M_earth
            target_a = config.orbit.semimajoraxis * AU
            chosen = min(
                survivors,
                key=lambda s: np.hypot(
                    (s['mass_initial'] - target_mass) / target_mass,
                    (s['a_initial'] - target_a) / target_a,
                ),
            )

    log.info(
        "Following body %s (selector '%s'): %.3f -> %.3f M_earth, %.4f -> %.4f AU",
        chosen['id'],
        mor.selector,
        chosen['mass_initial'] / M_earth,
        chosen['mass_final'] / M_earth,
        chosen['a_initial'] / AU,
        chosen['a_final'] / AU,
    )

    return chosen


def build_parameters(config: Config) -> dict:
    """Translate the PROTEUS configuration into Morrigan run parameters.

    The stellar mass is taken from ``star.mass`` rather than from the
    accretion section, so the dynamical model and the rest of the run
    cannot disagree about the host star.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    params : dict
        Keyword arguments for the Morrigan entry point. Masses are in kg
        and lengths in m.
    """
    mor = config.accretion.morrigan

    masses = list(mor.masses) if mor.masses else [mor.mass_equal] * mor.num_planets

    return {
        'seed': mor.seed,
        'masses': [m * M_earth for m in masses],
        'eccentricity': mor.eccentricity_init,
        'inner_edge': mor.inner_edge * AU,
        'spacing': mor.spacing,
        'density': mor.density,
        'impact_angle': mor.impact_angle,
        'evolution_time': mor.evolution_time,
        'inner_cutoff': mor.inner_cutoff * AU,
        'stellar_mass': config.star.mass,
    }


def get_timeline(config: Config) -> list[ImpactEvent]:
    """Run a system and return the selected body's impact history.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    events : list of ImpactEvent
        Impacts on the selected body, in time order.

    Raises
    ------
    ImportError
        If the morrigan package is unavailable.
    KeyError
        If the run reports no impact history for the selected body.
    ValueError
        If the model's outcome or its impact records do not carry the fields
        the coupling requires.
    """
    package = require_morrigan()

    params = build_parameters(config)
    log.info('Running giant-impact model for %d embryos', len(params['masses']))

    outcome = getattr(package, MORRIGAN_ENTRY_POINT)(**params)

    for key in ('survivors', 'impacts'):
        if key not in outcome:
            raise ValueError(
                f"The giant-impact model returned no '{key}' entry. Expected a mapping "
                f'carrying {sorted(("survivors", "impacts"))}, got '
                f'{sorted(outcome.keys())}. This usually means the installed '
                'fwl-morrigan is newer than the coupling expects.'
            )

    chosen = select_planet(outcome['survivors'], config)
    records = outcome['impacts'][chosen['id']]

    offset = config.accretion.time_offset
    # Select the fields explicitly rather than splatting each record, the same
    # way the file reader does. The dependency is pinned by a version floor, so
    # a later release may add fields to its records; ignoring the ones the
    # coupling does not consume keeps that from becoming a fatal argument error,
    # and a missing field still reports which one by name.
    events = []
    for index, record in enumerate(records):
        missing = [column for column in TIMELINE_COLUMNS if column not in record]
        if missing:
            raise ValueError(
                f'Impact record {index} from the giant-impact model is missing '
                f'required fields: {missing}. Expected all of: '
                f'{list(TIMELINE_COLUMNS)}'
            )
        fields = {
            column: float(record[column])
            for column in TIMELINE_COLUMNS
            if column not in _ID_COLUMNS
        }
        fields['time'] += offset
        for column in _ID_COLUMNS:
            fields[column] = int(record[column])
        events.append(ImpactEvent(**fields))
    events.sort(key=lambda e: e.time)

    validate_timeline(events)

    log.info('Body %s experienced %d impacts', chosen['id'], len(events))
    return events
