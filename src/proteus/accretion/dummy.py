# Timeline-replay accretion module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from proteus.accretion.common import read_timeline

if TYPE_CHECKING:
    from proteus.accretion.common import ImpactEvent
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def get_timeline(config: Config) -> list[ImpactEvent]:
    """Read a pre-written impact timeline.

    Replays a timeline produced earlier instead of running a dynamical
    model, so impact consequences can be driven from a known event
    sequence.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    events : list of ImpactEvent
        Impacts to apply during the run, in time order.
    """
    path = config.accretion.dummy.timeline_path
    log.info('Reading impact timeline from file')

    return read_timeline(path, time_offset=config.accretion.time_offset)
