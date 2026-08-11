# Impact timeline replayed from file
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

    Replays a sequence of impacts produced elsewhere instead of deriving one
    from a dynamical model. Every consequence is applied exactly as it is for a
    model-derived timeline, so a run reproduces a published impact history, an
    externally computed one, or a hand-written sequence.

    Parameters
    ----------
    config : Config
        Model configuration.

    Returns
    -------
    events : list of ImpactEvent
        Impacts to apply during the run, in time order.
    """
    path = config.accretion.timeline.timeline_path
    log.info('Reading impact timeline from file')

    return read_timeline(path, time_offset=config.accretion.time_offset)
