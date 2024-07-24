# Functions used to handle escape

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

log = logging.getLogger("PROTEUS")


def RunDummyEsc():
    """Run dummy escape model.

    Returns
    ----------
        out : dict
            Dictionary of bulk escape rates for each element [kg s-1]

    """
    log.info("Running dummy escape...")

    # Hardcoded dummy value of bulk volatile escape rate [kg/s]
    phi = 1e7 

    # Escape rates for each element 
    out = {}
    for e in element_list:
        out[e] = phi

    return out



# Zephyrus wrapper 
def RunZEPHYRUS():
    log.info("Running ZEPHYRUS...")
    raise Exception("Not yet implemented")

