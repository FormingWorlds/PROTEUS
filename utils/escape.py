# Functions used to handle escape

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

log = logging.getLogger("PROTEUS")


def RunDummyEsc():
    """Run dummy escape model.

    Returns
    ----------
        phi : float 
            Bulk escape rate [kg/s]

    """
    log.info("Running dummy escape...")

    # Hardcoded dummy value of bulk volatile escape rate [kg/s]
    phi = 1e10

    return phi



# Zephyrus wrapper 
def RunZEPHYRUS():
    log.info("Running ZEPHYRUS...")
    raise Exception("Not yet implemented")

