# Contains routines for model termination and convergence

# Import utils-specific modules
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Summarise convergence criteria
def print_termination_criteria(config:Config):
    log.info("Active termination criteria")

    def _print_criterion(state, msg):
        log.info(f"    {msg:16} {'✔' if state else '✗'}  ")

    # Optionally enabled:
    _print_criterion(config.params.stop.solid.enabled,  "Solidification")
    _print_criterion(config.params.stop.radeqm.enabled, "Energy balance")
    _print_criterion(config.params.stop.escape.enabled, "Volatile escape")
    _print_criterion(config.params.stop.time.enabled,   "Maximum time")
    _print_criterion(config.params.stop.iters.enabled,  "Maximum loops")

    # Always enabled:
    _print_criterion(True, "Keepalive file")

# Print message in common format
def _msg_termination(msg:str):
    log.info("")
    log.info(f"===> {msg}! <===")
    log.info("")

# Solidified condition
def _check_solid(handler: Proteus) -> bool:
    log.debug("Check solidification")
    if handler.hf_row["Phi_global"] <= handler.config.params.stop.solid.phi_crit:
        UpdateStatusfile(handler.directories, 10)
        _msg_termination("Planet solidified!")
        return True
    return False

# Energy balance condition
def _check_radeqm(handler: Proteus) -> bool:
    log.debug("Check energy balance / radeqm")

    # Radiative equilibrium nominally occurs when F_atm is zero.
    # However, with tidal heating included, this is offset by the interior
    #   heat production of the planet. Energy balance is instead achieved
    #   when F_atm and F_tidal are equal.

    F_eps = handler.hf_row["F_atm"] - handler.hf_row["F_tidal"] - handler.hf_row["F_radio"]
    F_ref = abs(handler.hf_row["F_atm"]) * handler.config.params.stop.radeqm.rtol + handler.config.params.stop.radeqm.atol
    log.debug("    val, req = %.3e, %.3e  W m-2"%(F_eps, F_ref))

    if abs(F_eps) <= F_ref:
        UpdateStatusfile(handler.directories, 14)
        _msg_termination("Planet has reached global energy balance")
        return True

     # Simulation when wants to warm up but cannot do so
    if handler.config.atmos_clim.prevent_warming and (F_eps < 0):
        UpdateStatusfile(handler.directories, 14)
        _msg_termination("Planet is no longer cooling")
        return True

    return False

# Volatiles have escaped
def _check_escape(handler: Proteus) -> bool:
    log.debug("Check escape")

    P_surf = handler.hf_row["P_surf"]
    P_stop = handler.config.params.stop.escape.p_stop
    log.debug("    val, req = %.3e, %.3e  bar"%(P_surf, P_stop))

    if handler.has_escaped or (P_surf <= P_stop):
        UpdateStatusfile(handler.directories, 15)
        _msg_termination("Atmosphere has escaped")
        return True

    return False

# Maximum time
def _check_time(handler: Proteus) -> bool:
    log.debug("Check maximum time")

    if handler.hf_row["Time"] >= handler.config.params.stop.time.maximum:
        UpdateStatusfile(handler.directories, 13)
        _msg_termination("Target time reached")
        return True
    return False

# Maximum iterations
def _check_maxiter(handler: Proteus) -> bool:
    log.debug("Check maximum iterations")

    if handler.loops["total"] > handler.loops["total_loops"]:
        UpdateStatusfile(handler.directories, 12)
        _msg_termination("Maximum number of iterations reached")
        return True
    return False

# Minimum iterations (return true when exit is allowed)
def _check_miniter(handler: Proteus, finished:bool) -> bool:
    log.debug("Check minimum iterations")

    # Number of iterations not reached
    if handler.loops["total"] <= handler.loops["total_min"]:
        if finished:
            # Model thinks that it is done
            log.warning("Minimum number of iterations not yet attained; continuing...")
            UpdateStatusfile(handler.directories, 1)
        return False # do not exit
    return True

def _check_keepalive(handler: Proteus) -> bool:
    log.debug("Check keepalive file")

    if not os.path.exists(handler.lockfile):
        UpdateStatusfile(handler.directories, 25)
        _msg_termination("Model exit was requested")
        return True
    return False

def check_termination(handler: Proteus) -> bool:
    """
    Decide if the model should stop or not.

    The model will stop when two sequential iterations satisfy the convergence criteria.
    This occurs when finished_prev = True  and  finished_both = True.

    Parameters:
    --------------
        handler: Proteus
            Proteus instance

    Returns:
    --------------
        fininshed: bool
            Should the model terminate now?
    """

    # Quantify model state at this iteration
    finished = False
    handler.finished_both = False

    # Check if keepalive file has been removed
    #    This means that the model should exit ASAP, regardless of the other criteria.
    if _check_keepalive(handler):
        handler.finished_both = True
        handler.finished_prev = True
        return True

    # Stop simulation when planet is completely solidified
    if handler.config.params.stop.solid.enabled:
        finished = finished or _check_solid(handler)

    # Stop simulation when at global energy balance
    if handler.config.params.stop.radeqm.enabled:
        finished = finished or _check_radeqm(handler)

    # Atmosphere has escaped
    if handler.config.params.stop.escape.enabled:
        finished = finished or _check_escape(handler)
        # By pass two-iteration check here, otherwise we will get a divide-by-zero
        #   error in the next iteration's calculation.
        handler.finished_prev = handler.finished_prev or handler.has_escaped

    # Maximum time reached
    if handler.config.params.stop.time.enabled:
        finished = finished or _check_time(handler)

    # Maximum loops reached
    if handler.config.params.stop.iters.enabled:
        finished = finished or _check_maxiter(handler)

    # Minimum loops reached
    if handler.config.params.stop.iters.enabled:
        finished = finished and _check_miniter(handler, finished)

    # Non-strict check
    if not handler.config.params.stop.strict:
        if finished:
            handler.finished_prev = True
            handler.finished_both = True
            log.info("Convergence criteria satisfied")
            log.debug("Model will exit")
            return True

    # Strict check
    # Ensure that criteria are satisfied for two sequential iterations
    else:
        if handler.finished_prev:
            # previous iteration satisfied the criteria...
            if finished:
                # convergence is also satisfied at this iteration
                handler.finished_both = True
                log.info("Convergence criteria satisfied twice")
                log.debug("Model will exit")
                return True
            else:
                # convergence no longer satisfied - reset flags
                handler.finished_prev = False
                log.warning("Convergence criteria no longer satisfied")
        else:
            # previous iteration DID NOT satisfy criteria...
            handler.finished_prev = finished
            if finished:
                log.info("Convergence criteria satisfied once")

    # Reset statusfile to 'Running'
    UpdateStatusfile(handler.directories, 1)
    return False
