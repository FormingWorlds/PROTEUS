# Setup logger for PROTEUS

import logging, sys, os

# Custom logger instance 
def setup_logger(logpath:str="new.log",level:str="INFO",logterm:bool=True):

    # https://stackoverflow.com/a/61457119

    custom_logger = logging.getLogger()
    custom_logger.handlers.clear()
    
    if os.path.exists(logpath):
        os.remove(logpath)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    level = str(level).strip().upper()
    if level not in ["INFO", "DEBUG", "ERROR", "CRITICAL"]:
        level = "INFO"
    level_code = logging.getLevelName(level)

    # Add terminal output to logger
    if logterm:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(level_code)
        custom_logger.addHandler(sh)

    # Add file output to logger
    fh = logging.FileHandler(logpath)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    custom_logger.addHandler(fh)
    custom_logger.setLevel(level_code)

    # Capture unhandled exceptions
    # https://stackoverflow.com/a/16993115
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            custom_logger.error("KeyboardInterrupt")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        custom_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception
    
    return 

    