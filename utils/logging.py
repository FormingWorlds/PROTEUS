# Setup logger for PROTEUS

import logging, sys, os

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(levelname)7s ] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Custom logger instance 
def setup_logger(logpath:str="new.log",level:str="INFO",logterm:bool=True):

    # https://stackoverflow.com/a/61457119

    custom_logger = logging.getLogger()
    custom_logger.handlers.clear()

    if os.path.exists(logpath):
        os.remove(logpath)
    
    level = str(level).strip().upper()
    if level not in ["INFO", "DEBUG", "ERROR", "CRITICAL"]:
        level = "INFO"
    level_code = logging.getLevelName(level)

    # Add terminal output to logger
    if logterm:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(CustomFormatter())
        sh.setLevel(level_code)
        custom_logger.addHandler(sh)

    # Add file output to logger
    fh = logging.FileHandler(logpath)
    fh.setFormatter(logging.Formatter("[%(levelname)7s ] %(message)s"))
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

    