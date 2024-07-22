# Setup logger for PROTEUS

import logging, sys, os

class CustomFormatter(logging.Formatter):

    part1 = "[\033["

    info =  "32"
    warn =  "93"
    debug = "96"
    error = "91"

    part2 = "m\033[1m %(levelname)-5s \033[21m\033[0m] %(message)s"

    FORMATS = {
        logging.DEBUG:   part1+debug+part2,
        logging.INFO:    part1+info +part2,
        logging.WARNING: part1+warn +part2,
        logging.ERROR:   part1+error+part2,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Custom logger instance 
def setup_logger(logpath:str="new.log",level:str="INFO",logterm:bool=True):

    # https://stackoverflow.com/a/61457119

    custom_logger = logging.getLogger("PROTEUS")
    custom_logger.handlers.clear()

    if os.path.exists(logpath):
        os.remove(logpath)
    
    level = str(level).strip().upper()
    if level not in ["INFO", "DEBUG", "ERROR", "WARNING"]:
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
    fh.setFormatter(logging.Formatter("[ %(levelname)-5s ] %(message)s"))

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

def GetNextLogfilePath(output_dir:str):
    '''
    Get path to logfile, handling existing files appropriately.
    '''

    i=0
    while i<10:
        fname = "%01d.log"%i
        fpath = os.path.join(output_dir, fname)

        # this will be the new logfile path
        if not os.path.exists(fpath):
            return fpath
        
        # try new name 
        i += 1
    
    raise Exception("Cannot create logfile - too many in output folder already")

    