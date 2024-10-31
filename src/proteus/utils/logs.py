# Setup logger for PROTEUS
from __future__ import annotations

import logging
import os
import sys


# Fake file-like stream object that redirects writes to a logger instance.
class StreamToLogger(object):
    # https://stackoverflow.com/a/36296215
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''

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

    logger_name = "fwl"

    # https://stackoverflow.com/a/61457119
    custom_logger = logging.getLogger(logger_name)

    if os.path.exists(logpath):
        os.remove(logpath)

    level = str(level).strip().upper()
    if level not in ["INFO", "DEBUG", "ERROR", "WARNING"]:
        raise ValueError(f"Invalid log level: {level}")
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

    return custom_logger

def GetCurrentLogfileIndex(output_dir:str):
    '''
    Get the index of the current logfile, returning -1 if none exists
    '''
    i=0
    j=-1
    while i<99:
        fname = "proteus_%02d.log"%i
        fpath = os.path.join(output_dir, fname)

        if os.path.exists(fpath):
            j=i
            i += 1
        else:
            break

    return j

def GetLogfilePath(output_dir:str, j:int):
    '''
    Get path to logfile with index j
    '''

    if j>99:
        raise Exception("Cannot create logfile - too many in output folder already")

    return os.path.join(output_dir, "proteus_%02d.log"%j)
