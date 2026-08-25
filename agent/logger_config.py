# logger_config.py
# Common logger configuration for all modules

import logging
import os
import datetime
import time
import inspect
from zoneinfo import ZoneInfo
from typing import Tuple

class LocalFormatter(logging.Formatter):
    """Formatter using local timezone"""
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.timetuple()
    
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return s

class ColorFormatter(LocalFormatter):
    """Console formatter with colors"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

# Global configuration state tracking
_logger_initialized = False
_log_filename = None
_log_dir = None

def setup_unified_logger(lib: str) -> str:
    """
    Unified logger configuration for all modules
    Should be called only once, then use get_logger() to obtain individual loggers
    """
    global _logger_initialized, _log_filename, _log_dir
    
    if _logger_initialized:
        return _log_filename
    
    # Create run-specific log directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_dir = os.path.join("log", f"{lib}_{timestamp}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_filename = os.path.join(_log_dir, "run.log")
    
    # Create handlers
    file_handler = logging.FileHandler(_log_filename)
    file_handler.setFormatter(LocalFormatter('[%(levelname)s] %(asctime)s - %(name)s.%(funcName)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter('[%(levelname)s] %(asctime)s - %(name)s.%(funcName)s - %(message)s'))
    
    # Root logger configuration (all child loggers inherit)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers (prevent duplicates)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Increase httpx log level (prevent too many HTTP request logs)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    _logger_initialized = True
    return _log_filename

def get_log_dir() -> str:
    global _log_dir
    return _log_dir or "log"

def get_perf_log_file() -> str:
    global _log_dir
    if _log_dir:
        return os.path.join(_log_dir, "metrics.log")
    return os.path.join("log", "metrics.log")

def get_prompt_log_file() -> str:
    global _log_dir
    if _log_dir:
        return os.path.join(_log_dir, "prompts.log")
    return os.path.join("log", "prompts.log")

def get_token_file() -> str:
    global _log_dir
    if _log_dir:
        return os.path.join(_log_dir, "token_usage.json")
    return os.path.join("log", "token_usage.json")

def get_logger(name: str = None) -> logging.Logger:
    """
    Get module-specific logger (use after calling setup_unified_logger)
    """
    if not _logger_initialized:
        raise RuntimeError("setup_unified_logger must be called first")
    
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)

def get_log_filename() -> str:
    """Return current log filename"""
    return _log_filename or "no_log_file"

# Functions for dashboard compatibility
def disable_all_console_logging():
    """Disable console output for all loggers"""
    disabled_handlers = []
    
    # Remove StreamHandler from root logger
    root_logger = logging.getLogger()
    handlers_to_remove = []
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
        disabled_handlers.append(('root', handler))
    
    # Remove StreamHandler from all child loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger_obj = logging.getLogger(logger_name)
        if hasattr(logger_obj, 'handlers'):
            handlers_to_remove = []
            for handler in logger_obj.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                logger_obj.removeHandler(handler)
                disabled_handlers.append((logger_name, handler))
    
    return disabled_handlers

def enable_console_logging(disabled_handlers):
    """Re-enable console output"""
    for logger_name, handler in disabled_handlers:
        if logger_name == 'root':
            logging.getLogger().addHandler(handler)
        else:
            logging.getLogger(logger_name).addHandler(handler)