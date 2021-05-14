'''
Provides a logger with clear format setting and log files recording

Please install coloredlogs for better display.
'''
import sys
import logging

# Clean existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up handlers
LOGGING_LEVEL = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

# Try to use colored formatter from coloredlogs
try:
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    stream_handler.setFormatter(formatter)
except ModuleNotFoundError as err:
    pass

handlers = [
    stream_handler
]
logging.basicConfig(
    format=format_,
    level=LOGGING_LEVEL,
    handlers=handlers
)
logger = logging.getLogger(__name__)
