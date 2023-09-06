import logging
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': 'yellow',
        'INFO': 'magenta',
        'DEBUG': 'blue',
        'CRITICAL': 'red',
        'ERROR': 'red'
    }

    def format(self, record):
        log_message = super(ColoredFormatter, self).format(record)
        return colored(log_message, self.COLORS.get(record.levelname))

logger = logging.getLogger(__name__)

termie_handler = logging.StreamHandler()
file_handler   = logging.FileHandler('app.log')

termie_format = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
file_format   = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

termie_handler.setFormatter(termie_format)
file_handler.setFormatter(file_format)

termie_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

logger.addHandler(termie_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)
