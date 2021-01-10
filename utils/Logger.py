import logging
import os


class LogSystem:
    """ Generates a new logger """

    def __init__(self, log_dir=None, name=None, log_name=None):
        self.name = "/".join((log_dir, name))
        self.log_name = log_name
        self.log_dir = log_dir
        self.logging = None

        self._create_dir(self.log_dir)
        self._initialize_logger()

    @staticmethod
    def _create_dir(log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _initialize_logger(self):
        logger = logging.getLogger(self.log_name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

        file_handler = logging.FileHandler(self.name, mode='a')
        file_handler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        self.logging = logger

    def log_info(self, message):
        try:
            self.logging.info(message)
        except Exception as e:
            print(f'Error when trying to log file {str(self.name)} {str(e)}')

    def log_error(self, message):
        try:
            self.logging.error(message)
        except Exception as e:
            print(f'Error when trying to log file {str(self.name)} {str(e)}')
