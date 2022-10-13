import logging
import os


def setup_logging(file_name):
    from importlib import reload
    logging.shutdown()
    reload(logging)

    log_path = "logs"

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(log_path, file_name)),
            logging.StreamHandler(),
        ],
    )
