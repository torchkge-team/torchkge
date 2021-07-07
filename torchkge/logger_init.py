import logging
import os
import datetime
from torchkge.torchkge.config import config, dump_config


def logger_init():
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if config().log.to_file:
        log_filename = os.path.join(config().log.dir,
                                    config().log.prefix + datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogger().addHandler(logging.FileHandler(log_filename))
    if config().log.dump_config:
        dump_config()
