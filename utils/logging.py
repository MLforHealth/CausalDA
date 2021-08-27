import logging
import os
import sys

def setup_logs(save_dir, run_name):
    
    # initialize logger
    logger = logging.getLogger("causal_bootstrap")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler(stream=sys.stdout)

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger