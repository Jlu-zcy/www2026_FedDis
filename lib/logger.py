import os
import logging
import pandas as pd 

def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)       
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger

class PD_Stats(object):
    def __init__(self, path, columns):
        self.path = path
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)
    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row
        if save:
            self.stats.to_pickle(self.path)