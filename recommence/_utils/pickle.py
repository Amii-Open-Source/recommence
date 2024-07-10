import pickle
import logging

logger = logging.getLogger('recommence')

def read_pickle(path: str, no_fail: bool):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        logger.exception(f'Failed to load checkpoint at {path}')
        if not no_fail:
            raise e
