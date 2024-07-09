import logging


logger = logging.getLogger('recommence')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('recommence.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('This is an info message')
logger.debug('This is a debug message')

logger.addHandler(handler)
logger.info('This is an info message')
logger.debug('This is a debug message')
