import logging
from EA.EAAFE.EAAFE import SEARCH_FE_PIPELINE
logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

class AutoFE:
    def __init__(self, config=None, X=None, y=None, schema=None):
        self.config = config.config
        self.config['X'] = X
        self.config['y'] = y
        self.config['schema'] = schema

    def run(self):
        logger.info('start run')
        if self.config['search_method'] == "EA":
            logger.info('EA')
            self.autofe = SEARCH_FE_PIPELINE(config=self.config)
            self.autofe.run()
        else:
            pass

    def transform_sequence(self):
        logger.info('start transform sequence')
        transformed_data = None
        if self.config['search_method'] == "EA":
            logger.info('EA')
            self.autofe = SEARCH_FE_PIPELINE(config=self.config)
            transformed_data = self.autofe.transform_sequence()
        else:
            pass
        return transformed_data
    
    def save(self):
        logger.info('save')
