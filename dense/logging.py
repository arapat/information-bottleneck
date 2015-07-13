import logging
import logging.config
 
LOGGING_CONFIG = {
    'version': 1, # required
    'disable_existing_loggers': True, # this config overrides all other loggers
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s -- %(message)s'
        },
        'whenAndWhere': {
            'format': '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
        }
    },
    'handlers': {
        'HTTPSHandler': {
            'class': 'loggly.handlers.HTTPSHandler',
            'formatter': 'whenAndWhere',
            'url': 'https://logs-01.loggly.com/inputs/%s/tag/python' % LOGGLY_TOKEN
        }
    },
    'loggers': {
        '': { # 'root' logger
            'level': 'INFO',
            'handlers': ['HTTPSHandler']
        }
    }
}
 
logging.config.dictConfig(LOGGING_CONFIG)
