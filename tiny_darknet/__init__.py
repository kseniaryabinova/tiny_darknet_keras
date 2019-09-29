import argparse
import logging
import logging.config as logging_config
import yaml
import os

parser = argparse.ArgumentParser(usage="python3 -m tiny_darknet [options]")
parser.add_argument("config")
args = parser.parse_args()

config_path = args.config
if not os.path.exists(config_path):
    raise Exception('config path {} does not exist!'.format(config_path))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(message)s')


with open(config_path) as file:
    try:
        config = yaml.load(file)

        if 'logging' in config:
            logging_config.dictConfig(config['logging'])
        logging.warning('Config loaded from %s', config_path)
    except yaml.YAMLError as exception:
        raise exception
