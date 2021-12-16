import yaml
from easydict import EasyDict as edict


def read_config(type):
    with open('configs/{}.yml'.format(type.lower())) as f:
        config = edict(yaml.load(f))
        return config
