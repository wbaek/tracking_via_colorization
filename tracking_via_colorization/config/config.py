# -*- coding: utf-8 -*-
import os
import logging
import yaml


LOGGER = logging.getLogger(__name__)

class Config():
    _instance = None

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config()
        return Config._instance

    @staticmethod
    def clear():
        Config._instance = None

    def dump(self, filename=None):
        dump_string = yaml.dump(self.conf)
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(dump_string)
        return dump_string

    def __init__(self, filename=None):
        if Config._instance is not None:
            raise Exception('This class is a singleton!')

        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filename = os.path.join(root_path, 'configs', 'basic.yaml') if filename is None else filename
        LOGGER.info('load config at: %s', filename)
        self.filename = filename

        with open(filename, 'r') as f:
            self.conf = yaml.load(f)
        Config._instance = self

    def __str__(self):
        return 'filename:%s\nconf:%s' % (self.filename, self.conf)

    def __getitem__(self, key):
        return self.conf[key]

    def __setitem__(self, key, value):
        self.conf[key] = value
