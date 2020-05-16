import configparser
import os


class LoadConfigFile:
    def __init__(self, config_fpath):
        self.config_fpath = config_fpath

    def read_config_file(self):
        config = configparser.ConfigParser()
        print(os.path.realpath(self.config_fpath))
        config.read("%s" % os.path.realpath(self.config_fpath), encoding="utf-8")
        return config
