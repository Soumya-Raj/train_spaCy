import configparser


class LoadConfigFile:

    def __init__(self):
        pass

    def read_config_file(self, config_fname):
        config = configparser.ConfigParser()
        config.read("config\\%s" % config_fname, encoding="utf-8")
        return config
