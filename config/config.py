import os
import yaml


class Config:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.config_data = self._load_config()

    def _load_config(self):
        with open(self.config_file, "r") as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config_data
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value


# 创建全局配置实例
app_config = Config(config_file=os.path.join(os.path.dirname(__file__), "config.yaml"))
