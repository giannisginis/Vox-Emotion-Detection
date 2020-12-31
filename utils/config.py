# -*- coding: utf-8 -*-
"""Config class"""

import yaml


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, cfg):
        self.config = cfg

    @classmethod
    def from_yaml(cls, cfg):
        """Creates config from YAML"""
        with open(cfg, "r") as ymlfile:
            params = yaml.safe_load(ymlfile)

        return cls(params)