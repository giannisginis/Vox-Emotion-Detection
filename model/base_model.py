# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = cfg

    @abstractmethod
    def _build(self, **kwargs):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
