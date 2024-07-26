from abc import ABCMeta, abstractmethod,abstractclassmethod
from typing import Dict, Any


class BaseDataGenerator(ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def generate(cls, n_obs: int,**kwargs):
        pass



