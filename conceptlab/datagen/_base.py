from abc import ABCMeta, abstractclassmethod


class DataGenerator(ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def generate(cls, n_obs: int, **kwargs):
        pass
