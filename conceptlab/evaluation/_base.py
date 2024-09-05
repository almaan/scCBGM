from abc import ABCMeta, abstractmethod, abstractclassmethod
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt




class EvaluationClass(ABCMeta):
    def __init__(self,*args,**kwargs):
        pass

    @abstractclassmethod
    def score(cls,*args,**kwargs) -> Dict[str,Any]:
        pass

    @abstractclassmethod
    def save(cls,*args,**kwargs) -> None:
        pass


