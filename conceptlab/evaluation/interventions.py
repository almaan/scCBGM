from conceptlab.evaluation._base import EvaluationClass


def array_to_dataframe(obj: pd.DataFrame | np.ndarray):
    if isinstance(obj,pd.DatFrame):
        return obj
    elif isinstance(obj,np.ndarray):



class DistributionShift(EvaluationClass):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def score(
        cls,
        X_old: np.ndarray | pd.DataFrame,
        X_new: np.ndarray | pd.DataFrame,
        concepts_old : np.ndarray | pd.DataFrame,
        concepts_new : np.ndarray | pd.DataFrame,
        concept_coef: np.ndarray | pd.DataFrame,
    ):


