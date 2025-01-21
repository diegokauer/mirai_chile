import numpy as np
from mirai_chile.model_evaluation.evaluation_functions.abstract_function import AbstractFunction
from pycox.evaluation.concordance import concordance_td


class TimeDependantConcordanceIndex(AbstractFunction):
    def __init__(self, function=concordance_td):
        super().__init__()
        self.name = "Time Dependant Concordance Index"
        self.function = function

    def __call__(self, data, *args, **kwargs):
        probs = data[[col for col in data.columns if "year" in col]].to_numpy()
        time_to_event = data["time_to_event"].to_numpy()
        cancer = data["cancer"].to_numpy()

        cancer[time_to_event >= probs.shape[1]] = 0
        time_to_event = np.clip(time_to_event, 0, probs.shape[1] - 1)

        surv = 1 - probs[np.arange(len(probs)), time_to_event]
        surv = np.expand_dims(surv, axis=0)

        return concordance_td(time_to_event, cancer, surv, time_to_event)
