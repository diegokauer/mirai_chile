import warnings

from mirai_chile.model_evaluation.evaluation_functions.abstract_function import AbstractFunction
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)


def compute_auc_at_followup(probs, censor_times, golds, followup, fup_lower_bound=-1):
    golds, censor_times = golds.ravel(), censor_times.ravel()
    if len(probs.shape) == 3:
        probs = probs.reshape(probs.shape[0] * probs.shape[1], probs.shape[2])

    def include_exam_and_determine_label(prob_arr, censor_time, gold):
        valid_pos = gold and censor_time <= followup and censor_time > fup_lower_bound
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(prob_arr, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(label)
    try:
        roc_auc = roc_auc_score(golds_for_eval, probs_for_eval, average="samples")
        ap_score = average_precision_score(
            golds_for_eval, probs_for_eval, average="samples"
        )
        precision, recall, _ = precision_recall_curve(golds_for_eval, probs_for_eval)
        pr_auc = auc(recall, precision)
    except Exception as e:
        warnings.warn("Failed to calculate AUC because {}".format(e))
        roc_auc = -1.0
        ap_score = -1.0
        pr_auc = -1.0
    return roc_auc, ap_score, pr_auc


def get_yearly_roc_auc(data, max_followup, category=None):
    yearly_roc_auc = []
    for followup in range(max_followup):
        probs = data[[f"year_{i + 1}" for i in range(max_followup)]].to_numpy()
        time_to_event = data["time_to_event"]
        cancer = data["cancer"]

        roc_auc, _, _ = compute_auc_at_followup(probs, time_to_event, cancer, followup)
        yearly_roc_auc.append(roc_auc)
    return yearly_roc_auc


class YearlyROCAUCFunction(AbstractFunction):
    def __init__(self, function=get_yearly_roc_auc):
        super().__init__()
        self.name = "Yearly ROC-AUC"
        self.function = function
