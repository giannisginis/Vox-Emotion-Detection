from sklearn import metrics
import numpy as np
from utils.Exceptions import InvalidKey


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def det_curve(**kwargs):
        return metrics.det_curve(**kwargs)

    @staticmethod
    def roc_curve(**kwargs):
        return metrics.roc_curve(**kwargs)

    @staticmethod
    def roc_auc_score(**kwargs):
        return metrics.roc_auc_score(**kwargs)

    @staticmethod
    def average_precision_score(**kwargs):
        return metrics.average_precision_score(**kwargs)

    @staticmethod
    def precision_recall_curve(**kwargs):
        return metrics.precision_recall_curve(**kwargs)

    @staticmethod
    def accuracy_score(**kwargs):
        return metrics.accuracy_score(**kwargs)

    @staticmethod
    def f1_score(**kwargs):
        return metrics.f1_score(**kwargs)

    @staticmethod
    def balanced_accuracy_score(**kwargs):
        return metrics.balanced_accuracy_score(**kwargs)

    @staticmethod
    def classification_report(**kwargs):
        return metrics.classification_report(**kwargs)

    @staticmethod
    def confusion_matrix(**kwargs):
        return metrics.confusion_matrix(**kwargs)

    @staticmethod
    def auc(**kwargs):
        return metrics.auc(**kwargs)

    def auc_roc_multiclass(self, n_classes, y_test, y_score):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = self.roc_curve(y_true=np.int32(y_test == i),
                                                   y_score=y_score[:, i], pos_label=0)
            except InvalidKey:
                continue
            roc_auc[i] = self.auc(x=fpr[i], y=tpr[i])

        return fpr, tpr, roc_auc
