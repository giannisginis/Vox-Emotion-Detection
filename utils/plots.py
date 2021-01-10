from sklearn import metrics
import matplotlib.pyplot as plt
import os
from itertools import cycle
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


class Plots:

    def __init__(self, outpath):
        self.outpath = outpath
        os.makedirs(self.outpath, exist_ok=True)

    def plot_confusion_matrix(self, title, outname, **kwargs):
        disp = metrics.ConfusionMatrixDisplay(**kwargs)
        disp.plot()
        disp.ax_.set_title(title)
        fname = "/".join((self.outpath, outname))
        plt.savefig(fname)

    def plot_roc_curve(self, title, outname, **kwargs):
        """ This method works only for binary classification"""
        disp = metrics.RocCurveDisplay(**kwargs)
        disp.ax_.set_title(title)
        disp.plot()
        fname = "/".join((self.outpath, outname))
        plt.savefig(fname)

    def plot_roc_curve_multiclass(self, title, outname, fpr, tpr, roc_auc, n_classes, lw=2):

        # Plot all ROC curves
        plt.figure()

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        fname = "/".join((self.outpath, outname))
        plt.savefig(fname)
        plt.close()

    def plot_learning_curves(self, clf):  # TODO: not implemented
        results = clf.cv_results_

        # The scorers can be either be one of the predefined metric strings or a scorer
        # callable, like the one returned by make_scorer
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

        plt.figure(figsize=(13, 13))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
                  fontsize=16)

        plt.xlabel("min_samples_split")
        plt.ylabel("Score")

        ax = plt.gca()
        ax.set_xlim(0, 402)
        ax.set_ylim(0.73, 1)

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

        for scorer, color in zip(sorted(scoring), ['g', 'k']):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]
                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)
                ax.plot(X_axis, sample_score_mean, style, color=color,
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()

        fname = "/".join((self.outpath, 'asfsda.png'))
        plt.savefig(fname)
        plt.close()