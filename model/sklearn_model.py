from model.base_model import BaseModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import pickle
import numpy as np
import os
from utils.plots import Plots
from utils.metrics import Metrics
import psutil


class Sklearn(BaseModel, Plots, Metrics):
    """Sklearn Model Class"""

    def __init__(self, cfg, x_train, x_test, y_train, y_test):
        super().__init__(cfg)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_val = None
        self.y_val = None
        self.clf = None
        self.predictions = None
        self.folds = self.config["train"]["folds"]
        self.groups = None
        self.val_fold_scores_ = []

        Plots.__init__(self, outpath=self.config["data"]["metrics_path"])

    def _build(self):
        define_clf = dict(
            lr=LogisticRegression,
            rfc=RandomForestClassifier,
            xgb=XGBClassifier,
            lgb=LGBMClassifier
        )
        params = self.config["model"]
        if self.config["train"]["classifier"] not in define_clf:
            raise ValueError("{} not suported in VOX-Emo sklearn models : {}".format(self.config["train"]["classifier"],
                                                                                     ", ".join(define_clf.keys())))
        self.clf = define_clf[self.config["train"]["classifier"]](**params) if params is not None else define_clf[
            self.config["train"]["classifier"]]()

    def train(self, save2disk=False):
        # Build Model
        self._build()

        # Training
        if self.config["train"]["grid_search"]["apply"]:
            if not self._can_perform_cv(self.config["param_grid"], self.clf):
                fail_msg = ("ERROR : impossible to cross validate the model `{}` "
                            "with the parameters {}".format(self.config["train"]["classifier"],
                                                            list(self.config["param_grid"].keys())))

                raise ValueError(fail_msg)
            self._apply_optimization(method="RandomizedSearch", plot=self.config["train"]["grid_search"]["plot"])

        else:
            self._define_jobs(self.clf, self.config["train"]["n_jobs"])
            self.clf.fit(self.x_train, self.y_train)

        # Save Model
        if save2disk:
            self._save_model()

    def train_kfold(self, save2disk=False):

        # Build Model
        self._build()

        # Training with cross validation

        logo = LeaveOneGroupOut()
        self._create_groups()
        for train_index, test_index in logo.split(self.x_train, self.y_train, self.groups):
            x_train, x_test = self.x_train[train_index], self.x_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]

            self._define_jobs(self.clf, self.config["train"]["n_jobs"])
            self.clf.fit(x_train, y_train)
            self._predict(x_test)

            fold_acc = accuracy_score(y_test, self.predictions)
            self.val_fold_scores_.append(fold_acc)

        print('Train accuracy is {}'.format(np.mean(self.val_fold_scores_)))

        # Save model
        if save2disk:
            self._save_model()
        return self.val_fold_scores_

    def evaluate(self, save2disk=False, load_model=False, plot=True):

        if load_model:
            self._load_model()

        self._predict(self.x_test)

        if self.config['train']['encoder'] == 'OneHotEncoder':
            self.y_test = self.y_test.argmax(axis=1)
            self.predictions = self.predictions.argmax(axis=1)

        fpr, tpr, roc_auc = self.auc_roc_multiclass(n_classes=self.clf.classes_.shape[0], y_test=self.y_test,
                                                    y_score=self.probas)

        print('Test accuracy is {}'.format(accuracy_score(self.y_test, self.predictions)))
        print('Test f1 score is {}\n'.format(f1_score(self.y_test, self.predictions, average='weighted')))
        print('Classification Report')
        print(classification_report(self.y_test, self.predictions))

        cm = self.confusion_matrix(y_true=self.y_test, y_pred=self.predictions, labels=self.clf.classes_)

        if plot:
            self.plot_confusion_matrix(title=self.config["train"]["classifier"] + '_confusion_matrix',
                                       outname=self.config["train"]["classifier"] + '_confusion_matrix.png',
                                       confusion_matrix=cm, display_labels=self.clf.classes_)
            self.plot_roc_curve_multiclass(title=self.config["train"]["classifier"] + '_roc_curve',
                                           outname=self.config["train"]["classifier"] + '_roc_curve.png', fpr=fpr,
                                           tpr=tpr, roc_auc=roc_auc, n_classes=self.clf.classes_.shape[0])

    def _predict(self, eval_set):
        self.predictions = self.clf.predict(eval_set)
        self.probas = self.clf.predict_proba(eval_set)

    def _save_model(self):
        # save the model to disk
        filename = './pretrained_models/' + 'finalized_model_' + self.config["train"]["classifier"] + '.sav'
        # if directory already exists leaves it unaltered and saves the file inside.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(self.clf, open(filename, 'wb'))

    def _load_model(self):
        # load a model from disk
        filename = 'finalized_model_' + self.config["train"]["classifier"] + '.sav'
        self.clf = pickle.load(open('./pretrained_models/' + filename, 'rb'))
        if self.config["train"]["grid_search"]["apply"]:
            filename = 'finalized_model_' + self.config["train"]["classifier"] + "_" + \
                       self.config["train"]["grid_search"]["method"] + '.sav'
            self.grid_search = pickle.load(open('./pretrained_models/' + filename, 'rb'))

    def _create_groups(self):
        # create the proper groups array for the cross validation
        group_list = []
        step = (self.y_train.shape[0] // 10) + 1
        counter, group = 0, 0
        for i in range(self.y_train.shape[0]):
            if counter == step:
                group += 1
                counter = 0
            group_list.append(group)
            counter += 1
        self.groups = np.array(group_list)

    def _apply_optimization(self, method="GridSearch", plot=False):

        self.grid_search = None
        # Use stratification within KFold Split inside GridSearchCV
        kf = StratifiedKFold(**self.config['kf_dict'])

        # Parameter Optimization
        if method == "GridSearch":
            self.grid_search = GridSearchCV(estimator=self.clf,
                                            param_grid=self.config['param_grid'],
                                            cv=kf,
                                            return_train_score=True,
                                            verbose=1,
                                            **self.config['grid_dict'])
        elif method == "RandomizedSearch":
            self.grid_search = RandomizedSearchCV(estimator=self.clf,
                                                  param_distributions=self.config['param_grid'],
                                                  cv=kf,
                                                  return_train_score=True,
                                                  verbose=1,
                                                  **self.config['grid_dict'])

        # refit the best estimator on the FULL train set
        self.grid_search.fit(self.x_train, self.y_train)
        self.clf = self.grid_search.best_estimator_

        if self.config["train"]["save2disk"]:
            filename = './pretrained_models/' + 'finalized_model_' + self.config["train"]["classifier"] + "_" + \
                       self.config["train"]["grid_search"]["method"] + '.sav'
            pickle.dump(self.grid_search, open(filename, 'wb'))

        if plot:
            self.plot_learning_curves(self.grid_search)  # TODO: bug

    @staticmethod
    def _can_perform_cv(cv_paramerters, clf) -> bool:
        """check if the cross validation can be done
        Parameters
        ----------
        cv_paramerters : dict
            model parameters to estimate
        clf : ABCMeta
            scikit learn estimator class
        Return
        ------
        bool
            True if all user cross validation parameters can be estimate, else False
        """
        # get_params is comming from scikit-learn BaseEstimator class -> Base class for all estimators
        user_cv_parameters = list(cv_paramerters.keys())
        avail_cv_parameters = list(clf.get_params(clf).keys())
        check = [user_cv_parameter in avail_cv_parameters for user_cv_parameter in user_cv_parameters]

        return all(check)

    @staticmethod
    def _define_jobs(clf, jobs):
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = jobs
        if hasattr(clf, "pre_dispatch"):
            clf.pre_dispatch = jobs
