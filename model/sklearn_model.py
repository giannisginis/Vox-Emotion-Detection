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


class Sklearn(BaseModel):
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

    def _build(self):
        define_clf = dict(
            lr=LogisticRegression,
            rfc=RandomForestClassifier,
            xgb=XGBClassifier,
            lgb=LGBMClassifier
        )
        params = self.config["model"]
        self.clf = define_clf[self.config["train"]["classifier"]](**params) if params is not None else define_clf[
            self.config["train"]["classifier"]]()

    def train(self, save2disk=False):
        # Build Model
        self._build()

        # Training
        if self.config["train"]["grid_search"]:
            self._apply_optimization(method="RandomizedSearch")
        else:
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

            self.clf.fit(x_train, y_train)
            self._predict(x_test)

            fold_acc = accuracy_score(y_test, self.predictions)
            self.val_fold_scores_.append(fold_acc)

        print('Train accuracy is {}'.format(np.mean(self.val_fold_scores_)))

        # Save model
        if save2disk:
            self._save_model()
        return self.val_fold_scores_

    def evaluate(self, save2disk=False, load_model=False):

        if load_model:
            self._load_model()

        self._predict(self.x_test)

        if self.config['train']['encoder'] == 'OneHotEncoder':
            self.y_test = self.y_test.argmax(axis=1)
            self.predictions = self.predictions.argmax(axis=1)

        print('Test accuracy is {}'.format(accuracy_score(self.y_test, self.predictions)))
        print('Test f1 score is {}\n'.format(f1_score(self.y_test, self.predictions, average='weighted')))
        print('Classification Report')
        print(classification_report(self.y_test, self.predictions))

    def _predict(self, eval_set):
        self.predictions = self.clf.predict(eval_set)

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

    def _apply_optimization(self, method="GridSearch"):

        # Use stratification within KFold Split inside GridSearchCV

        kf = StratifiedKFold(**self.config['kf_dict'])

        # Parameter Optimization
        if method == "GridSearch":
            grid_search = GridSearchCV(estimator=self.clf,
                                       param_grid=self.config['param_grid'],
                                       cv=kf,
                                       return_train_score=True,
                                       verbose=1,
                                       **self.config['grid_dict'])
        elif method == "RandomizedSearch":
            grid_search = RandomizedSearchCV(estimator=self.clf,
                                             param_distributions=self.config['param_grid'],
                                             cv=kf,
                                             return_train_score=True,
                                             verbose=1,
                                             **self.config['grid_dict'])

        # refit the best estimator on the FULL train set
        grid_search.fit(self.x_train, self.y_train)
        self.clf = grid_search.best_estimator_
