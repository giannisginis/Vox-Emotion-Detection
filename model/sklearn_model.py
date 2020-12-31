from model.base_model import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import numpy as np


class Sklearn(BaseModel):
    """Sklearn Model Class"""

    def __init__(self, cfg, x_train, x_test, y_train, y_test, classifier):
        super().__init__(cfg)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = None
        self.predictions = None
        self.classifier = classifier

    def _build(self, *args):
        define_clf = dict(
            lr=LogisticRegression,
            gnb=GaussianNB,
            svc=LinearSVC,
            rfc=RandomForestClassifier
        )
        self.clf = define_clf[self.classifier](*args)
        self.clf = RandomForestClassifier()

    def train(self):
        self._build(self.config.model)
        self.clf.fit(self.x_train, self.y_train)

    def predict(self):
        self.predictions = self.clf.predict(self.x_test)

    def evaluate(self, holdout_type='eval'):

        if len(self.y_test.shape):
            self.y_test = self.y_test.argmax(axis=1)
            self.predictions = self.predictions.argmax(axis=1)

        print('Test accuracy is {}'.format(accuracy_score(self.y_test, self.predictions)))
        print('Test f1 score is {}\n'.format(f1_score(self.y_test, self.predictions, average='weighted')))
        print('Classification Report')
        print(classification_report(self.y_test, self.predictions))
