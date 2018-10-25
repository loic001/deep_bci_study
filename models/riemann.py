import logging
import os
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pyriemann.estimation import XdawnCovariances, Covariances, ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.classification import KNearestNeighbor

from mne.decoding import Vectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class RiemannXdawnTangLog():
    def __init__(self, params):
        self.params = params
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 2)
        self.pipeline = make_pipeline(XdawnCovariances(
            self.n_components), TangentSpace(metric='riemann'), LogisticRegression())

    def fit(self, X_train, y_train, **kwargs):
        self.pipeline.fit(X_train, y_train)
        return {}

    def predict(self, X_test, **kwargs):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test, **kwargs):
        return self.pipeline.predict_proba(X_test)


class RiemannTangLog():
    def __init__(self, params):
        self.params = params
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 2)
        self.pipeline = make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression())

    def fit(self, X_train, y_train, **kwargs):
        self.pipeline.fit(X_train, y_train)
        return {}

    def predict(self, X_test, **kwargs):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test, **kwargs):
        return self.pipeline.predict_proba(X_test)


class RiemannXdawnMDM():
    def __init__(self, params):
        self.params = params
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 2)
        self.pipeline = make_pipeline(XdawnCovariances(self.n_components), MDM())

    def fit(self, X_train, y_train, **kwargs):
        self.pipeline.fit(X_train, y_train)
        return {}

    def predict(self, X_test, **kwargs):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test, **kwargs):
        return self.pipeline.predict_proba(X_test)

class RiemannMDM():
    def __init__(self, params):
        self.params = params
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 2)
        self.pipeline = make_pipeline(ERPCovariances(estimator='oas'), MDM())

    def fit(self, X_train, y_train, **kwargs):
        self.pipeline.fit(X_train, y_train)
        return {}

    def predict(self, X_test, **kwargs):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test, **kwargs):
        return self.pipeline.predict_proba(X_test)


class RegLDA():
    def __init__(self, params):
        self.params = params
        self._init()

    def _init(self):
        self.n_components = self.params.get('n_components', 2)
        self.pipeline = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))

    def fit(self, X_train, y_train, **kwargs):
        self.pipeline.fit(X_train, y_train)
        return {}

    def predict(self, X_test, **kwargs):
        return self.pipeline.predict(X_test)

    def predict_proba(self, X_test, **kwargs):
        return self.pipeline.predict_proba(X_test)
