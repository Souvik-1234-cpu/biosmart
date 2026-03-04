# ============================================================
# wrappers.py  —  BioSmart v5
# sklearn-compatible CatBoost wrappers with GPU + full param support.
# ============================================================

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, iterations=200, depth=6, learning_rate=0.1,
                 random_state=42, verbose=0,
                 task_type="CPU", devices="0",
                 l2_leaf_reg=3, bagging_temperature=1.0,
                 auto_class_weights=None):          # v5.2: class imbalance support
        self.iterations          = iterations
        self.depth               = depth
        self.learning_rate       = learning_rate
        self.random_state        = random_state
        self.verbose             = verbose
        self.task_type           = task_type
        self.devices             = devices
        self.l2_leaf_reg         = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.auto_class_weights  = auto_class_weights

    def _build(self):
        kwargs = dict(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state, verbose=self.verbose,
            task_type=self.task_type, devices=self.devices,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
        )
        if self.auto_class_weights is not None:
            kwargs["auto_class_weights"] = self.auto_class_weights
        return CatBoostClassifier(**kwargs)

    def fit(self, X, y, **kw):
        self._clf = self._build()
        self._clf.fit(X, y, **kw)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._clf.predict(X).flatten().astype(int)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=200, depth=6, learning_rate=0.1,
                 random_state=42, verbose=0,
                 task_type="CPU", devices="0",
                 l2_leaf_reg=3, bagging_temperature=1.0):
        self.iterations         = iterations
        self.depth              = depth
        self.learning_rate      = learning_rate
        self.random_state       = random_state
        self.verbose            = verbose
        self.task_type          = task_type
        self.devices            = devices
        self.l2_leaf_reg        = l2_leaf_reg
        self.bagging_temperature= bagging_temperature

    def _build(self):
        return CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state, verbose=self.verbose,
            task_type=self.task_type, devices=self.devices,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
        )

    def fit(self, X, y, **kw):
        self._reg = self._build()
        self._reg.fit(X, y, **kw)
        return self

    def predict(self, X):
        return self._reg.predict(X)
