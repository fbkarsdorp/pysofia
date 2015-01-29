import cStringIO as StringIO

import numpy as np
from sklearn import base, datasets
from sklearn.metrics import f1_score
import _sofia_ml


def sgd_train(name, X, y, query_id, alpha, n_features=0, eta_type='pegasos',
              model="stochastic", max_iter=100000, step_probability=0.5,
              learner='pegasos'):
    if isinstance(X, basestring):
        coef = _sofia_ml.train(X, n_features, alpha, max_iter, True,
                               model, step_probability, eta_type, learner)
    else:
        f = StringIO.StringIO()
        datasets.dump_svmlight_file(X, y, f, zero_based=False, query_id=query_id)
        coef = _sofia_ml.train(f.getvalue(), n_features, alpha, max_iter, True,
                               model, step_probability, eta_type, learner)
        f.close()
    return coef

def sgd_predict(name, X, coef, y=None):
    string_coef = ' '.join('%.7f' % e for e in coef)
    if isinstance(X, basestring):
        prediction = _sofia_ml.predict(X, string_coef, True)
    else:
        f = StringIO.StringIO()
        y = np.ones(X.shape[0]) if y is None else y
        datasets.dump_svmlight_file(X, y, f, zero_based=False, query_id=np.ones(X.shape[0]))
        prediction = _sofia_ml.predict(f.getvalue(), string_coef, True)
        f.close()
    return prediction


class SofiaML(base.BaseEstimator):
    def __init__(self, name="", alpha=0.1, learner='pegasos', model="stochastic",
                 eta_type='pegasos', n_features=500000, max_iter=100000,
                 step_probability=0.5):
        self.name = name
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta_type = eta_type
        self.learner = learner
        self.model = model
        self.n_features = n_features
        self.step_probability = step_probability

    def fit(self, X, y=None, query_id=None):
        self.coef_ = sgd_train(
            self.name + '-train.svm', X, y, query_id, self.alpha, n_features=self.n_features,
            max_iter=self.max_iter, model=self.model,
            step_probability=self.step_probability, eta_type=self.eta_type,
            learner=self.learner)
        return self

    def decision_function(self, X, y=None):
        return sgd_predict(self.name + '-test.svm', X, self.coef_, y)

    def rank(self, X, y=None):
        ranking = self.decision_function(X, y)
        order = np.argsort(ranking)[::-1]
        return order

    predict = rank

    def score(self, X, y, cutoff=0):
        preds = self.decision_function(X)
        preds[preds>cutoff] = 1
        preds[preds<cutoff] = -1
        return f1_score(y, preds)

