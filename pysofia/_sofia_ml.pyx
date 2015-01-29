from cython.operator cimport dereference as deref 
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np 
import numpy as np

cdef int BUFFER_MB = 40
cdef int DIMENSIONALITY = 100000

cdef extern from "src/sofia-ml-methods.h":
    cdef cppclass SfDataSet:
        SfDataSet(bool)
        SfDataSet(string, int, bool)
        SfDataSet(string, int, bool, bool)

    cdef cppclass SfWeightVector:
        SfWeightVector(int)
        SfWeightVector(string)
        string AsString()
        float ValueOf(int)

cdef extern from "src/sofia-ml-methods.h" namespace "sofia_ml":
    cdef enum LearnerType:
        PEGASOS, MARGIN_PERCEPTRON, PASSIVE_AGGRESIVE, LOGREG_PEGASOS,
        LOGREG, LMS_REGRESSION, SGD_SVM, ROMMA

    cdef enum EtaType:
        BASIC_ETA
        PEGASOS_ETA
        CONSTANT

    void StochasticOuterLoop(SfDataSet, LearnerType, EtaType,
        float, float, int, SfWeightVector*)

    void BalancedStochasticOuterLoop(SfDataSet, LearnerType, EtaType,
        float, float, int, SfWeightVector*)

    void StochasticRocLoop(SfDataSet, LearnerType, EtaType,
        float, float, int, SfWeightVector*)

    void StochasticClassificationAndRocLoop(SfDataSet, LearnerType, EtaType,
        float, float, float, int, SfWeightVector*)

    void StochasticClassificationAndRankLoop(SfDataSet, LearnerType, EtaType,
        float, float, float, int, SfWeightVector*)

    void StochasticRankLoop(SfDataSet, LearnerType, EtaType,
        float, float, int, SfWeightVector*)

    void StochasticQueryNormRankLoop(SfDataSet, LearnerType, EtaType,
        float, float, int, SfWeightVector*)

    void SvmPredictionsOnTestSet(SfDataSet, SfWeightVector, vector[float]*)


def train(train_data, int n_features, float alpha, int max_iter, 
          bool fit_intercept, model, float step_probability, eta_type, learner_type):
    cdef SfDataSet *data = new SfDataSet(train_data, BUFFER_MB, fit_intercept, True)
    if n_features == 0:
        n_features = DIMENSIONALITY
    cdef SfWeightVector *weights = new SfWeightVector(n_features)
    cdef float c = 0.0
    cdef int i
    cdef EtaType eta
    cdef LearnerType learner

    if eta_type == 'pegasos':
        eta = PEGASOS_ETA
    elif eta_type == 'basic':
        eta = BASIC_ETA
    elif eta_type == 'constant':
        eta = CONSTANT
    else:
        raise NotImplementedError("No eta type implemented with name %s" % eta_type)

    if learner_type == 'pegasos':
        learner = PEGASOS
    elif learner_type == 'sgd-svm':
        learner = SGD_SVM
    else:
        raise NotImplementedError("No learner type implemented with name %s" % learner_type)

    if model == 'stochastic':
        StochasticOuterLoop(deref(data), learner, eta, alpha, c,
                            max_iter, weights)
    elif model == 'balanced-stochastic':
        BalancedStochasticOuterLoop(deref(data), learner, eta, alpha, c,
                                    max_iter, weights)
    elif model == 'rank':
        StochasticRankLoop(deref(data), learner, eta, alpha, c,
                           max_iter, weights)
    elif model == 'roc':
        StochasticRocLoop(deref(data), learner, eta, alpha, c,
                          max_iter, weights)
    elif model == 'query-norm-rank':
        StochasticQueryNormRankLoop(deref(data), learner, eta, alpha, c,
                                    max_iter, weights)
    elif model == 'combined-ranking':
        StochasticClassificationAndRankLoop(deref(data), learner, eta,
            alpha, c, step_probability, max_iter, weights)
    else:
        raise NotImplementedError
        # TODO: implement other loop types and combinations
        # with different learner types.
    cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
    for i in range(n_features):
        coef[i] = weights.ValueOf(i)
    return coef

def predict(test_data, string coef, bool fit_intercept):
    cdef SfDataSet *data = new SfDataSet(test_data, BUFFER_MB, fit_intercept, True)
    cdef SfWeightVector *weights = new SfWeightVector(coef)
    cdef vector[float] *predictions = new vector[float]()
    SvmPredictionsOnTestSet(deref(data), deref(weights), predictions)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] out = np.empty(predictions.size())
    for i in range(predictions.size()):
        out[i] = predictions.at(i)
    return out
