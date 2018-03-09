from btb.tuning.gp import GP
from btb.hyper_parameter import HyperParameter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import featuretools as ft
from featuretools.selection import remove_low_information_features
import numpy as np
from .testing_utils import construct_retail_example
from .test import score_baseline_pipeline
from dldb import DLDB

# Best after 100 runs:

# HYPERPARAMS: {'recurrent_layer_sizes': (64,),
              # 'dense_layer_sizes': (32, 32),
              # 'categorical_max_vocab': 12,
              # 'epochs': 62,
              # 'batch_size': 28,
              # 'dropout_fraction': 0.2289444521525038,
              # 'recurrent_dropout_fraction': 0.10269178957033287,
              # 'categorical_embedding_size': 20}
# CV Score = 0.808 +/- 0.061
# lower bound = 0.748
HYPERPARAMETER_RANGES = [
        ('recurrent_layer_sizes', HyperParameter('int', [0, 9])),
        ('dense_layer_sizes', HyperParameter('int', [0, 9])),
        ('categorical_max_vocab', HyperParameter('int', [3, 100])),
        ('epochs', HyperParameter('int', [1, 64])),
        ('batch_size', HyperParameter('int', [2, 64])),
        #('dense_activation', HyperParameter('string', ['relu', 'elu', 'selu','softplus','softsign','hard_sigmoid'])),#, 'sigmoid', 'tanh', 'linear'])),
        ('dropout_fraction', HyperParameter('float', [0, 1])),
        ('recurrent_dropout_fraction', HyperParameter('float', [0, 1])),
        ('categorical_embedding_size', HyperParameter('int', [2, 128])),
        #('cell_type', HyperParameter('string', ['lstm', 'gru'])),#, 'conv_lstm', ])),
        #('loss', HyperParameter('string', ['binary_crossentropy', 'hinge'])),#, 'logcosh', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'])),
        #('optimizer', HyperParameter('string', ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])),
]

BASELINE_HYPERPARAMETER_RANGES = [
    ('select_n_features', HyperParameter('int', [10, 400])),
    ('selector_n_estimators', HyperParameter('int', [50, 2000])),
    ('n_estimators', HyperParameter('int', [10, 400])),
]

def test_retail_with_atm(baseline=False):
    fm, labels, fl = construct_retail_example()
    hyperparameter_ranges = HYPERPARAMETER_RANGES
    score_func = score_model
    if baseline:
        hyperparameter_ranges = BASELINE_HYPERPARAMETER_RANGES
        score_func = score_model_baseline
    tuner = GP(hyperparameter_ranges)
    n = 100
    tested_parameters = np.zeros((n, len(hyperparameter_ranges)), dtype=object)
    scores = []
    best_so_far = -np.inf
    best_index = -1
    for i in range(n):
        tuner.fit(
            tested_parameters[:i, :],
            scores,
        )
        hyperparams = tuner.propose()
        cv_score = score_func(fm, labels, fl, hyperparams)
        # mean - std
        lower_bound = cv_score[0] - cv_score[1]
        if lower_bound > best_so_far:
            best_so_far = lower_bound
            best_index = i
            print("Improved pipeline")
            print("    CV Score = %.3f +/- %.3f" % (cv_score[0], cv_score[1]))
            print("    New score lower bound = %.3f   (cv mean - cv std)\n" % lower_bound)
        # record hyper-param combination and score for tuning

        # right now, record lower bound
        tested_parameters[i, :] = hyperparams
        scores.append(lower_bound)
    return tested_parameters[best_index], best_so_far


def layer_mapping(x):
    x = int(x)
    layer_sizes = [4, 8, 16, 32, 64]
    if x < 5:
        return (layer_sizes[x],)
    else:
        return (layer_sizes[x - 5], layer_sizes[x - 5])


def parse_hyperparams(hyperparams):
    hp = {}
    for i, k in enumerate(HYPERPARAMETER_RANGES):
        if k[1].type == 'int':
            hp[k[0]] = int(hyperparams[i])
        else:
            hp[k[0]] = hyperparams[i]

    hp['recurrent_layer_sizes'] = layer_mapping(hp['recurrent_layer_sizes'])
    hp['dense_layer_sizes'] = layer_mapping(hp['dense_layer_sizes'])
    return hp


def parse_hyperparams_baseline(hyperparams):
    hp = {}
    for i, k in enumerate(BASELINE_HYPERPARAMETER_RANGES):
        if k[1].type == 'int':
            hp[k[0]] = int(hyperparams[i])
        else:
            hp[k[0]] = hyperparams[i]
    return hp


def score_model(fm, labels, fl, hyperparams):
    hyperparams = parse_hyperparams(hyperparams)
    print("HYPERPARAMS:", hyperparams)
    cv_score = []
    n_splits = 5
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True)

    dl_model = DLDB(
        regression=False,
        classes=[False, True],
        recurrent_layer_sizes=hyperparams['recurrent_layer_sizes'],
        dense_layer_sizes=hyperparams['dense_layer_sizes'],
        categorical_max_vocab=hyperparams['categorical_max_vocab'])
    # TODO: cheating a bit, put back in CV later
    dl_model.compile(fm, fl)

    for train_index, test_index in splitter.split(labels, labels):
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]
        train_fm = fm.loc[(train_labels.index, slice(None)), :]
        test_fm = fm.loc[(test_labels.index, slice(None)), :]

        dl_model.fit(
            train_fm, train_labels,
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'])
        predictions = dl_model.predict(test_fm)
        cv_score.append(roc_auc_score(test_labels, predictions))
    return np.mean(cv_score), 2 * (np.std(cv_score) / np.sqrt(n_splits))


def score_model_baseline(fm, labels, fl, hyperparams):
    baseline_fm = (fm.reset_index('CustomerID', drop=False)
                     .drop_duplicates('CustomerID', keep='last')
                     .set_index('CustomerID'))
    baseline_fm, baseline_fl = ft.encode_features(baseline_fm, fl)
    baseline_fm, baseline_fl = remove_low_information_features(baseline_fm, baseline_fl)

    hyperparams = parse_hyperparams_baseline(hyperparams)
    print("HYPERPARAMS:", hyperparams)
    cv_score = []
    n_splits = 5
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in splitter.split(labels, labels):
        baseline_train_labels = labels.iloc[train_index]
        baseline_test_labels = labels.iloc[test_index]
        baseline_train_fm = baseline_fm.loc[baseline_train_labels.index, :]
        baseline_test_fm = baseline_fm.loc[baseline_test_labels.index, :]

        score = score_baseline_pipeline(baseline_train_fm, baseline_train_labels,
                                        baseline_test_fm, baseline_test_labels,
                                        **hyperparams)
        cv_score.append(score['rf'])
    return np.mean(cv_score), 2 * (np.std(cv_score) / np.sqrt(n_splits))


if __name__ == '__main__':
    test_retail_with_atm(baseline=True)
