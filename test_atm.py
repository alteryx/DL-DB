from btb.tuning.gp import GP
from btb.hyper_parameter import HyperParameter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from testing_utils import construct_retail_example
from dldb.dldb import DLDB

HYPERPARAMETER_RANGES = [
        ('recurrent_layer_sizes', HyperParameter('int', [0, 9])),
        ('dense_layer_sizes', HyperParameter('int', [0, 9])),
        ('categorical_max_vocab', HyperParameter('int', [3, 100])),
        ('epochs', HyperParameter('int', [1, 1])),
        ('batch_size', HyperParameter('int', [2, 64])),
        #('dense_activation', HyperParameter('string', ['relu', 'elu', 'selu','softplus','softsign','hard_sigmoid'])),#, 'sigmoid', 'tanh', 'linear'])),
        ('dropout_fraction', HyperParameter('float', [0, 1])),
        ('recurrent_dropout_fraction', HyperParameter('float', [0, 1])),
        ('categorical_embedding_size', HyperParameter('int', [2, 128])),
        #('cell_type', HyperParameter('string', ['lstm', 'gru'])),#, 'conv_lstm', ])),
        #('loss', HyperParameter('string', ['binary_crossentropy', 'hinge'])),#, 'logcosh', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'])),
        #('optimizer', HyperParameter('string', ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])),
]


def test_retail_with_atm():
    fm, labels, fl = construct_retail_example()
    tuner = GP(HYPERPARAMETER_RANGES)
    n = 100
    tested_parameters = np.zeros((n, len(HYPERPARAMETER_RANGES)), dtype=object)
    scores = []
    best_so_far = -np.inf
    best_index = -1
    for i in range(n):
        tuner.fit(
            tested_parameters[:i, :],
            scores,
        )
        hyperparams = tuner.propose()
        cv_score = score_model(fm, labels, fl, hyperparams)
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


def score_model(fm, labels, fl, hyperparams):
    hyperparams = parse_hyperparams(hyperparams)
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

if __name__ == '__main__':
    test_retail_with_atm()
