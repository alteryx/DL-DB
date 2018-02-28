from featuretools.tests.testing_utils import make_ecommerce_entityset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import featuretools as ft

from featuretools.variable_types import Numeric
from model.build_keras_rnn import build_keras_rnn


def test_ecommerce_binary():
    es = make_ecommerce_entityset()
    cutoffs = es['log'].df[['session_id', 'datetime']]
    cutoffs = cutoffs.rename(columns={'session_id': 'id'})
    fm, fl = ft.dfs(entityset=es,
                    cutoff_time=cutoffs,
                    target_entity="sessions",
                    cutoff_time_in_index=True)
    fm.sort_index(inplace=True)
    labels = pd.Series(np.random.randint(2, size=(fm.shape[0],))).astype(bool)

    train_fm, test_fm, train_labels, test_labels = train_test_split(
        fm, labels, test_size=0.4, shuffle=False)

    dl_model, input_transform = build_keras_rnn(
        fm, fl, labels, categorical_max_vocab=10)

    train_inputs, train_outputs = input_transform(train_fm, train_labels)
    dl_model.fit(train_inputs, train_outputs, epochs=1, batch_size=4)

    test_inputs = input_transform(test_fm)
    predictions = dl_model.predict(test_inputs)
    return predictions

if __name__ == '__main__':
    test_ecommerce_binary()
