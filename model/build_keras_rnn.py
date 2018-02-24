# import featuretools as ft
from featuretools.variable_types import Discrete
# import pandas as pd
from keras.layers import Dense, Activation, LSTM, GRU, Embedding, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras
import numpy as np

RNN_CELLS = {
    'lstm': LSTM,
    'gru': GRU,
}


def build_keras_rnn(fm, fl, labels,
                    cell_type='lstm',
                    recurrent_layer_sizes=(64, 64),
                    dense_layer_sizes=(10,),
                    dense_activation='relu',
                    dropout_fraction=0.2,
                    recurrent_dropout_fraction=0.2,
                    categorical_max_vocab=None,
                    categorical_embedding_size=10,
                    loss=None,
                    metrics=None,
                    optimizer='rmsprop'):
    '''
    fm (pd.DataFrame): Time-varying feature matrix with multiple time points per instance. Can contain both categorical
        as well as numeric features.
    fl (list[ft.PrimitiveBase]): List of feature objects representing each column in fm
    labels (pd.Series): Labels indexed by same instance and time and fm
    cell_type (str or keras.layers.Layer, optional): Type of Keras cell to use for the recurrent layers. Either provide
        a Keras layer object, or one of ['lstm', 'gru']
    recurrent_layer_sizes (tuple, optional): Number of units in each recurrent layer in network
    dense_layer_sizes (tuple, optional): Number of units in each dense layer in network (which come after recurrent layers)
    dense_activation (str, optional): Keras activation function to use for each dense layer
    dropout_fraction (float, optional): Fraction of outputs to drop out of each (non-recurrent portion of each) layer
    recurrent_dropout_fraction (float, optional): Fraction of outputs to drop out of each recurrent iteration
    categorical_max_vocab (int, optional): If provided, will take the top categorical_max_vocab - 1 categories from
        each categorical variable, and will set the rest to a single "unknown" category.
    categorical_embedding_size (int, optional): If categorical features provided, will embed them each into
        a dense vector of this size
    loss (str, optional): loss function to use for gradient calculation. If labels is a Boolean Series, defaults
        to `binary_crossentropy`. If labels is an object (multiclass), defaults to `categorical_crossentropy`.
        If labels is numeric, defaults to 'mse'.
    metrics (list[str], optional): List of metrics for Keras to compute internally on validation set.
        If labels is a Boolean Series, defaults
        to ['accuracy', 'f1', 'roc_auc']. If labels is an object (multiclass), defaults to ['accuracy', 'f1_macro'].
        If labels is numeric, defaults to ['mse', 'r2'].
    optimizer (str, optional): Optimizer to use for gradient descent


    '''
    fm, categorical_vocab = map_categorical_fm_to_int(fm, fl, categorical_max_vocab)
    sequence_input, sequence_output = sequences_from_fm(fm, labels)
    categorical_features = [f for f in fl
                            if issubclass(f.variable_type, Discrete)]

    max_values_per_instance = sequence_input.shape[1]
    inputs = {}
    cat_embedding_layers = []
    for i, f in enumerate(categorical_features):
        feature_max_vocab = len(categorical_vocab[f.get_name()])
        if categorical_max_vocab is not None:
            feature_max_vocab = min(feature_max_vocab, categorical_max_vocab)
        cat_input = Input(shape=(max_values_per_instance,),
                          dtype='int32',
                          name=f.get_name())
        inputs[f.get_name()] = cat_input
        embedding = Embedding(output_dim=categorical_embedding_size,
                              input_dim=feature_max_vocab,
                              input_length=max_values_per_instance)(cat_input)
        cat_embedding_layers.append(embedding)

    num_numeric_features = fm.shape[1] - len(cat_embedding_layers)
    numeric_input = None
    if num_numeric_features > 0:
        numeric_input_name = 'numeric_input'
        numeric_input = Input(shape=(max_values_per_instance, num_numeric_features),
                              dtype='int32',
                              name=numeric_input_name)
        inputs[numeric_input_name] = numeric_input
    rnn_inputs = []
    rnn_input_size = 0
    if len(cat_embedding_layers):
        rnn_inputs.extend(cat_embedding_layers)
        rnn_input_size += (categorical_embedding_size * len(cat_embedding_layers))
    if numeric_input is not None:
        rnn_inputs.append(numeric_input)
        rnn_input_size += num_numeric_features
    rnn_inputs = keras.layer.concatenate(rnn_inputs)

    if isinstance(cell_type, str):
        cell_type = RNN_CELLS[cell_type]
    else:
        RNNCell = cell_type
    prev_layer = rnn_inputs
    for i, layer_size in enumerate(recurrent_layer_sizes):
        return_sequences = True
        _rnn_input_shape = (max_values_per_instance, layer_size)
        if i == 0:
            _rnn_input_shape = (max_values_per_instance, rnn_input_size)
        if i == len(recurrent_layer_sizes) - 1:
            return_sequences = False
        layer = RNNCell(layer_size,
                        return_sequences=return_sequences,
                        dropout=dropout_fraction,
                        recurrent_dropout=recurrent_dropout_fraction,
                        input_shape=_rnn_input_shape)
        layer = layer(prev_layer)
        prev_layer = layer
    for layer_size in dense_layer_sizes:
        layer = Dense(layer_size,
                      activation=dense_activation,
                      dropout=dropout_fraction)(prev_layer)
        prev_layer = layer

    is_binary = labels.dtype == np.bool_
    is_numeric = labels.dtype != object
    if is_binary:
        output_size = 2
        loss = loss or 'binary_crossentropy'
    elif is_numeric:
        output_size = 1
        loss = loss or 'mse'
    else:
        output_size = labels.nunique()
        loss = loss or 'categorical_crossentropy'
    output_layer = Dense(output_size, activation='sigmoid',
                         name='target')(prev_layer)
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def sequences_from_fm(fm, labels):
    instance_id_name = fm.index.names[0]
    fm.reset_index(instance_id_name, drop=False, inplace=True)
    fm.reset_index(drop=False, inplace=True)
    sequences = [group for _, group in fm.groupby(instance_id_name)]
    sequence_input = pad_sequences(sequences, padding='pre')
    # TODO: resample?
    # TODO: cap length of each time series?

    # TODO: labels -> sequence_output
    return sequence_input, labels

def map_categorical_fm_to_int(fm, fl, categorical_max_vocab, cat_mapping=None):
    if cat_mapping is None:
        cat_mapping = {}
    new_fm = fm.copy()
    for f in fl:
        if issubclass(f.variable_type, Discrete):
            numeric_series, new_mapping = map_categorical_series_to_int(
                fm[f.get_name()],
                categorical_max_vocab,
                mapping=cat_mapping.get(f.get_name(), None))
            new_fm[f.get_name()] = numeric_series
            cat_mapping[f.get_name()] = new_mapping
    return new_fm, cat_mapping

def map_categorical_series_to_int(input_series, categorical_max_vocab=None, mapping=None):
    input_series_name = input_series.name
    val_counts = input_series.value_counts().to_frame()
    index_name = val_counts.index.name
    if index_name is None:
        if 'index' in val_counts.columns:
            index_name = 'level_0'
        else:
            index_name = 'index'
    val_counts.reset_index(inplace=True)
    val_counts = val_counts.sort_values([input_series_name, index_name],
                                        ascending=False)
    val_counts.set_index(index_name, inplace=True)
    full_mapping = val_counts.index.tolist()
    unique = val_counts.head(categorical_max_vocab).index.tolist()
    unknown = [v for v in full_mapping if v not in unique]

    if mapping is None:
        mapping = {v: k + 1 for k, v in enumerate(unique)}
        mapping.update({v: 0 for v in unknown})
    numeric = input_series.replace(mapping)
    return numeric, mapping


