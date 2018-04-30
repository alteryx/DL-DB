from keras.layers import Dense, LSTM, GRU, Embedding, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from .preprocessor import MLPreprocessor
from itertools import groupby
import keras
import numpy as np
import re
import uuid
from math import ceil


RNN_CELLS = {
    'lstm': LSTM,
    'gru': GRU,
}


def feature_name_to_valid_keras_name(fname):
    return re.sub(r'[(.]', '_', fname).replace(')', '')


class DLDBInputGenerator(Sequence):
    def __init__(self, ftens,
                 categorical_feature_names,
                 numeric_input_name,
                 name_mapping,
                 numeric_columns,
                 batch_size=32,
                 labels=None):
        self.ftens = ftens
        self.labels = labels
        self.instance_id_name = self.ftens.index.names[0]
        self.ftens.reset_index(self.instance_id_name, drop=False, inplace=True)
        self.batch_size = batch_size
        if self.batch_size:
            self.batch_col = uuid.uuid4()

            self.ftens[self.batch_col] = self.ftens[self.instance_id_name].astype(
                'category').cat.codes // self.batch_size
            self.ftens.set_index(self.batch_col, inplace=True)

        # TODO: figure out what to do about these
        self.name_mapping = name_mapping
        self.categorical_feature_names = categorical_feature_names
        self.numeric_input_name = numeric_input_name
        self.numeric_columns = numeric_columns

        self._length = 1
        if self.labels is not None:
            self.labels = self.labels.to_frame().reset_index(
                self.instance_id_name,
                drop=False)
            if self.batch_size:
                self.labels[self.batch_col] = self.labels[self.instance_id_name].astype(
                    'category').cat.codes // self.batch_size
                self.labels.set_index(self.batch_col, inplace=True)

                self._length = int(ceil(self.labels.shape[0] / self.batch_size))
            else:
                self._length = int(ceil(self.labels.shape[0] / self.batch_size))
        elif self.batch_size:
            self._length = int(ceil(self.ftens[self.instance_id_name].nunique() / self.batch_size))

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        labels = None
        if self.batch_size:
            ftens = self.ftens.loc[idx].set_index(self.instance_id_name)
            if self.labels is not None:
                labels = self.labels.loc[idx].set_index(self.instance_id_name)
        else:
            ftens = self.ftens.set_index(self.instance_id_name)
            if self.labels is not None:
                labels = self.labels.set_index(self.instance_id_name)

        inputs = {self.name_mapping[f]: self._sequences_from_ftens(
                ftens[[f]])[:, :, 0]
              for f in self.categorical_feature_names}
        if self.numeric_columns:
            inputs[self.numeric_input_name] = self._sequences_from_ftens(
                ftens[self.numeric_columns])
        if labels is None:
            return inputs
        else:
            return inputs, labels

    def _sequences_from_ftens(self, ftens):
        cols = list(ftens.columns)
        instance_id_name = ftens.index.names[0]
        ftens.reset_index(inplace=True, drop=False)
        ftens = ftens[cols + [instance_id_name]]
        # TODO: revert back to pandas here? since its batched
        sequences = [np.array(list(group))[:, :-1]
                     for _, group in groupby(ftens.values, lambda row: row[-1])]
        sequence_input = pad_sequences(sequences,
                                       padding='pre')
        return sequence_input


class DLDB(object):
    numeric_input_name = 'numeric_input'

    def __init__(self,
                 regression=False,
                 classes=None,
                 cell_type='lstm',
                 recurrent_layer_sizes=(64, 64),
                 dense_layer_sizes=(10,),
                 dense_activation='relu',
                 dropout_fraction=0.2,
                 recurrent_dropout_fraction=0.2,
                 categorical_max_vocab=None,
                 categorical_embedding_size=10,
                 conv_kernel_dim=None,
                 conv_activation='relu',
                 pool_size=4,
                 conv_batch_normalization=False,
                 loss=None,
                 metrics=None,
                 optimizer='rmsprop'):
        '''
        regression (bool): If True, labels represent continuous values to predict (otherwise represent class labels)
        classes (list[object] or np.ndarray[object] or pd.Series[object]): If regression is False, classes contains all possible class labels
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
        conv_kernel_dim (int, optional): If provided, will add a 1D Convolutional layer prior to the recurrent layers
        conv_activation (str, optional): Activation to use for the optional convolutional layer
        pool_size (int, optional): Size of max pooling layer that will be used after the convolutional layer if it is present
        conv_batch_normalization (bool, optional): If true, will apply batch normalization to the outputs of the convolutional layer
        loss (str, optional): loss function to use for gradient calculation. If labels is a Boolean Series, defaults
            to `binary_crossentropy`. If labels is an object (multiclass), defaults to `categorical_crossentropy`.
            If labels is numeric, defaults to 'mse'.
        metrics (list[str], optional): List of metrics for Keras to compute internally on validation set.
            If labels is a Boolean Series, defaults
            to ['accuracy', 'f1', 'roc_auc']. If labels is an object (multiclass), defaults to ['accuracy', 'f1_macro'].
            If labels is numeric, defaults to ['mse', 'r2'].
        optimizer (str, optional): Optimizer to use for gradient descent

        '''
        self.regression = regression
        self.classes = classes

        if self.regression:
            self.output_size = 1
            self.loss = loss or 'mse'
        elif len(self.classes) == 2:
            self.output_size = 1
            self.loss = loss or 'binary_crossentropy'
        else:
            self.output_size = len(self.classes)
            self.loss = loss or 'categorical_crossentropy'

        self.cell_type = cell_type
        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activation = dense_activation
        self.dropout_fraction = dropout_fraction
        self.recurrent_dropout_fraction = recurrent_dropout_fraction
        self.categorical_max_vocab = categorical_max_vocab
        self.categorical_embedding_size = categorical_embedding_size
        self.conv_kernel_dim = conv_kernel_dim
        self.conv_activation = conv_activation
        self.pool_size = pool_size
        self.conv_batch_normalization = conv_batch_normalization
        self.metrics = metrics
        self.optimizer = optimizer
        self.max_values_per_instance = None
        self.name_mapping = None
        self.ml_preprocessor = MLPreprocessor(
            categorical_max_vocab=self.categorical_max_vocab,
            classes=self.classes,
            regression=self.regression)

    @property
    def categorical_vocab(self):
        return self.ml_preprocessor.categorical_vocab

    @property
    def numeric_columns(self):
        return self.ml_preprocessor.numeric_columns

    @property
    def categorical_feature_names(self):
        return self.ml_preprocessor.categorical_feature_names

    def _preprocess(self, ftens, labels=None,
                    fl=None, categorical_feature_names=None,
                    batch_size=32,
                    fit=True):
        if fit:
            ftens = self.ml_preprocessor.fit_transform(
                ftens, fl=fl,
                categorical_feature_names=categorical_feature_names)
            self.name_mapping = {c: feature_name_to_valid_keras_name(c)
                                 for c in ftens.columns}
        else:
            ftens = self.ml_preprocessor.transform(ftens)
        return DLDBInputGenerator(ftens,
                                  self.categorical_feature_names,
                                  self.numeric_input_name,
                                  self.name_mapping,
                                  self.numeric_columns,
                                  batch_size=batch_size,
                                  labels=labels)

    def partial_fit(self,
                    ftens=None,
                    labels=None,
                    generator=None,
                    batch_size=32,
                    **kwargs):
        if generator is None:
            generator = self._preprocess(ftens,
                                         labels,
                                         batch_size=batch_size,
                                         fit=False)

        return (self.model.fit_generator(generator,
                                         **kwargs),
                generator)

    def fit(self,
            ftens,
            labels,
            fl=None, categorical_feature_names=None,
            batch_size=32,
            **kwargs):
        generator = self._preprocess(
            ftens,
            labels,
            fl=fl,
            categorical_feature_names=categorical_feature_names,
            batch_size=batch_size,
            fit=True)
        self._compile_keras_model()
        return (self.model.fit_generator(generator,
                                         **kwargs),
                generator)

    def predict(self, ftens, verbose=1, **kwargs):
        if verbose > 0:
            print("Transforming input tensor into numeric sequences")
        generator = self._preprocess(ftens, batch_size=None, fit=False)
        if verbose > 0:
            print("Predicting using Keras model")
        predictions = self.model.predict_generator(generator, **kwargs)
        if verbose > 0:
            print("Transforming outputs")
        if not self.regression and len(self.classes) > 2:
            predictions = np.array([self.lb.classes_[i]
                                    for i in predictions.argmax(axis=1)])
        return predictions

    def _compile_keras_model(self):
        inputs = []
        cat_embedding_layers = []
        for i, f in enumerate(self.categorical_feature_names):
            feature_max_vocab = len(self.categorical_vocab[f]) + 1
            if self.categorical_max_vocab is not None:
                feature_max_vocab = min(feature_max_vocab,
                                        self.categorical_max_vocab + 1)
            cat_input = Input(shape=(None,),
                              dtype='int32',
                              name=self.name_mapping[f])
            inputs.append(cat_input)
            embedding = Embedding(output_dim=self.categorical_embedding_size,
                                  input_dim=feature_max_vocab,
                                  mask_zero=True)
            embedding = embedding(cat_input)
            cat_embedding_layers.append(embedding)

        numeric_input = None
        if len(self.numeric_columns) > 0:
            numeric_input = Input(shape=(None,
                                         len(self.numeric_columns)),
                                  dtype='float32',
                                  name=self.numeric_input_name)
            inputs.append(numeric_input)

        rnn_inputs = []
        rnn_input_size = 0
        if len(cat_embedding_layers):
            rnn_inputs.extend(cat_embedding_layers)
            rnn_input_size += (self.categorical_embedding_size *
                               len(cat_embedding_layers))
        if numeric_input is not None:
            rnn_inputs.append(numeric_input)
            rnn_input_size += len(self.numeric_columns)
        if len(rnn_inputs) > 1:
            rnn_inputs = keras.layers.concatenate(rnn_inputs)
        else:
            rnn_inputs = rnn_inputs[0]

        if self.conv_kernel_dim is not None:

            conv_layer = Conv1D(self.categorical_embedding_size//2,
                                self.conv_kernel_dim,
                                activation=self.conv_activation)
            if self.conv_batch_normalization:
                rnn_inputs = BatchNormalization()(rnn_inputs)
            conv_layer = conv_layer(rnn_inputs)
            mp_layer = MaxPooling1D(pool_size=self.pool_size)
            rnn_inputs = mp_layer(conv_layer)

        if isinstance(self.cell_type, str):
            self.RNNCell = RNN_CELLS[self.cell_type]
        else:
            self.RNNCell = self.cell_type
        prev_layer = rnn_inputs
        for i, layer_size in enumerate(self.recurrent_layer_sizes):
            return_sequences = True
            if i == len(self.recurrent_layer_sizes) - 1:
                return_sequences = False
            layer = self.RNNCell(
                layer_size,
                return_sequences=return_sequences,
                dropout=self.dropout_fraction,
                recurrent_dropout=self.recurrent_dropout_fraction)
            layer = layer(prev_layer)
            prev_layer = layer
        for layer_size in self.dense_layer_sizes:
            layer = Dense(layer_size,
                          activation=self.dense_activation)(prev_layer)
            dropout_layer = Dropout(self.dropout_fraction)(layer)
            prev_layer = dropout_layer

        output_layer = Dense(self.output_size, activation='sigmoid',
                             name='target')(prev_layer)
        self.model = Model(inputs=inputs, outputs=output_layer)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
