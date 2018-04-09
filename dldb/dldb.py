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
import os
import pandas as pd

RNN_CELLS = {
    'lstm': LSTM,
    'gru': GRU,
}


def feature_name_to_valid_keras_name(fname):
    return re.sub(r'[(.]', '_', fname).replace(')', '')


class DLDBInputGenerator(Sequence):
    def __init__(self, filenames, instance_id_name, batch_size, chunk_size):
        self.instance_id_name = instance_id_name
        self.filenames = filenames
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self._cached_chunk_idx = -1
        self._cached_ftens = None

    def __len__(self):
        return len(self.filenames) * self.chunk_size

    def __getitem__(self, idx):
        chunk = idx // self.chunk_size
        if chunk == self._cached_chunk_idx:
            ftens = self._cached_ftens
        else:
            self._cached_chunk_idx = chunk
            ftens = pd.read_csv(self.filenames[chunk])
            self._cached_ftens = ftens
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        instances = ftens[self.instance_id_name].drop_duplicates()
        instances = instances.iloc[start_idx: end_idx]
        ftens = ftens.merge(instances.to_frame(),
                            on=self.instance_id_name, how='left')
        labels = (ftens[[self.instance_id_name, 'label']]
                  .drop_duplicates([self.instance_id_name])
                  .set_index(self.instance_id_name))['label']
        ftens.set_index(self.instance_id_name, inplace=True)
        return ftens, labels


class InputGeneratorTransformer(Sequence):
    def __init__(self, input_generator, input_transform):
        self.input_generator = input_generator
        self.input_transform = input_transform

    def __len__(self):
        return len(self.input_generator)

    def __getitem__(self, idx):
        ftens, labels = self.input_generator[idx]
        return self.input_transform(ftens, labels)


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

    def compile(self, ftens, fl=None, categorical_feature_names=None):
        '''
        ftens (pd.DataFrame): 3D (flattened to 2D DataFrame) feature tensor with multiple time points or observations per instance. Can contain both categorical
            as well as numeric features. Index should either be just the instance_id, or a MultiIndex with the instance_id and time/step,
            with the instance_id as the first level.
        fl (list[ft.PrimitiveBase], optional): List of feature objects representing each column in ftens. Will be used if
            categorical_feature_names not provided.
        categorical_feature_names (list[str], optional): List of feature names that are categorical

        Note: If neither categorical_feature_names nor fl provided, will assume all features of dtype object
        are categorical

        Assumes all provided features can be treated as either categorical or numeric (including Booleans).
        If a data type exists in ftens other than a numeric or object type (such as a datetime), then make sure to
        include that feature in categorical_feature_names to treat it as a categorical.
        '''

        ftens = self.ml_preprocessor.fit_transform(
            ftens, fl=fl,
            categorical_feature_names=categorical_feature_names)
        self.name_mapping = {c: feature_name_to_valid_keras_name(c)
                             for c in ftens.columns}

        instance_id_name = ftens.index.names[0]
        self.max_values_per_instance = (
            ftens.reset_index(instance_id_name, drop=False)
              .groupby(instance_id_name)[instance_id_name]
              .count()
              .max())

        self._compile_keras_model()

    def compile_generator(self, ftens, labels,
                          fl=None, categorical_feature_names=None,
                          batch_size=128,
                          save_chunk_multiplier=10,
                          on_disk_save_location="dldb_temp_{}"):
        '''
        ftens (pd.DataFrame): 3D (flattened to 2D DataFrame) feature tensor with multiple time points or observations per instance. Can contain both categorical
            as well as numeric features. Index should either be just the instance_id, or a MultiIndex with the instance_id and time/step,
            with the instance_id as the first level.
        fl (list[ft.PrimitiveBase], optional): List of feature objects representing each column in ftens. Will be used if
            categorical_feature_names not provided.
        categorical_feature_names (list[str], optional): List of feature names that are categorical

        Note: If neither categorical_feature_names nor fl provided, will assume all features of dtype object
        are categorical

        Assumes all provided features can be treated as either categorical or numeric (including Booleans).
        If a data type exists in ftens other than a numeric or object type (such as a datetime), then make sure to
        include that feature in categorical_feature_names to treat it as a categorical.
        '''

        ftens = self.ml_preprocessor.fit_transform(
            ftens, fl=fl,
            categorical_feature_names=categorical_feature_names)
        self.name_mapping = {c: feature_name_to_valid_keras_name(c)
                             for c in ftens.columns}
        self._compile_keras_model_batched()
        instance_id_name = ftens.index.names[0]
        on_disk_save_location = on_disk_save_location.format(pd.Timestamp.now())
        if not os.path.exists(on_disk_save_location):
            os.makedirs(on_disk_save_location)

        ftens.reset_index(instance_id_name, drop=False, inplace=True)
        ftens = ftens.merge(labels.to_frame('label'), left_on=[instance_id_name], right_index=True)
        chunk_col = uuid.uuid4()
        chunk_size = batch_size * save_chunk_multiplier
        ftens[chunk_col] = ftens[instance_id_name].astype('category').cat.codes // chunk_size
        filenames = []
        for name, group in ftens.groupby(chunk_col):
            filename = os.path.join(on_disk_save_location, str(name)) + ".csv"
            filenames.append(filename)
            group.drop([chunk_col], axis=1).to_csv(filename)
        return DLDBInputGenerator(filenames, instance_id_name,
                                  batch_size, chunk_size)

    def fit(self, ftens, labels,
            epochs=1, batch_size=32,
            reset_model=True,
            **kwargs):
        if reset_model:
            self.ml_preprocessor.fit_scaler_imputer(ftens)
            self._compile_keras_model()

        if kwargs.get('verbose', 1) > 0:
            print("Transforming input tensor into numeric sequences")
        inputs, outputs = self._input_transform(ftens, labels)
        if kwargs.get('verbose', 1) > 0:
            print("Fitting Keras model")
        return self.model.fit(
            inputs,
            outputs,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs)

    def fit_generator(self,
                      generator,
                      **kwargs):
        if isinstance(generator, Sequence):
            generator = InputGeneratorTransformer(generator, self._input_transform)
        else:
            def decorate_generator(g):
                return self._input_transform(*next(g))
            generator = decorate_generator(generator)
        return self.model.fit_generator(generator, **kwargs)

    def predict(self, ftens, verbose=1):
        if verbose > 0:
            print("Transforming input tensor into numeric sequences")
        inputs = self._input_transform(ftens)
        if verbose > 0:
            print("Predicting using Keras model")
        predictions = self.model.predict(inputs)
        if verbose > 0:
            print("Transforming outputs")
        return self._output_transform(predictions)

    def _compile_keras_model_batched(self):
        inputs = []
        cat_embedding_layers = []
        for i, f in enumerate(self.categorical_feature_names):
            feature_max_vocab = len(self.categorical_vocab[f])
            if self.categorical_max_vocab is not None:
                feature_max_vocab = min(feature_max_vocab,
                                        self.categorical_max_vocab)
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
            layer = self.RNNCell(layer_size,
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

    def _compile_keras_model(self):
        inputs = []
        cat_embedding_layers = []
        for i, f in enumerate(self.categorical_feature_names):
            feature_max_vocab = len(self.categorical_vocab[f])
            if self.categorical_max_vocab is not None:
                feature_max_vocab = min(feature_max_vocab,
                                        self.categorical_max_vocab)
            cat_input = Input(shape=(self.max_values_per_instance,),
                              dtype='int32',
                              name=self.name_mapping[f])
            inputs.append(cat_input)
            embedding = Embedding(output_dim=self.categorical_embedding_size,
                                  input_dim=feature_max_vocab,
                                  input_length=self.max_values_per_instance)
            embedding = embedding(cat_input)
            cat_embedding_layers.append(embedding)

        numeric_input = None
        if len(self.numeric_columns) > 0:
            numeric_input = Input(shape=(self.max_values_per_instance,
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
            _rnn_input_shape = (self.max_values_per_instance, layer_size)
            if i == 0:
                _rnn_input_shape = (self.max_values_per_instance,
                                    rnn_input_size)
            if i == len(self.recurrent_layer_sizes) - 1:
                return_sequences = False
            layer = self.RNNCell(layer_size,
                                 return_sequences=return_sequences,
                                 dropout=self.dropout_fraction,
                                 recurrent_dropout=self.recurrent_dropout_fraction,
                                 input_shape=_rnn_input_shape)
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

    def _input_transform(self, ftens, labels=None):
        # Assume index in (instance_id, time) and index already sorted
        if labels is not None:
            ftens, labels = self.ml_preprocessor.transform(ftens, labels=labels)
        else:
            ftens = self.ml_preprocessor.transform(ftens)

        inputs = {self.name_mapping[f]: self._sequences_from_ftens(ftens[[f]])[:, :, 0]
                  for f in self.categorical_feature_names}

        if self.numeric_columns:
            inputs[self.numeric_input_name] = self._sequences_from_ftens(ftens[self.numeric_columns])

        if labels is not None:
            return inputs, labels
        else:
            return inputs

    def _output_transform(self, predictions):
        if not self.regression and len(self.classes) > 2:
            predictions = np.array([self.lb.classes_[i]
                                    for i in predictions.argmax(axis=1)])
        return predictions

    def _sequences_from_ftens(self, ftens):
        cols = list(ftens.columns)
        instance_id_name = ftens.index.names[0]
        ftens.reset_index(inplace=True, drop=False)
        fm = ftens[cols + [instance_id_name]]
        sequences = [np.array(list(group))[:, :-1]
                     for _, group in groupby(fm.values, lambda row: row[-1])]
        sequence_input = pad_sequences(sequences,
                                       maxlen=self.max_values_per_instance,
                                       padding='pre')
        return sequence_input
