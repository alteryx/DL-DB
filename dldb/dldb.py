from keras.layers import Dense, LSTM, GRU, Embedding, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from featuretools.variable_types import Discrete, Boolean
import keras
import numpy as np
import pandas as pd
import re
import uuid

RNN_CELLS = {
    'lstm': LSTM,
    'gru': GRU,
}


def feature_name_to_valid_keras_name(fname):
    return re.sub(r'[(.]', '_', fname).replace(')', '')


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
        self.categorical_feature_names = None
        self.categorical_vocab = None
        self.max_values_per_instance = None
        self.name_mapping = None

    def compile(self, fm, fl=None, categorical_feature_names=None):
        '''
        fm (pd.DataFrame): Time-varying feature matrix with multiple time points per instance. Can contain both categorical
            as well as numeric features. Index should either be just the instance_id, or a MultiIndex with the instance_id and time,
            with the instance_id as the first level.
        fl (list[ft.PrimitiveBase], optional): List of feature objects representing each column in fm. Will be used if
            categorical_feature_names not provided.
        categorical_feature_names (list[str], optional): List of feature names that are categorical

        Note: If neither categorical_feature_names nor fl provided, will assume all features of dtype object
        are categorical

        Assumes all provided features can be treated as either categorical or numeric (including Booleans).
        If a data type exists in fm other than a numeric or object type (such as a datetime), then make sure to
        include that feature in categorical_feature_names to treat it as a categorical.
        '''

        if categorical_feature_names is not None:
            self.categorical_feature_names = categorical_feature_names
        elif fl is not None:
            self.categorical_feature_names = [f.get_name() for f in fl
                                              if issubclass(f.variable_type,
                                                            Discrete)
                                              and not f.variable_type == Boolean]
        else:
            self.categorical_feature_names = [c for c in fm.columns
                                              if fm[c].dtype == object]

        self.categorical_vocab = self._gen_categorical_mapping(fm)
        fm = self._map_categorical_fm_to_int(fm)

        self.name_mapping = {c: feature_name_to_valid_keras_name(c)
                             for c in fm.columns}

        self.numeric_columns = [f for f in fm.columns
                                if f not in self.categorical_feature_names]

        self._fit_scaler_imputer(fm)

        instance_id_name = fm.index.names[0]
        self.max_values_per_instance = (
            fm.reset_index(instance_id_name, drop=False)
              .groupby(instance_id_name)[instance_id_name]
              .count()
              .max())

        if not self.regression:
            self.lb = LabelBinarizer().fit(self.classes)
        self._compile_keras_model()

    def fit(self, fm, labels,
            epochs=1, batch_size=32,
            reset_model=True,
            **kwargs):
        if reset_model:
            self._fit_scaler_imputer(fm)
            self._compile_keras_model()

        if kwargs.get('verbose', 1) > 0:
            print("Transforming input matrix into numeric sequences")
        inputs, outputs = self._input_transform(fm, labels)
        if kwargs.get('verbose', 1) > 0:
            print("Fitting Keras model")
        self.model.fit(
            inputs,
            outputs,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs)

    def predict(self, fm, verbose=1):
        if verbose > 0:
            print("Transforming input matrix into numeric sequences")
        inputs = self._input_transform(fm)
        if verbose > 0:
            print("Predicting using Keras model")
        predictions = self.model.predict(inputs)
        if verbose > 0:
            print("Transforming outputs")
        return self._output_transform(predictions)

    def _fit_scaler_imputer(self, fm):
        self.fill_vals = {}
        if len(self.numeric_columns) > 0:
            numeric_fm = fm[self.numeric_columns]

            numeric_fm = numeric_fm.astype(np.float32)
            for f in self.numeric_columns:
                if fm[f].dropna().shape[0] == 0:
                    fill_val = 0
                else:
                    fill_val = numeric_fm[f].dropna().mean()
                self.fill_vals[f] = fill_val
                numeric_fm[f] = numeric_fm[f].map({np.inf: np.nan})
            numeric_fm.fillna(value=self.fill_vals, inplace=True)
            self.scaler = MinMaxScaler()
            self.scaler.fit(numeric_fm)

        for f in self.categorical_feature_names:
            if fm[f].dropna().shape[0] == 0:
                fill_val = 0
            else:
                fill_val = fm[f].dropna().mode().iloc[0]
            self.fill_vals[f] = fill_val

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

    def _input_transform(self, fm, labels=None):
        fm = self._map_categorical_fm_to_int(fm)
        inputs = {}
        for i, f in enumerate(self.categorical_feature_names):
            keras_name = self.name_mapping[f]
            feature_max_vocab = len(self.categorical_vocab[f])
            if self.categorical_max_vocab is not None:
                feature_max_vocab = min(feature_max_vocab,
                                        self.categorical_max_vocab)
            vals = fm[[f]]
            if vals.dropna().shape[0] != vals.shape[0]:
                vals = vals.fillna(self.fill_vals[f])
            inputs[keras_name] = vals
        inputs = {k: self._sequences_from_fm(i)[:, :, 0]
                  for k, i in inputs.items()}

        if self.numeric_columns:
            numeric_fm = fm[self.numeric_columns]
            for f in self.numeric_columns:
                vals = numeric_fm[f]
                if vals.dropna().shape[0] != vals.shape[0]:
                    numeric_fm[f] = vals.fillna(self.fill_vals[f])
            numeric_fm = self.scaler.transform(numeric_fm)
            numeric_fm = pd.DataFrame(numeric_fm, index=fm.index,
                                      columns=self.numeric_columns)

            numeric_inputs = self._sequences_from_fm(numeric_fm)
            inputs[self.numeric_input_name] = numeric_inputs

        if labels is not None:
            if not self.regression:
                labels = pd.Series(labels).astype(int)
                if len(self.classes) > 2:
                    labels = self.lb.transform(labels)
            return inputs, labels
        else:
            return inputs

    def _output_transform(self, predictions):
        if not self.regression and len(self.classes) > 2:
            predictions = np.array([self.lb.classes_[i]
                                    for i in predictions.argmax(axis=1)])
        return predictions

    def _sequences_from_fm(self, fm):
        instance_id_name = fm.index.names[0]
        fm.reset_index(instance_id_name, drop=False, inplace=True)
        fm.reset_index(drop=True, inplace=True)

        sequences = [group.drop([instance_id_name], axis=1)
                     for _, group in fm.groupby(instance_id_name)]
        sequence_input = pad_sequences(sequences,
                                       maxlen=self.max_values_per_instance,
                                       padding='pre')
        return sequence_input

    def _map_categorical_fm_to_int(self, fm):
        new_fm = fm.copy()
        for f in self.categorical_feature_names:
            numeric_series, new_mapping = self._map_categorical_series_to_int(
                fm[f],
                self.categorical_vocab.get(f, None))
            new_fm[f] = numeric_series
            self.categorical_vocab[f] = new_mapping
        return new_fm

    def _gen_categorical_mapping(self, fm):
        categorical_vocab = {}
        if self.categorical_max_vocab is None:
            self.categorical_max_vocab = max(fm[f].dropna().nunique()
                                             for f in fm)
        for f in self.categorical_feature_names:
            nan_val = str(uuid.uuid4())
            val_counts = (fm[f].astype(str)
                               .fillna(nan_val)
                               .value_counts()
                               .to_frame())
            index_name = val_counts.index.name
            if index_name is None:
                if 'index' in val_counts.columns:
                    index_name = 'level_0'
                else:
                    index_name = 'index'
            val_counts.reset_index(inplace=True)
            val_counts = val_counts.sort_values([f, index_name],
                                                ascending=False)
            val_counts.set_index(index_name, inplace=True)
            full_mapping = val_counts.index.tolist()
            unique = set(val_counts.head(self.categorical_max_vocab - 1).index.tolist())
            unknown = [v for v in full_mapping if v not in unique]

            mapping = {v: k + 1 for k, v in enumerate(unique)}
            mapping.update({v: 0 for v in unknown})
            mapping[nan_val] = -1
            categorical_vocab[f] = mapping
        return categorical_vocab

    def _map_categorical_series_to_int(self, input_series,
                                       mapping):
        nan_val = [k for k, v in mapping.items()
                   if v == -1][0]

        input_series = input_series.astype(str).fillna(nan_val)

        unique_vals = input_series.unique()
        new_mapping = {u: 0 for u in unique_vals}
        new_mapping.update(mapping)
        numeric = input_series.map(new_mapping)
        return numeric, new_mapping
