from featuretools.variable_types import Discrete, Boolean
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import pandas as pd
import uuid


class MLPreprocessor(object):
    def __init__(self,
                 categorical_max_vocab=None,
                 classes=None,
                 regression=False):
        self.categorical_max_vocab = categorical_max_vocab
        self.classes = classes
        self.regression = regression
        self.categorical_vocab = None

    def fit(self, ftens, fl=None, categorical_feature_names=None):
        if categorical_feature_names is not None:
            self.categorical_feature_names = categorical_feature_names
        elif fl is not None:
            self.categorical_feature_names = [f.get_name() for f in fl
                                              if issubclass(f.variable_type,
                                                            Discrete)
                                              and not
                                              f.variable_type == Boolean]
        else:
            self.categorical_feature_names = [c for c in ftens.columns
                                              if ftens[c].dtype == object]

        self.categorical_vocab = self._gen_categorical_mapping(ftens)
        ftens = self._map_categorical_ftens_to_int(ftens)

        self.numeric_columns = [f for f in ftens.columns
                                if f not in self.categorical_feature_names]

        self.fit_scaler_imputer(ftens)

        if not self.regression:
            self.lb = LabelBinarizer().fit(self.classes)

    def fit_scaler_imputer(self, ftens):
        self.fill_vals = {}
        if len(self.numeric_columns) > 0:
            numeric_ftens = ftens[self.numeric_columns]

            numeric_ftens = numeric_ftens.astype(np.float32)
            for f in self.numeric_columns:
                if ftens[f].dropna().shape[0] == 0:
                    fill_val = 0
                else:
                    fill_val = numeric_ftens[f].dropna().mean()
                self.fill_vals[f] = fill_val
                numeric_ftens[f] = numeric_ftens[f].map({np.inf: np.nan})
            numeric_ftens.fillna(value=self.fill_vals, inplace=True)
            self.scaler = MinMaxScaler()
            self.scaler.fit(numeric_ftens)

        for f in self.categorical_feature_names:
            if ftens[f].dropna().shape[0] == 0:
                fill_val = 0
            else:
                fill_val = ftens[f].dropna().mode().iloc[0]
            self.fill_vals[f] = fill_val

    def transform(self, ftens, labels=None):
        ftens = self._map_categorical_ftens_to_int(ftens)
        for i, f in enumerate(self.categorical_feature_names):
            vals = ftens[[f]]
            if vals.dropna().shape[0] != vals.shape[0]:
                ftens[f] = vals.fillna(self.fill_vals[f])

        if self.numeric_columns:
            numeric_ftens = ftens[self.numeric_columns]
            for f in self.numeric_columns:
                vals = numeric_ftens[f]
                if vals.dropna().shape[0] != vals.shape[0]:
                    numeric_ftens[f] = vals.fillna(self.fill_vals[f])
            numeric_ftens = self.scaler.transform(numeric_ftens)
            numeric_ftens = pd.DataFrame(numeric_ftens, index=ftens.index,
                                      columns=self.numeric_columns)

            ftens[self.numeric_columns] = numeric_ftens

        if labels is not None:
            if not self.regression:
                labels = pd.Series(labels).astype(int)
                if len(self.classes) > 2:
                    labels = self.lb.transform(labels)
            return ftens, labels
        else:
            return ftens

    def _map_categorical_ftens_to_int(self, ftens):
        new_ftens = ftens.copy()
        for f in self.categorical_feature_names:
            numeric_series, new_mapping = self._map_categorical_series_to_int(
                ftens[f],
                self.categorical_vocab.get(f, None))
            new_ftens[f] = numeric_series
            self.categorical_vocab[f] = new_mapping
        return new_ftens

    def _gen_categorical_mapping(self, ftens):
        categorical_vocab = {}
        if self.categorical_max_vocab is None:
            self.categorical_max_vocab = max(ftens[f].dropna().nunique()
                                             for f in ftens)
        for f in self.categorical_feature_names:
            nan_val = str(uuid.uuid4())
            val_counts = (ftens[f].astype(str)
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
            unique = set(
                val_counts.head(self.categorical_max_vocab - 1).index.tolist())
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
