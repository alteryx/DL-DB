from featuretools.variable_types import Discrete, Boolean
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import pandas as pd


class MLPreprocessor(object):
    def __init__(self,
                 categorical_max_vocab=None,
                 classes=None,
                 regression=False):
        self.categorical_max_vocab = categorical_max_vocab
        self.classes = classes
        self.regression = regression
        self.categorical_vocab = None

    def fit_transform(self, ftens, fl=None, categorical_feature_names=None, labels=None):
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

        # Can't handle multiindex
        if len(ftens.index.names) > 1:
            index_name = ftens.index.names[0]
            ftens = ftens.reset_index(index_name, drop=False).set_index(index_name)
        self.categorical_vocab = self._gen_categorical_mapping(ftens)

        self.numeric_columns = [f for f in ftens.columns
                                if f not in self.categorical_feature_names]

        ftens = self.fit_transform_scaler_imputer(ftens)

        if not self.regression:
            self.lb = LabelBinarizer().fit(self.classes)

        if labels is not None:
            return ftens, self.transform_labels(labels)
        else:
            return ftens

    def fit_transform_scaler_imputer(self, ftens):
        self.fill_vals = {}
        new_ftens = ftens
        if len(self.numeric_columns) > 0:
            numeric_ftens = ftens[self.numeric_columns]

            numeric_ftens = numeric_ftens.astype(np.float32)
            for f in self.numeric_columns:
                if ftens[f].dropna().shape[0] == 0:
                    fill_val = 0
                else:
                    fill_val = numeric_ftens[f].dropna().mean()
                self.fill_vals[f] = fill_val
                numeric_ftens.loc[~np.isfinite(numeric_ftens[f]), f] = np.nan
            numeric_ftens.fillna(value=self.fill_vals, inplace=True)
            self.scaler = MinMaxScaler()
            numeric_ftens = self.scaler.fit_transform(numeric_ftens)
            new_ftens[self.numeric_columns] = numeric_ftens

        return self._map_categorical_ftens_to_int(new_ftens)

    def transform(self, ftens, labels=None):
        ftens = self._map_categorical_ftens_to_int(ftens)
        if len(self.numeric_columns) > 0:
            numeric_ftens = ftens[self.numeric_columns]
            numeric_ftens = numeric_ftens.astype(np.float32)
            for f in self.numeric_columns:
                vals = numeric_ftens[f]
                numeric_ftens.loc[~np.isfinite(numeric_ftens[f]), f] = np.nan
                if vals.dropna().shape[0] != vals.shape[0]:
                    numeric_ftens[f].fillna(self.fill_vals[f], inplace=True)
            numeric_ftens = self.scaler.transform(numeric_ftens)
            ftens[self.numeric_columns] = numeric_ftens
        if labels is not None:
            return ftens, self.transform_labels(labels)
        else:
            return ftens

    def transform_labels(self, labels):
        if not self.regression:
            labels = pd.Series(labels).astype(int)
            if len(self.classes) > 2:
                labels = self.lb.transform(labels)
        return labels

    def _map_categorical_ftens_to_int(self, ftens):
        new_ftens = ftens
        for f in self.categorical_feature_names:
            numeric_series, new_mapping = self._map_categorical_series_to_int(
                ftens[f],
                self.categorical_vocab.get(f, None))
            new_ftens[f] = numeric_series
            self.categorical_vocab[f] = new_mapping
        return new_ftens

    def _gen_categorical_mapping(self, ftens):
        categorical_vocab = {}
        for f in self.categorical_feature_names:
            val_counts = ftens[f].dropna().value_counts()
            mapping = {v: k + 1 for k, v in enumerate(val_counts.index)}
            mapping[np.nan] = 0
            if (self.categorical_max_vocab is not None and
                    self.categorical_max_vocab < len(val_counts)):
                num_unique = len(val_counts) - self.categorical_max_vocab
                unknown = val_counts.tail(num_unique).index.tolist()
                mapping.update({u: 0 for u in unknown})
            categorical_vocab[f] = mapping
        return categorical_vocab

    def _map_categorical_series_to_int(self, input_series,
                                       mapping):
        unique_vals = set(input_series.unique())
        # make sure we don't add any new nans
        # since id(np.float64('nan')) != id(np.nan),
        # and so we could end up with multiple nans in the
        # mapping dict
        new_mapping = {u: 0 for u in unique_vals if not pd.isnull(u)}
        new_mapping.update(mapping)
        numeric = input_series.map(new_mapping)
        numeric.fillna(0, inplace=True)
        return numeric, new_mapping
