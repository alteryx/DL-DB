# DLDB

Deep learning for time-varying multi-entity datasets

# Installation

You should be able to just run:
```
pip install -e .
```

If that fails due to Tensor Flow, please visit [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/) and follow their instructions for installing Tensor Flow on your system.
You can also follow their instructions to install the GPU version to allow DLDB to use the GPU.

# API

See docstrings in `dldb/tdfs.py` and `dldb/dldb.py`

# Usage

### `tdfs()` function

Generates a feature matrix and list of Featuretools feature definitions. This function
takes in the same parameters as `featuretools.dfs`, with the following additional ones:

 * `window_size (str or pandas.DateOffset)`: amount of time between each cutoff time in the created time series
 * `start (datetime.datetime or pd.Timestamp)`: first cutoff time in the created time series
 * `num_windows (int)`: number of cutoff times in the created time series

Only 2 of these need to be specified to uniquely determine an equally-spaced set of cutoff times at which to compute each instance.

Let's say the final cutoff times (which could be directly passed into `featuretools.dfs()`) look like this:

```
cutoffs = pd.DataFrame({
  'customer_id': [13458, 13602, 15222],
  'cutoff_time': [pd.Timestamp('2011/12/15'), pd.Timestamp('2012/10/05'), pd.Timestamp('2012/01/25')]
})
```

Then passing in `window_size='3d'` and `num_windows=2` produces the following cutoff times to be passed into DFS:
```
pd.DataFrame({
  'customer_id': [13458, 13438, 13602, 13602, 15222, 15222],
  'cutoff_time': [pd.Timestamp('2011/12/12'), pd.Timestamp('2011/12/15'),
                  pd.Timestamp('2012/10/02'), pd.Timestamp('2012/10/05'),
                  pd.Timestamp('2012/01/22'), pd.Timestamp('2012/01/25')]
})
```

Example:
```
import featuretools as ft
from dldb import tdfs
entityset = ft.demo.load_retail()
feature_matrix, feature_defs = tdfs(entityset=entityset,
                                    target_entity='customers',
                                    cutoffs=cutoffs,
                                    window_size='3d',
                                    num_windows=2)
feature_matrix
>>> CustomerID | time       | Country        | Count(invoices) | ...
         13458 | 2011-12-12 | United Kingdom | 22              | ...
         13458 | 2011-12-15 | United Kingdom | 22              | ...
         13602 | 2012-10-02 | United Kingdom |  1              | ...
         13602 | 2012-10-05 | United Kingdom |  1              | ...
         15222 | 2012-01-22 | United Kingdom |  1              | ...
         15222 | 2012-01-25 | United Kingdom |  1              | ...
```
### `DLDB` class

Builds a recurrent neural network model using Keras from a time-varying feature matrix, and list of categorical feature names.

Specify hyperparameters in the constructor:

```
dldb = DLDB(regression=False, classes=[False, True],
            cell_type='GRU')
```

Then compile with the feature matrix and definitions:
```
dldb.compile(feature_matrix, feature_defs)
```

Or, if feature matrix was not generated from DFS, explicitly pass in the categorical feature names:
```
dldb.compile(feature_matrix_not_from_dfs,
             categorical_feature_names=['categorical1', 'categorical2'])
```

And fit:

```
labels = pd.Series([False, True, True],
                   index=[13458, 13602, 15222])
dldb.fit(feature_matrix, labels, batch_size=3, epochs=1)
predictions = dldb.predict(feature_matrix)
predictions
>>> array([[0.50211424],
           [0.5629099 ],
           [0.57218206]], dtype=float32)
```
