# DLDB

Deep learning for time-varying multi-entity datasets

# Installation

You should be able to just run:
```
pip install -e .
```

If that fails due to Tensor Flow, please visit [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/) and follow their instructions for installing Tensor Flow on your system.
You can also follow their instructions to install the GPU version to allow DLDB to use the GPU.

Be aware that recently users have reported issues installing Tensor Flow on Macs due to a new version of GRPC failing to make. If that happens, try installing grpc==1.9.1 and tensorflow without "-U" or "--upgrade":

```
pip install gprc==1.9.1 tensorflow
```

# API

See docstrings in `dldb/preprocessing.py` and `dldb/dldb.py`

# Usage
### `DLDB` class

Builds a recurrent neural network model using Keras from a feature tensor (flattened along the time/sequence dimension into a 2D Pandas DataFrame), and list of categorical feature names.

Specify hyperparameters in the constructor:

```
dldb = DLDB(regression=False, classes=[False, True],
            cell_type='GRU')
```

Then compile with the feature tensor and definitions:
```
dldb.compile(feature_tensor, feature_defs)
```

Or, if feature tensor was not generated from DFS, explicitly pass in the categorical feature names:
```
dldb.compile(feature_tensor_not_from_dfs,
             categorical_feature_names=['categorical1', 'categorical2'])
```

And fit:

```
labels = pd.Series([False, True, True],
                   index=[13458, 13602, 15222])
dldb.fit(feature_tensor, labels, batch_size=3, epochs=1)
predictions = dldb.predict(feature_tensor)
predictions
>>> array([[0.50211424],
           [0.5629099 ],
           [0.57218206]], dtype=float32)
```

### `MLPreprocessing` class
