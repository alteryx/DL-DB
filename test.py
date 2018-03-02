from featuretools.tests.testing_utils import make_ecommerce_entityset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Imputer
import pandas as pd
import numpy as np
import featuretools as ft
import featuretools.selection
from model.build_keras_rnn import build_keras_rnn
from tdfs.utils import create_labels, sample_labels
from tdfs.tdfs import make_temporal_cutoffs


def test_ecommerce_binary():
    es = make_ecommerce_entityset()
    cutoffs = es['log'].df[['session_id', 'datetime']]
    cutoffs = cutoffs.rename(columns={'session_id': 'id'})
    fm, fl = ft.dfs(entityset=es,
                    cutoff_time=cutoffs,
                    target_entity="sessions",
                    cutoff_time_in_index=True)
    fm.sort_index(inplace=True)

    ids = fm.index.get_level_values('id').drop_duplicates()
    n_instances = ids.shape[0]
    labels = pd.Series(np.random.randint(2, size=(n_instances,)),
                       index=ids,
                       name='label').to_frame()
    fm = (fm.reset_index('id', drop=False)
            .merge(labels, left_on='id',
                   right_index=True,
                   how='left')
            .set_index('id', append=True)
          )

    train_fm, test_fm = train_test_split(
        fm, test_size=0.4, shuffle=False)
    train_labels = train_fm['label'].values
    test_labels = test_fm['label'].values
    del train_fm['label']
    del test_fm['label']

    dl_model, input_transform = build_keras_rnn(
        train_fm, fl, train_labels, categorical_max_vocab=10)

    train_inputs, train_outputs = input_transform(train_fm, train_labels)
    dl_model.fit(train_inputs, train_outputs, epochs=1, batch_size=4)

    test_inputs = input_transform(test_fm)
    predictions = dl_model.predict(test_inputs)
    score = roc_auc_score(test_labels, predictions)
    return predictions, score

def test_retail_binary():
    es = ft.demo.load_retail()
    labels = create_labels(es, min_training_data='8 days',lead='7 days',window='30 days', reduce='sum',binarize=None,iterate_by=None)
    labels_binary = labels > 300
    sampled = sample_labels(labels_binary)
    cutoffs = make_temporal_cutoffs(sampled[['CustomerID', 'time']], window_size='30d', num_windows=5)
    sampled = sampled[['CustomerID', 'label']]

    fm, fl = ft.dfs(entityset=es,
                    cutoff_time=cutoffs,
                    target_entity="customers",
                    cutoff_time_in_index=True,
                    verbose=True)
    fm.sort_index(inplace=True)

    fm = (fm.reset_index('CustomerID', drop=False)
            .reset_index(drop=False)
            .merge(sampled, on='CustomerID',how='left')
            .set_index('CustomerID')
            .set_index('time', append=True))

    train_fm, test_fm = train_test_split(
        fm, test_size=0.1, shuffle=False)
    train_labels = (train_fm['label']
                    .reset_index('CustomerID', drop=False)
                    .drop_duplicates('CustomerID')
                    .set_index('CustomerID'))['label'].values
    test_labels = (test_fm['label']
                   .reset_index('CustomerID', drop=False)
                   .drop_duplicates('CustomerID')
                   .set_index('CustomerID'))['label'].values
    del train_fm['label']
    del test_fm['label']

    dl_model, input_transform = build_keras_rnn(
        train_fm, fl, train_labels,
        recurrent_layer_sizes=(10,),
        dense_layer_sizes=(10,),
        categorical_max_vocab=10)

    train_inputs, train_outputs = input_transform(train_fm, train_labels)
    dl_model.fit(train_inputs, train_outputs, validation_split=.1,
                 epochs=10, batch_size=32)

    test_inputs = input_transform(test_fm)
    predictions = dl_model.predict(test_inputs)
    score = roc_auc_score(test_labels, predictions)

    baseline_fm, baseline_fl = ft.dfs(entityset=es,
                    cutoff_time=sampled,
                    target_entity="customers",
                    cutoff_time_in_index=True,
                    verbose=True)
    baseline_fm, baseline_fl = ft.encode_features(baseline_fm, baseline_fl)
    baseline_fm, baseline_fl = ft.selection.remove_low_information_features(baseline_fm, baseline_fl)
    baseline_fm.sort_index(inplace=True)

    baseline_fm = (baseline_fm.reset_index('CustomerID', drop=False)
            .reset_index(drop=False)
            .merge(sampled, on='CustomerID',how='left')
            .set_index('CustomerID')
            .set_index('time', append=True))

    baseline_train_fm, baseline_test_fm = train_test_split(
        baseline_fm, test_size=0.1, shuffle=False)
    baseline_train_labels = (baseline_train_fm['label']
                    .reset_index('CustomerID', drop=False)
                    .drop_duplicates('CustomerID')
                    .set_index('CustomerID'))['label'].values
    baseline_test_labels = (baseline_test_fm['label']
                   .reset_index('CustomerID', drop=False)
                   .drop_duplicates('CustomerID')
                   .set_index('CustomerID'))['label'].values
    del baseline_train_fm['label']
    del baseline_test_fm['label']
    baseline_score = score_baseline_pipeline(baseline_train_fm, baseline_train_labels,
                                             baseline_test_fm, baseline_test_labels)
    return score, baseline_score


def score_baseline_pipeline(X_train, y_train, X_test, y_test):
    feature_names = X_train.columns
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=feature_names)

    original_train_fm = X_train
    select_n_features = 200
    selector_rf = RandomForestClassifier(n_estimators=1000,
                                         class_weight='balanced',
                                         n_jobs=-1,
                                         verbose=True)
    selector_rf.fit(original_train_fm, y_train)

    importances = sorted(zip(selector_rf.feature_importances_, feature_names),
                         key=lambda x: x[0], reverse=True)
    selected = [i[1] for i in importances[:select_n_features]]

    X_train = original_train_fm[selected]

    # Train another Random Forest on selected features as our model

    model_rf = RandomForestClassifier(n_estimators=400,
                                      class_weight='balanced',
                                      n_jobs=-1)
    model_rf.fit(X_train, y_train)

    # Calculate features on test data

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    X_test = X_test[selected]

    # Predict targets for test data

    predicted_targets = model_rf.predict(X_test)

    # Compute metrics

    score = roc_auc_score(y_test, predicted_targets)
    return score

if __name__ == '__main__':
    score, baseline_score= test_ecommerce_binary()
    print("ROC score:", score)
    print("Baseline ROC score (using RF):", baseline_score)
