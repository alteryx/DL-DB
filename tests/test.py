from featuretools.tests.testing_utils import make_ecommerce_entityset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Imputer
import pandas as pd
import numpy as np
import featuretools as ft
from featuretools.selection import remove_low_information_features
from dldb import DLDB
from testing_utils import construct_retail_example


def f1_macro(actual, predicted):
    return f1_score(actual, predicted, average='macro')


def test_ecommerce():
    es = make_ecommerce_entityset()
    cutoffs = es['log'].df[['session_id', 'datetime']]
    cutoffs = cutoffs.rename(columns={'session_id': 'id'})
    ftens, fl = ft.dfs(entityset=es,
                    cutoff_time=cutoffs,
                    target_entity="sessions",
                    cutoff_time_in_index=True)
    ftens.sort_index(inplace=True)

    ids = ftens.index.get_level_values('id').drop_duplicates()
    n_instances = ids.shape[0]

    labels_binary = [i % 2 for i in range(n_instances)]
    labels_multiclass = np.random.randint(10, size=(n_instances,))
    labels_regression = np.random.random(size=(n_instances,))
    labels = pd.DataFrame({'label_binary': labels_binary,
                           'label_multiclass': labels_multiclass,
                           'label_regression': labels_regression},
                          index=ids)

    ftens = (ftens.reset_index('id', drop=False)
            .merge(labels, left_on='id',
                   right_index=True,
                   how='left')
            .set_index('id', append=True)
          )

    train_ftens, test_ftens = train_test_split(
        ftens, test_size=0.4, shuffle=False)
    train_labels = train_ftens[labels.columns]
    test_labels = test_ftens[labels.columns]
    for c in labels.columns:
        del train_ftens[c]
        del test_ftens[c]

    scores = {}
    scoring_functions = {'label_regression': mean_absolute_error,
                         'label_binary': roc_auc_score,
                         'label_multiclass': f1_macro}
    for label_type in labels.columns:
        classes = labels[label_type].unique()
        dl_model = DLDB(
            regression=label_type == 'label_regression',
            classes=classes,
            categorical_max_vocab=10)
        dl_model.compile(train_ftens, fl)
        dl_model.fit(train_ftens,
                     train_labels[label_type].values,
                     epochs=1,
                     batch_size=4)
        predictions = dl_model.predict(test_ftens)
        score = scoring_functions[label_type](test_labels[label_type].values,
                                              predictions)
        scores[label_type] = score
    return scores


def test_retail_binary(ftens_file='retail_binary_files/ftens.csv',
                       labels_file='retail_binary_files/labels.csv',
                       fl_file='retail_binary_files/fl.p'):
    ftens, labels, fl = construct_retail_example(ftens_file, labels_file, fl_file)
    baseline_ftens = (ftens.reset_index('customer_id', drop=False)
                     .drop_duplicates('customer_id', keep='last')
                     .set_index('customer_id'))
    baseline_ftens, baseline_fl = ft.encode_features(baseline_ftens, fl)
    baseline_ftens, baseline_fl = remove_low_information_features(baseline_ftens, baseline_fl)
    train_customers, test_customers = train_test_split(baseline_ftens.index.values, shuffle=True, test_size=0.1)
    train_labels = labels.loc[train_customers]
    test_labels = labels.loc[test_customers]
    train_ftens = ftens.loc[(train_customers, slice(None)), :]
    test_ftens = ftens.loc[(test_customers, slice(None)), :]
    baseline_train_fm = baseline_ftens.loc[train_customers, :]
    baseline_test_fm = baseline_ftens.loc[test_customers, :]

    dl_model = DLDB(
        regression=False,
        classes=[False, True],
        recurrent_layer_sizes=(32,),
        dense_layer_sizes=(32, 32),
        categorical_max_vocab=10)
    dl_model.compile(train_ftens, fl)
    dl_model.fit(
            train_ftens,
            train_labels,
            validation_split=0.1,
            epochs=1,
            batch_size=32)
    predictions = dl_model.predict(test_ftens)
    score = roc_auc_score(test_labels, predictions)

    baseline_scores = score_baseline_pipeline(baseline_train_fm,
                                              train_labels,
                                              baseline_test_fm,
                                              test_labels)
    return score, baseline_scores


def score_baseline_pipeline(X_train, y_train, X_test, y_test, **hyperparams):
    feature_names = X_train.columns
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=feature_names)

    original_train_fm = X_train
    select_n_features = hyperparams.get('select_n_features', 200)
    selector_rf = RandomForestClassifier(n_estimators=hyperparams.get('selector_n_estimators', 1000),
                                         class_weight='balanced',
                                         n_jobs=-1,
                                         verbose=True)
    selector_rf.fit(original_train_fm, y_train)

    importances = sorted(zip(selector_rf.feature_importances_, feature_names),
                         key=lambda x: x[0], reverse=True)
    selected = [i[1] for i in importances[:select_n_features]]

    X_train = original_train_fm[selected]

    # Train another Random Forest on selected features as our model

    model_rf = RandomForestClassifier(n_estimators=hyperparams.get('n_estimators', 400),
                                      class_weight='balanced',
                                      n_jobs=-1)
    model_rf.fit(X_train, y_train)

    model_svm = SVC()
    model_svm.fit(X_train, y_train)

    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    X_test = X_test[selected]

    # Predict targets for test data

    predicted_targets = model_rf.predict(X_test)
    predicted_targets_svm = model_svm.predict(X_test)
    predicted_targets_lr = model_lr.predict(X_test)

    # Compute metrics

    score_rf = roc_auc_score(y_test, predicted_targets)
    score_svm = roc_auc_score(y_test, predicted_targets_svm)
    score_lr = roc_auc_score(y_test, predicted_targets_lr)
    return {'rf': score_rf, 'svm': score_svm, 'lr': score_lr}


if __name__ == '__main__':
    #scores = test_ecommerce()
    score, baseline_scores = test_retail_binary()
    print("ROC score:", score)
    print("Baseline ROC scores (using RF, SVM, LogisticRegression):", baseline_scores)
