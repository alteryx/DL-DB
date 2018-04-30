from labeling_utils import create_labels, sample_labels
import pandas as pd
import featuretools as ft
import os


def construct_retail_example(ftens_file='retail_binary_files/ftens.csv',
                             labels_file='retail_binary_files/labels.csv',
                             fl_file='retail_binary_files/fl.p'):
    es = ft.demo.load_retail()
    if os.path.exists(ftens_file):
        ftens = pd.read_csv(ftens_file, index_col=['customer_id', 'time'], parse_dates=['time'])
        labels = pd.read_csv(labels_file, index_col='customer_id')['label']
        fl = ft.load_features(fl_file, es)
    else:
        labels = create_labels(es,
                               min_training_data='8 days',
                               lead='7 days',
                               window='30 days',
                               reduce='sum',
                               binarize=None,
                               iterate_by=None)
        labels_binary = labels.copy()
        labels_binary['label'] = labels_binary['label'] > 300
        sampled = sample_labels(labels_binary, n=1)
        sampled = sampled[['customer_id', 'time', 'label']]
        sampled = sampled.sample(300)

        ftens, fl = ft.tdfs(target_entity='customers',
                      entityset=es,
                      cutoffs=sampled,
                      window_size='30d',
                      num_windows=5,
                      verbose=True)

        ftens = (ftens.reset_index('customer_id', drop=False)
                .reset_index(drop=False)
                .merge(sampled[['customer_id', 'label']],
                       on='customer_id',
                       how='left')
                .set_index('customer_id')
                .set_index('time', append=True))

        labels = (ftens['label']
                  .reset_index('customer_id', drop=False)
                  .drop_duplicates('customer_id')
                  .set_index('customer_id'))
        del ftens['label']
        ftens.to_csv(ftens_file)
        labels.to_csv(labels_file)
        labels = labels['label']
        ft.save_features(fl, fl_file)
    return ftens, labels, fl


if __name__ == '__main__':
    if not os.path.exists('retail_binary_files'):
        os.makedirs('retail_binary_files')
    construct_retail_example()
