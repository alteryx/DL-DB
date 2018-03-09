import pandas as pd
import numpy as np
from tqdm import tqdm


def create_labels(entityset,
                  min_training_data='28 days',
                  lead='7 days',
                  window='28 days',
                  reduce='sum',
                  binarize=None,
                  iterate_by=None):
    """
    Inputs
      * entityset (ft.EntitySet): featuretools EntitySet for SLA violation
            predictor
      * min_training_data (str): number with appropriate units representing the
            minimum abount of training data each project must have to have
            a label computed for it
      * lead (str): number with appropriate units representing the
            amount of time in the future to predict
      * window (str): number with appropriate units representing the
            amount of time in which to check for violations
      * reduce (str): how to reduce the multiple values for SLAs met/total SLAs
            across different weeks. Sum will sum up the SLAs met and total SLAs
            and then divide
      * binarize (function): Function that takes in a numeric label and returns
            a True/False value
      * iterate_by (str): number with appropriate units representing the
            amount of time to iterate the label generation algorithm. In
            other words, this is the amount of time between successive labels.
            Defaults to the same size as `window`.
    """

    label_cols = ['Quantity', 'UnitPrice']
    time_index = "InvoiceDate"
    index = "CustomerID"
    df = entityset['invoices'].df.merge(
        entityset['item_purchases'].df, how='outer')

    tqdm.pandas(desc="Creating Labels", unit="customer")

    # # Only use data after one of the label columns has been non-null
    # for i, v in df[label_cols].iterrows():
        # if v.dropna(how='all').shape[0] > 0:
            # df = df.loc[slice(i, None), :]
            # break
    grouped = df.groupby(index, as_index=True)

    project_cutoff_dates = grouped.progress_apply(
        lambda df: make_labels_from_windows(
            df,
            cols=label_cols,
            min_training_data=min_training_data,
            lead=lead, window=window,
            index_col=index,
            date_col=time_index,
            reduce=reduce,
            iterate_by=iterate_by))

    project_cutoff_dates = project_cutoff_dates.dropna()

    cutoff_with_labels = (project_cutoff_dates.reset_index(level=0)
                                              .reset_index()
                                              .rename(columns={'index': 'time',
                                                               0: 'label'}))
    if binarize:
        cutoff_with_labels['label'] = binarize(cutoff_with_labels['label'])

    return (cutoff_with_labels[[index, "time", "label"]]
            .sort_values(["time", index]))


def sample_labels(labels, random_seed=1, n=1, gap=None):
    """
    Select 1 label per customer
    """
    def sample(df):
        if gap is not None:
            samples = [df.iloc[0]]
            for i, row in df.iloc[1:].iterrows():
                if row['time'] - samples[-1]['time'] > gap:
                    samples.append(row)
            samples = pd.DataFrame(samples)
            return samples.sample(min(n, samples.shape[0]), random_state=random_seed)
        else:
            return df.sample(min(n, df.shape[0]), random_state=random_seed)

    labels = labels.groupby(labels['CustomerID']).apply(sample)
    return labels.sort_values(['time', 'CustomerID'])


def make_labels_from_windows(df, cols,
                             min_training_data, lead, window,
                             index_col, date_col,
                             reduce='min', iterate_by=None):
    customer_id = df[index_col].iloc[0]

    if iterate_by is not None:
        iterate_by = pd.Timedelta(iterate_by)

    vals = df[[date_col] + cols]

    date_series = vals[date_col]
    start = date_series.min() + pd.Timedelta(min_training_data)
    end = date_series.max()

    if end - start < pd.Timedelta(lead):
        return pd.Series([np.nan], index=[pd.NaT], name=customer_id)
    else:
        labels = iterate_through_cutoffs(vals, start, end,
                                         pd.Timedelta(window),
                                         pd.Timedelta(lead),
                                         cols,
                                         date_col,
                                         reduce,
                                         iterate_by=iterate_by)
        labels.name = customer_id
        return labels


def iterate_through_cutoffs(vals, start, end, window, lead, cols,
                            date_col,
                            reduce, iterate_by):
    labels = []
    cutoffs = []
    cutoff = start
    if iterate_by is None:
        iterate_by = window

    while cutoff + lead < end:
        start_window = cutoff + lead
        end_window = start_window + window
        _vals = vals[(vals[date_col] > start_window) &
                     (vals[date_col] < end_window)]

        label_vals = np.multiply(*[_vals[c] for c in cols])
        label = getattr(label_vals.dropna(), reduce)()
        labels.append(label)
        cutoffs.append(cutoff)
        cutoff = cutoff + iterate_by
    return pd.Series(labels, index=cutoffs)
