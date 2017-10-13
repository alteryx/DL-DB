import pandas as pd

def make_spending_labels(es, cutoff_time, prediction_window, training_window=None, lead=None):
    data = es["item_purchases"].df.merge(es["invoices"].df)[["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]]
    data["amount"] = data["Quantity"] * data["UnitPrice"]

    prediction_window_start = cutoff_time
    prediction_window_end = cutoff_time + prediction_window
    
    if lead is not None:
        cutoff_time = cutoff_time - lead

    t_start = None
    if training_window is not None:
        t_start = cutoff_time - training_window

    if t_start is not None:
        training_data = data[(data["InvoiceDate"] <= cutoff_time) & (data["InvoiceDate"] > t_start)]
    else:
        training_data = data[(data["InvoiceDate"] <= cutoff_time)]
    prediction_data = data[(data["InvoiceDate"] > prediction_window_start) & (data["InvoiceDate"] < prediction_window_end)]


    # get customers in training data
    label_times = pd.DataFrame()
    label_times["CustomerID"] = training_data["CustomerID"].dropna().unique()
    label_times["cutoff_time"] = cutoff_time
    label_times["t_start"] = t_start




    labels = prediction_data.groupby("CustomerID")[["amount"]].sum()

    label_times = label_times.merge(labels, how="left", left_on="CustomerID", right_index=True)

    # if the amount is nan that means the customer made no purchases in prediction window
    label_times["amount"] = label_times["amount"].fillna(0) 
    label_times.rename(columns={"amount": "total_spent"}, inplace=True)
    
    label_times.drop("t_start", axis=1, inplace=True)
    ## label_times.set_index("CustomerID", inplace=True)

    return label_times   