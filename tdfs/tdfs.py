import featuretools as ft
import pandas as pd

def tdfs(target_entity, cutoffs, window_size, training_window, num_windows, entityset, **kwargs):
    temporal_cutoffs = make_temporal_cutoffs(cutoffs, window_size, num_windows)
    
    
    feature_matrix, feature_defs = ft.dfs(entityset=entityset,
                                          cutoff_time=temporal_cutoffs,
                                          target_entity="customers",
                                          cutoff_time_in_index=True,
                                          training_window=training_window,
                                           **kwargs)
    return feature_matrix.sort_index()
    
    
    
    
def make_temporal_cutoffs(cutoffs, window_size, num_windows):
    out = []
    for r in cutoffs.itertuples():
        to_add = pd.DataFrame()
        to_add['instance_id'] = [r[1]] * num_windows
        to_add["cutoff_time"] = pd.date_range(end=r[2], periods=num_windows, freq=window_size)

        out.append(to_add)
        
    return pd.concat(out).reset_index(drop=True)
  