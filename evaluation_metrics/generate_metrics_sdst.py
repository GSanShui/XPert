import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from get_evaluation_metrics import get_metrics_new, get_metrics_distribution


path = 'evaluation_metrics/sdst_results/'
models = ['deepce','prnet','transigen','xpert']
splits = ['split', 'split_cold_cell', 'split_cold_drug']
folds = [1,2,3,4,5]

for split in splits:
    result_dfs = []
    for model in models:
        print('model:', model)
        metrics_ls = []
        for k in folds:
            print('     fold:', k)
            profile_path = path+f'{model}/{split}_{k}_predict_profile.npy'
            print('     profile_path:', profile_path)
            profile = np.load(profile_path, allow_pickle=True).item()
            y_true = profile['y_true']
            print('     y_true.shape:', y_true.shape)
            y_pred = profile['y_pred']
            ctl_true = profile['ctl_true']
            metrics = get_metrics_new(y_true, y_pred, ctl_true)
            print('     metrics:', metrics)
            metrics_ls.append(metrics)
        result_df = pd.DataFrame(metrics_ls)
        result_df.index = [model]*len(folds)
        result_dfs.append(result_df)
    merge_df = pd.concat(result_dfs, axis=0)
    merge_df.to_csv(f'{path}/{split}_merge_results.csv')
    # merge_df.to_csv(f'{path}/xpert_{split}_merge_results.csv')