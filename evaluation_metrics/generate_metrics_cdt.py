import pandas as pd
import numpy as np
import argparse
from get_evaluation_metrics import get_metrics_new, get_metrics_distribution

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')

    return parser.parse_args()


args = arg_parse()
path = 'evaluation_metrics/cdt_results/'
model = args.model
print('model:', model)

folds = [1,2,3,4,5]
if "pretrain" in model:
    ratios = ["zero","one","0.2","0.3","0.5","0.8"]
else:
    ratios = ["one","0.2","0.3","0.5","0.8"]


result_dfs = []
for ratio in ratios:
    metrics_ls = []
    for k in folds:
        split = f"split_cold_dose&time_random_{k}_{ratio}_shot"
        print('     split:', split)

        profile_path = path+f'{model}/{split}_predict_profile.npy'
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
    result_df.index = [ratio]*len(folds)
    result_dfs.append(result_df)
merge_df = pd.concat(result_dfs, axis=0)
merge_df.to_csv(f'{path}/{model}_merge_results.csv')