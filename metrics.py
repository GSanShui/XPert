from math import sqrt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def precision_k(label_test, label_predict, k, flag=False):
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test, axis=1)
    label_predict = np.argsort(label_predict, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    neg_test_set = label_test[:, :num_neg]
    pos_test_set = label_test[:, -num_pos:]
    neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:]
    
    for i in range(len(neg_test_set)):
        neg_test = set(neg_test_set[i])
        pos_test = set(pos_test_set[i])
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    
    mean_precision_neg = np.mean(precision_k_neg)
    mean_precision_pos = np.mean(precision_k_pos)
    
    if flag:
        return mean_precision_neg, mean_precision_pos, precision_k_neg, precision_k_pos
    else:
        return mean_precision_neg, mean_precision_pos


def mae(label_test, label_predict, flag=False):

    mae_value = mean_absolute_error(label_test, label_predict)
    if flag:
        mae_ls = []
        for y,f in zip(label_test, label_predict):
            mae_ls.append(mean_absolute_error(y, f))
        return mae_value, mae_ls
    else:
        return mae_value



def rmse(label_test, label_predict, flag=False):
    rmse_value = sqrt(mean_squared_error(label_test, label_predict))
    if flag:
        rmse_ls = []
        for y, f in zip(label_test, label_predict):
            rmse_ls.append(sqrt(mean_squared_error([y], [f])))
        return rmse_value, rmse_ls
    else:
        return rmse_value


def mse(label_test, label_predict, flag=False):
    mse_value = mean_squared_error(label_test, label_predict)
    if flag:
        mse_ls = []
        for y, f in zip(label_test, label_predict):
            mse_ls.append(mean_squared_error([y], [f]))
        return mse_value, mse_ls
    else:
        return mse_value


def pearson(label_test, label_predict, flag=False):
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(pearsonr(lb_test, lb_predict)[0])
    mean_score = np.mean(score)
    if flag:
        return mean_score, score
    else:
        return mean_score


def spearman(label_test, label_predict, flag=False):
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(spearmanr(lb_test, lb_predict)[0])
    mean_score = np.mean(score)
    if flag:
        return mean_score, score
    else:
        return mean_score



def get_metrics(y, f, precision_degree = [10, 20, 50, 100]):

    metrics = {}
    # metrics['MSE'] = round(mse(y, f), 3)
    metrics['MSE'] = round(float(mse(y, f)), 3)
    metrics['RMSE'] = round(rmse(y, f), 3)
    # metrics['MAE'] = round(mae(y, f), 3)
    metrics['MAE'] = round(float(mae(y, f)), 3)
    metrics['Pearson'] = round(pearson(y, f), 3)
    metrics['Spearman'] = round(spearman(y, f), 3)

    for k in precision_degree:
        precision_neg, precision_pos = precision_k(y, f, k)
        metrics[f'Precision@{k} Positive'] = round(precision_pos, 3)
        metrics[f'Precision@{k} Negative'] = round(precision_neg, 3)

    return metrics


# y: y_true; f: y_pred; ctl: control_exp
def get_metrics_new(y, f, ctl, precision_degree = [10, 20, 50, 100]):

    metrics = {}
    metrics['MSE'] = round(float(mse(y, f)), 3)
    metrics['RMSE'] = round(rmse(y, f), 3)
    metrics['MAE'] = round(float(mae(y, f)), 3)
    metrics['Pearson'] = round(pearson(y, f), 3)
    metrics['Spearman'] = round(spearman(y, f), 3)

    for k in precision_degree:
        precision_neg, precision_pos = precision_k(y, f, k)
        metrics[f'Precision@{k} Positive'] = round(precision_pos, 3)
        metrics[f'Precision@{k} Negative'] = round(precision_neg, 3)

    y_deg = y - ctl
    f_deg = f - ctl
    metrics['MSE_deg'] = round(float(mse(y_deg, f_deg)), 3)
    metrics['RMSE_deg'] = round(rmse(y_deg, f_deg), 3)
    metrics['MAE_deg'] = round(float(mae(y_deg, f_deg)), 3)
    metrics['Pearson_deg'] = round(pearson(y_deg, f_deg), 3)
    metrics['Spearman_deg'] = round(spearman(y_deg, f_deg), 3)

    for k in precision_degree:
        precision_neg, precision_pos = precision_k(y_deg, f_deg, k)
        metrics[f'Precision@{k} Positive_deg'] = round(precision_pos, 3)
        metrics[f'Precision@{k} Negative_deg'] = round(precision_neg, 3)

    return metrics



def get_metrics_infer(y, f, ctl, precision_degree=[10, 20, 50, 100]):
    metrics = {}
    metrics_all_ls = {}

    # MSE
    mse_value, mse_ls = mse(y, f, flag=True)
    metrics['MSE'] = round(float(mse_value), 3)
    metrics_all_ls['MSE'] = mse_ls

    # RMSE
    rmse_value, rmse_ls = rmse(y, f, flag=True)
    metrics['RMSE'] = round(rmse_value, 3)
    metrics_all_ls['RMSE'] = rmse_ls

    # MAE
    mae_value, mae_ls = mae(y, f, flag=True)
    metrics['MAE'] = round(float(mae_value), 3)
    metrics_all_ls['MAE'] = mae_ls

    # Pearson
    pearson_value, pearson_ls = pearson(y, f, flag=True)
    metrics['Pearson'] = round(pearson_value, 3)
    metrics_all_ls['Pearson'] = pearson_ls

    # Spearman
    spearman_value, spearman_ls = spearman(y, f, flag=True)
    metrics['Spearman'] = round(spearman_value, 3)
    metrics_all_ls['Spearman'] = spearman_ls

    for k in precision_degree:
        precision_neg, precision_pos, precision_k_neg_ls, precision_k_pos_ls = precision_k(y, f, k, flag=True)
        metrics[f'Precision@{k} Positive'] = round(precision_pos, 3)
        metrics[f'Precision@{k} Negative'] = round(precision_neg, 3)
        metrics_all_ls[f'Precision@{k} Positive'] = precision_k_pos_ls
        metrics_all_ls[f'Precision@{k} Negative'] = precision_k_neg_ls

    y_deg = y - ctl
    f_deg = f - ctl

    # MSE_deg
    mse_deg_value, mse_deg_ls = mse(y_deg, f_deg, flag=True)
    metrics['MSE_deg'] = round(float(mse_deg_value), 3)
    metrics_all_ls['MSE_deg'] = mse_deg_ls

    # RMSE_deg
    rmse_deg_value, rmse_deg_ls = rmse(y_deg, f_deg, flag=True)
    metrics['RMSE_deg'] = round(rmse_deg_value, 3)
    metrics_all_ls['RMSE_deg'] = rmse_deg_ls

    # MAE_deg
    mae_deg_value, mae_deg_ls = mae(y_deg, f_deg, flag=True)
    metrics['MAE_deg'] = round(float(mae_deg_value), 3)
    metrics_all_ls['MAE_deg'] = mae_deg_ls

    # Pearson_deg
    pearson_deg_value, pearson_deg_ls = pearson(y_deg, f_deg, flag=True)
    metrics['Pearson_deg'] = round(pearson_deg_value, 3)
    metrics_all_ls['Pearson_deg'] = pearson_deg_ls

    # Spearman_deg
    spearman_deg_value, spearman_deg_ls = spearman(y_deg, f_deg, flag=True)
    metrics['Spearman_deg'] = round(spearman_deg_value, 3)
    metrics_all_ls['Spearman_deg'] = spearman_deg_ls

    for k in precision_degree:
        precision_deg_neg, precision_deg_pos, precision_deg_k_neg_ls, precision_deg_k_pos_ls = precision_k(y_deg, f_deg, k, flag=True)
        metrics[f'Precision@{k} Positive_deg'] = round(precision_deg_pos, 3)
        metrics[f'Precision@{k} Negative_deg'] = round(precision_deg_neg, 3)
        metrics_all_ls[f'Precision@{k} Positive_deg'] = precision_deg_k_pos_ls
        metrics_all_ls[f'Precision@{k} Negative_deg'] = precision_deg_k_neg_ls

    return metrics, metrics_all_ls