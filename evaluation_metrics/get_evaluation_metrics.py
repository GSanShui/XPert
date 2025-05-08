from math import sqrt
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from scipy.spatial.distance import cdist
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


def compute_mmd(pred_data, true_data, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        dist_pred = cdist(pred_data, pred_data, metric='sqeuclidean')
        dist_truth = cdist(true_data, true_data, metric='sqeuclidean')
        dist_cross = cdist(pred_data, true_data, metric='sqeuclidean')

        Kxx = np.exp(-gamma * dist_pred)
        Kyy = np.exp(-gamma * dist_truth)
        Kxy = np.exp(-gamma * dist_cross)

        return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf'.")


def compute_wasserstein(pred_data, true_data):
    return wasserstein_distance(pred_data.flatten(), true_data.flatten())



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
    
    # metrics for pert
    metrics = {}
    metrics['MSE'] = round(float(mse(y, f)), 3)
    metrics['RMSE'] = round(rmse(y, f), 3)
    metrics['MAE'] = round(float(mae(y, f)), 3)
    metrics['R2'] = round(r2_score(y, f), 3)
    metrics['Pearson'] = round(pearson(y, f), 3)
    metrics['Spearman'] = round(spearman(y, f), 3)

    # MMD
    mmd_per_gene = []
    for gene_idx in range(f.shape[1]):
        gene_pred = f[:, gene_idx].reshape(-1, 1)
        gene_truth = y[:, gene_idx].reshape(-1, 1)
        mmd = compute_mmd(gene_pred, gene_truth, kernel='rbf', gamma=1.0)
        mmd_per_gene.append(mmd)
    average_mmd = np.mean(mmd_per_gene)

    # Wasserstein
    ws_per_gene = []
    for gene_idx in range(f.shape[1]):
        gene_pred = f[:, gene_idx].reshape(-1, 1)
        gene_truth = y[:, gene_idx].reshape(-1, 1)
        ws = compute_wasserstein(gene_pred, gene_truth)
        ws_per_gene.append(ws)
    average_ws = np.mean(ws_per_gene)

    metrics['MMD'] = round(average_mmd, 3)
    metrics['Wasserstein'] = round(average_ws, 3)

    # metrics for deg
    y_deg = y - ctl
    f_deg = f - ctl
    metrics['R2_deg'] = round(r2_score(y_deg, f_deg), 3)
    metrics['Pearson_deg'] = round(pearson(y_deg, f_deg), 3)
    metrics['Spearman_deg'] = round(spearman(y_deg, f_deg), 3)
    

    for k in precision_degree:
        precision_neg, precision_pos = precision_k(y_deg, f_deg, k)
        metrics[f'Precision@{k} Positive_deg'] = round(precision_pos, 3)
        metrics[f'Precision@{k} Negative_deg'] = round(precision_neg, 3)
    

    # MMD和Wasserstein是考察特定基因的分布的
    # MMD
    mmd_per_gene = []
    for gene_idx in range(f_deg.shape[1]):
        gene_pred = f_deg[:, gene_idx].reshape(-1, 1)
        gene_truth = y_deg[:, gene_idx].reshape(-1, 1)
        mmd = compute_mmd(gene_pred, gene_truth, kernel='rbf', gamma=1.0)
        mmd_per_gene.append(mmd)

    average_mmd = np.mean(mmd_per_gene)

    # Wasserstein
    ws_per_gene = []
    for gene_idx in range(f_deg.shape[1]):
        gene_pred = f_deg[:, gene_idx].reshape(-1, 1)
        gene_truth = y_deg[:, gene_idx].reshape(-1, 1)
        ws = compute_wasserstein(gene_pred, gene_truth)
        ws_per_gene.append(ws)

    average_ws = np.mean(ws_per_gene)

    metrics['MMD_deg'] = round(average_mmd, 3)
    metrics['Wasserstein_deg'] = round(average_ws, 3)

    return metrics



def get_metrics_distribution(y, f, ctl, precision_degree = [10, 20, 50, 100]):

    metrics = {}
    
    # MMD和Wasserstein是考察特定基因的分布的
    # MMD
    mmd_per_gene = []
    for gene_idx in range(f.shape[1]):
        gene_pred = f[:, gene_idx].reshape(-1, 1)
        gene_truth = y[:, gene_idx].reshape(-1, 1)
        mmd = compute_mmd(gene_pred, gene_truth, kernel='rbf', gamma=1.0)
        mmd_per_gene.append(mmd)

    average_mmd = np.mean(mmd_per_gene)

    # Wasserstein
    ws_per_gene = []
    for gene_idx in range(f.shape[1]):
        gene_pred = f[:, gene_idx].reshape(-1, 1)
        gene_truth = y[:, gene_idx].reshape(-1, 1)
        ws = compute_wasserstein(gene_pred, gene_truth)
        ws_per_gene.append(ws)

    average_ws = np.mean(ws_per_gene)

    metrics['MMD'] = round(average_mmd, 3)
    metrics['Wasserstein'] = round(average_ws, 3)

    return metrics