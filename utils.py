import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import scanpy as sc

from metrics import get_metrics_new, get_metrics_infer
from datasets.MyDataset import MyDataset



def load_dataloader(args,config,logger,nfold, return_rawdata=False):

    batch_size = config['train']['batch_size']

    if args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'


    if args.drug_feat == 'unimol':
        drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
        max_atom_size = config['dataset']['max_atom_size']
        if max_atom_size < 122:
            drug_feat = drug_feat[:, :max_atom_size, :]
            logger.info(f'Using {max_atom_size} atom')
    elif args.drug_feat == 'KPGT':
        drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
    elif args.drug_feat == 'smi':
        drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
    elif args.drug_feat == 'morgan':
        drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
    else:
        assert False, 'Drug feature not found!'


    tr_data = data[data.obs[nfold] == 'train']
    val_data = data[data.obs[nfold] == 'valid']
    test_data = data[data.obs[nfold] == 'test']
    

    # for five-fold cross-validation
    if val_data.n_obs == 0:
        val_data = test_data

    logger.info(f'train data:{tr_data.n_obs}')  
    logger.info(f'Valid data:{val_data.n_obs}')
    logger.info(f'Test data:{test_data.n_obs}')
    
    if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
        Dataset = MyDataset
    else:
        assert False, 'Dataset Class is not Assigned!'
    
    max_value = config['dataset']['max_value']
    min_value = config['dataset']['min_value']
    logger.info(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')
    
    logger.info("Starting load train data...")
    tr_dataset = Dataset(tr_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    logger.info("Starting load valid data...")
    val_dataset = Dataset(val_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    logger.info("Starting load test data...")
    test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    logger.info(f'train dataloader:{len(tr_dataloader)}')  
    logger.info(f'Valid dataloader:{len(val_dataloader)}')
    logger.info(f'Test dataloader:{len(test_dataloader)}')

    if return_rawdata:
        return tr_dataloader, val_dataloader, test_dataloader, data
    else:
        return tr_dataloader, val_dataloader, test_dataloader


def load_test_dataloader(args,config,logger,nfold):

    batch_size = config['train']['batch_size']

    if args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'

    if args.drug_feat == 'unimol':
        drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
        max_atom_size = config['dataset']['max_atom_size']
        if max_atom_size < 122:
            drug_feat = drug_feat[:, :max_atom_size, :]
            logger.info(f'Using {max_atom_size} atom')
    elif args.drug_feat == 'KPGT':
        drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
    elif args.drug_feat == 'smi':
        drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
    elif args.drug_feat == 'morgan':
        drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
    else:
        assert False, 'Drug feature not found!'


    test_data = data[data.obs[nfold] == 'test']

    logger.info(f'Test data:{test_data.n_obs}')

    if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
        Dataset = MyDataset
    else:
        assert False, 'Dataset Class is not Assigned!'

    max_value = config['dataset']['max_value']
    min_value = config['dataset']['min_value']
    logger.info(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')

    logger.info("Starting load test data...")
    test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    logger.info(f'Test dataloader:{len(test_dataloader)}')
    
    return test_dataloader



def load_infer_dataloader(args,config,logger,nfold):

    if args.output_attention:
        batch_size = 1
    else:
        batch_size = 256
    logger.info(f'batch_size:{batch_size}')

    if args.dataset == 'l1000_sdst':
        data = sc.read_h5ad(config['dataset']['l1000_sdst_data_root'])
    elif args.dataset == 'l1000_mdmt':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_data_root'])
    elif args.dataset == 'l1000_mdmt_pretrain':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_pretrain_data_root'])
    elif args.dataset == 'l1000_mdmt_full':
        data = sc.read_h5ad(config['dataset']['l1000_mdmt_full_data_root'])
    elif args.dataset == 'panacea_mdmt':
        data = sc.read_h5ad(config['dataset']['PANACEA_data_root'])
    elif args.dataset == 'cdsdb_mdmt':
        data = sc.read_h5ad(config['dataset']['CDSDB_data_root'])
    else:
        data = sc.read_h5ad(f'processed_data/{args.dataset}.h5ad')
        # assert False, 'Dataset not found!'


    if args.drug_feat == 'unimol':
        drug_feat = np.load(config['dataset']['drug_unimol_path'], allow_pickle=True)
        max_atom_size = config['dataset']['max_atom_size']
        if max_atom_size < 122:
            drug_feat = drug_feat[:, :max_atom_size, :]
            logger.info(f'Using {max_atom_size} atom')
    elif args.drug_feat == 'KPGT':
        drug_feat = np.load(config['dataset']['drug_KPGT_path'], allow_pickle=True).item()
    elif args.drug_feat == 'smi':
        drug_feat = np.load(config['dataset']['drug_smi_path'], allow_pickle=True).item()
    elif args.drug_feat == 'morgan':
        drug_feat = np.load(config['dataset']['drug_morgan_path'], allow_pickle=True).item()
    else:
        assert False, 'Drug feature not found!'

    
    # test_data = data[data.obs[nfold] == "infer" ]
    test_data = data

    logger.info(f'Test data:{test_data.n_obs}')

    if args.model in ['XPert', 'TranSiGen', 'DeepCE', 'PRnet', 'CIGER', 'MLP']:
        Dataset = MyDataset
    else:
        assert False, 'Dataset Class is not Assigned!'


    max_value = config['dataset']['max_value']
    min_value = config['dataset']['min_value']
    logger.info(f'For dataset {args.dataset}, max_value:{max_value}, min_value:{min_value}')

    logger.info("Starting load infer data...")
    test_dataset = Dataset(test_data, drug_feat, args=args, config=config, logger=logger, max_value=max_value, min_value=min_value)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)

    logger.info(f'Test dataloader:{len(test_dataloader)}')
    
    return test_dataloader



def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def log_nested_dict(raw_dict, logger, indent=0):
    for key, value in raw_dict.items():
        if isinstance(value, dict):
            logger.info(f'{"    " * indent}{key}:')
            log_nested_dict(value, logger, indent + 1)
        else:
            logger.info(f'{"    " * indent}{key}: {value}')


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def mse_loss_ls_sum(pred, target):
    return torch.mean((pred - target) ** 2, dim=1).sum()



def pcc_loss_sum(pred, target):
    """
    Calculate the Pearson Correlation Coefficient (PCC) loss.
    
    Args:
        pred (torch.Tensor): Model predicted values, shape (batch_size, n_features).
        target (torch.Tensor): Ground truth values, shape (batch_size, n_features).
    
    Returns:
        torch.Tensor: Computed PCC loss for the batch.
    """
    # Ensure the inputs are 2D tensors
    assert pred.ndim == 2 and target.ndim == 2, "Inputs must be 2D tensors."
    
    # Flatten along the batch dimension to compute correlation
    pred_mean = torch.mean(pred, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=1, keepdim=True)

    # Center the data
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Compute numerator and denominator for PCC
    numerator = torch.sum(pred_centered * target_centered, dim=1)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1) + 1e-6)

    # Compute PCC for each sample in the batch
    pcc = numerator / denominator

    # Loss is 1 - PCC, so maximizing PCC minimizes the loss
    loss = 1 - pcc

    # Return the mean loss across the batch
    return torch.sum(loss)




def cos_loss_sum(pred, target):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
        
    cos = nn.CosineSimilarity(dim=-1)
    cos_loss = 1 - cos(pred, target)
    return cos_loss.sum() 




def train(model, opt, dataloader, args, config, scaler=None, epoch=0):
    model.train()
    total_loss = 0.
    total_loss1 = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    total_loss4 = 0.
    num_batches = 0


    a, b = config['train']['loss_weight']
    for data in dataloader:

        model.zero_grad()

        if scaler:
            with autocast():
                output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
                num_samples =  trt_raw_data.shape[0]
                num_batches += num_samples
                loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                weighted_loss = loss1 * a + loss2 * b
                batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b
                total_loss += weighted_loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

            # scaler.scale(weighted_loss).backward()
            if epoch < 70:
                scaler.scale(batch_weighted_loss).backward()
            else:
                scaler.scale(weighted_loss).backward()
            
            scaler.step(opt)
            scaler.update()

        else:
            output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
            num_samples =  trt_raw_data.shape[0]
            num_batches += num_samples
            loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
            loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
            weighted_loss = loss1 * a + loss2 * b
            batch_weighted_loss = torch.sqrt(loss1/num_samples) * a + (loss2/num_samples) * b
            total_loss += weighted_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            if epoch < 70:
                batch_weighted_loss.backward()
            else:
                weighted_loss.backward()
            
            opt.step()


    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    # avg_loss1 = total_loss1 / num_batches if total_loss1 else -999
    avg_loss2 = total_loss2 / num_batches if total_loss2 else -999
    avg_loss3 = total_loss3 / num_batches if total_loss3 else -999
    avg_loss4 = total_loss4 / num_batches if total_loss4 else -999

    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4




def validate(model, dataloader, args, config, flag=False, scaler=None):
    model.eval()  
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0
    num_batches = 0
    trt_raw_data_ls = []
    output2_ls = []
    ctl_raw_data_ls = []
    

    with torch.no_grad():  
        for data in dataloader:
            if scaler:
                with autocast():
                    output_deg, trt_raw_data, ctl_raw_data, _ = model(data)
                    num_batches += trt_raw_data.shape[0] 
                    loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                    loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                    total_loss1 += loss1.item()
                    total_loss2 += loss2.item()
            else:
                output_deg, trt_raw_data, ctl_raw_data, _  = model(data)
                num_batches += trt_raw_data.shape[0] 
                loss1 = mse_loss_ls_sum(output_deg, trt_raw_data-ctl_raw_data)
                loss2 = cos_loss_sum(output_deg, trt_raw_data-ctl_raw_data)
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()                        

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(output_deg+ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)


    if flag:
        y_true = torch.cat(trt_raw_data_ls, dim=0).cpu().detach().numpy()
        y_pred = torch.cat(output2_ls, dim=0).cpu().detach().numpy()
        ctl_true = torch.cat(ctl_raw_data_ls, dim=0).cpu().detach().numpy()
        metrics = get_metrics_new(y_true, y_pred, ctl_true)
    else:
        metrics = None

    avg_total_loss = total_loss / num_batches
    avg_loss1 = np.sqrt(total_loss1 / num_batches) if total_loss1 else -999
    avg_loss2 = total_loss2 / num_batches if total_loss2 else -999
    avg_loss3 = total_loss3 / num_batches if total_loss3 else -999
    avg_loss4 = total_loss4 / num_batches if total_loss4 else -999

    return avg_total_loss, avg_loss1, avg_loss2, avg_loss3, avg_loss4, metrics





def infer(model, dataloader, args, config, flag=True, output_profile=False, scaler=None):
    model.eval()  
    trt_raw_data_ls = []
    output2_ls = []
    ctl_raw_data_ls = []
    pert_id_ls = []
    cell_id_ls = []
    pert_dose_ls = []
    pert_time_ls = []
    all_attention_dict_ls= {}


    with torch.no_grad():  

        for data in dataloader:
            trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx, pert_id, cell_id, pert_dose, pert_time = data
            # feat_data = trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx  

            if scaler:
                with autocast():
                    output_deg, _, _, attention_dict, _, _ = model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))
            else:
                output_deg, _, _, attention_dict, _, _ = model((trt_raw_data, ctl_raw_data, trt_raw_data_binned, ctl_raw_data_binned,drug_feat, pert_dose_idx, pert_time_idx, drug_idx, cell_idx, tissue_idx))                     

            if flag:
                trt_raw_data_ls.append(trt_raw_data)
                output2_ls.append(output_deg.cpu()+ctl_raw_data)
                ctl_raw_data_ls.append(ctl_raw_data)
                if args.output_attention:
                    for key, value in attention_dict.items():
                        if key not in all_attention_dict_ls:
                            all_attention_dict_ls[key] = []
                        if value is None:
                            print(f'value of {key} is None!')
                        all_attention_dict_ls[key].append(value)

                pert_id = list(drug_idx)
                cell_id = list(cell_id)
                pert_dose = list(pert_dose)
                pert_time = list(pert_time)

                pert_id_ls.extend(pert_id)
                cell_id_ls.extend(cell_id)
                pert_dose_ls.extend(pert_dose)
                pert_time_ls.extend(pert_time)


        if flag:
            y_true = torch.cat(trt_raw_data_ls, dim=0).cpu().detach().numpy()
            y_pred = torch.cat(output2_ls, dim=0).cpu().detach().numpy()
            ctl_true = torch.cat(ctl_raw_data_ls, dim=0).cpu().detach().numpy()
            metrics, metrics_all_ls = get_metrics_infer(y_true, y_pred, ctl_true)

            metrics_all_ls['pert_id'] = np.array(pert_id_ls)
            metrics_all_ls['cell_id'] = np.array(cell_id_ls)
            metrics_all_ls['pert_dose'] = np.array(pert_dose_ls)
            metrics_all_ls['pert_time'] = np.array(pert_time_ls)
        else:
            metrics = None
            metrics_all_ls = None

        if args.output_attention:
            all_attention_dict = {}
            for key, value in all_attention_dict_ls.items():
                all_attention_dict[key] = torch.cat(value, dim=0).cpu().detach().numpy()
        else:
           all_attention_dict = None 
        
        output_pred_profile = y_pred if output_profile else None

    return metrics, metrics_all_ls, output_pred_profile, all_attention_dict





class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=None, metric=None, n_fold=None, folder=None, logger=None):

        filepath = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filepath = filepath
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model, current_epoch, optimizer):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, current_epoch, optimizer):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        }, self.filepath)

    def load_checkpoint(self, model, optimizer):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model




def save_loss_fig(tr_loss_list,val_loss_list,n_fold,folder,logger):
    x = range(0, len(tr_loss_list))
    y1 = tr_loss_list
    y2 = val_loss_list
    plt.figure(dpi=150)  
    # plt.style.use('seaborn-bright')
    plt.plot(x, y1, '.-', label='Train loss', color="b")
    plt.plot(x, y2, '.-', label='Valid loss', color="g")
    plt.legend(loc='best',edgecolor='white')
    plt.xlabel('Epoches')
    plt.ylabel('Train/Valid loss')
    # plt.xticks(np.arange(0,len(tr_loss_list),25))
    
    plt.savefig(folder + f'/{n_fold}_fold_train_loss.png')

    logger.info('Loss figure saved!')



def set_requires_grad(model, finetune_params="all"):
    """
    Set requires_grad for model parameters.

    Args:
        model: The model containing parameters to update.
        finetune_params (list or all): List of parameter name fragments to finetune.
                                         If all, all parameters are set to requires_grad=True.
    """
    # enable all parameters
    if finetune_params == "all":
        pass
    else:
        # Otherwise, set requires_grad based on finetune_params
        for name, param in model.named_parameters():
            if any(module in name for module in finetune_params):
                param.requires_grad = True
            else:
                param.requires_grad = False
