# XPert: A Knowledge-Informed Dual-Branch Transformer Model for Predicting Drug-Induced Cellular Perturbation Effects

## Introduction
Systematic mapping of chemical perturbation responses is revolutionizing polypharmacological drug discovery, yet remains constrained by experimental scalability. Here, we introduce XPert, a biologically informed Dual-Branch Transformer model that predicts gene-specific drug responses across dose-time conditions, outperforming VAE-based methods. It generalizes to unseen cells, transfers knowledge to clinical settings, and reveals mechanistic insights, offering a scalable solution for precision medicine and perturbation-based drug discovery. 

## Overview
The repository is organised as follows:
- `processed_data/` contains processed data files;
- `HG_data/` contains the data for training the knowledge heterogeneous graph and the pre-trained embeddings;
- `dataset/` contains the necessary files for creating the dataset;
- `models/` contains different modules of XPert;
- `configs/` contains all the config files adapted for different datasets and scenarios;
- `experiment/` contains log files and output files;
- `scripts/` contains the scripts for training, inference, and testing the model;
- `saved_model/` contains the trained and pretrained weights;
- `evaluation_metrics/` contains all the evaluation metrics mentioned in the paper;
- `reproducing/` contains the code for reproducing the analysis results and figures from the paper;

## Requirements
The XPert network is built using PyTorch. You can use following commands to create conda env with related dependencies.
``` 
conda create -n xpert python=3.9
pip install -r requirements.txt
``` 

## Implementation

**Note:** Due to file size limitations, some files are not included in this repository. Please refer to [Zenodo](https://zenodo.org/uploads/15357712) and [Figshare] for the complete set of resources.

### Data Format
1. Paired pre- and post-treatment datasets are stored in `h5ad` format. The post-treatment gene expression data is stored in `adata.X`, while the pre-treatment data is stored in `adata.obsm["X_ctl"]`, and metadata is stored in `adata.obs`. All datasets used for XPert follow this format.

2. The processed datasets can be accessed at Figshare (URL to be updated after paper acceptance).

### Model Training
The scripts for training, inference, and testing the model are located in the `scripts/` folder. Example usage is shown below:

Run the following command to train XPert using the L1000 dataset (for predictions in multi-dose-multi-time mode, change the dataset suffix to `_mdmt`, e.g., `l1000_mdmt`). Set the `use_gradscaler` parameter to `True` to deploy flash attention for accelerating the attention computation.

``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
``` 


### Model Finetuning

Our paper demonstrates the powerful potential of using the L1000 dataset as a pre-training dataset. If you wish to extend this application to other datasets, we recommend using the finetuning mode.

All trained models are available for download on [Figshare](URL to be updated after paper acceptance). Please store them in the `saved_models/` folder. Modify the config file for different datasets and specify the path to the pre-trained model using the `pretrained_model` parameter.


For cold-dose-time scenarios, use the following code:

```
python train_xpert.py --model XPert 
                      --config config_l1000_cdt 
                      --drug_feat unimol
                      --nfold split_cold_dose&time_random_1_one_shot
                      --dataset l1000_mdmt
                      --use_gradscaler True
                      --pretrained_model saved_model/pretrain_mdmt_new.pth
```

For independent datasets, use the following code:
```
python train_xpert.py --model XPert 
                      --config config_cdsdb
                      --drug_feat unimol
                      --nfold split_breast_1
                      --dataset cdsdb_mdmt
                      --use_gradscaler True
                      --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
```

### Model Testing/Inferencing

You can use our saved model for result reproduction or novel prediction. Please set the `output_profile` parameter to `True` if you wish to output the prediction profile.


``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
                      --mode test
                      --output_profile True
                      --saved_model_path saved_model/l1000_sdst_warm_split.pth
``` 

If you need to output the `cls` embeddings or attention matrix for downstream analysis, we recommend using the `Infer` mode. Please set the `output_cls_embed` and `output_attention` parameters as needed. The resulting outputs will be stored in the `experiment` directory.

``` 
python train_xpert.py --model XPert 
                      --config config_l1000
                      --drug_feat unimol
                      --nfold split_1
                      --dataset l1000_sdst
                      --use_gradscaler True
                      --mode infer
                      --output_cls_embed True
                      --output_attention True
                      --saved_model_path saved_model/l1000_sdst_warm_split.pth
``` 
