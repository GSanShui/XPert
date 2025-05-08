# dataset： cdsdb

# five-folds, from scratch
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_1,split_2,split_3,split_4,split_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_cold_drug_1,split_cold_drug_2,split_cold_drug_3,split_cold_drug_4,split_cold_drug_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_cold_tissue_1,split_cold_tissue_2,split_cold_tissue_3,split_cold_tissue_4,split_cold_tissue_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True


# five-folds, pretrain-finetune
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_1,split_2,split_3,split_4,split_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_cold_drug_1,split_cold_drug_2,split_cold_drug_3,split_cold_drug_4,split_cold_drug_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_cold_tissue_1,split_cold_tissue_2,split_cold_tissue_3,split_cold_tissue_4,split_cold_tissue_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth


# cdsdb-breast
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_breast_1,split_breast_2,split_breast_3,split_breast_4,split_breast_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_breast_1,split_breast_2,split_breast_3,split_breast_4,split_breast_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth

# cdsdb-leukemia
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_leukemia_1,split_leukemia_2,split_leukemia_3,split_leukemia_4,split_leukemia_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_cdsdb --drug_feat unimol --nfold split_leukemia_1,split_leukemia_2,split_leukemia_3,split_leukemia_4,split_leukemia_5 --dataset cdsdb_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
