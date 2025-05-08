# dataset： panacea_mdmt

# five-folds, from scratch
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_1,split_2,split_3,split_4,split_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_cold_drug_1,split_cold_drug_2,split_cold_drug_3,split_cold_drug_4,split_cold_drug_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_cold_cell_1,split_cold_cell_2,split_cold_cell_3,split_cold_cell_4,split_cold_cell_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True


# five-folds, pretrain-finetune
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_1,split_2,split_3,split_4,split_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_cold_drug_1,split_cold_drug_2,split_cold_drug_3,split_cold_drug_4,split_cold_drug_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth
# python train_xpert.py --model XPert --config config_panacea --drug_feat unimol --nfold split_cold_cell_1,split_cold_cell_2,split_cold_cell_3,split_cold_cell_4,split_cold_cell_5 --dataset panacea_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model saved_model/pretrain_mdmt_full_50_epoch.pth

