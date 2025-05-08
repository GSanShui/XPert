# dataset: l1000_sdst

# five-fold
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_sdst --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_2,split_cold_cell_2,split_2 --dataset l1000_sdst --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_3,split_cold_cell_3,split_3 --dataset l1000_sdst --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_4,split_cold_cell_4,split_4 --dataset l1000_sdst --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_5,split_cold_cell_5,split_5 --dataset l1000_sdst --use_gradscaler True --include_cell_idx True



# dataset: l1000_mdmt

# five-fold
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_1,split_cold_cell_1,split_1 --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_2,split_cold_cell_2,split_2 --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_3,split_cold_cell_3,split_3 --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_4,split_cold_cell_4,split_4 --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_cold_drug_5,split_cold_cell_5,split_5 --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True


# cold-dose-time five-seeds from-scratch
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_1_one_shot','split_cold_dose&time_random_1_0.2_shot','split_cold_dose&time_random_1_0.3_shot','split_cold_dose&time_random_1_0.5_shot','split_cold_dose&time_random_1_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_2_one_shot','split_cold_dose&time_random_2_0.2_shot','split_cold_dose&time_random_2_0.3_shot','split_cold_dose&time_random_2_0.5_shot','split_cold_dose&time_random_2_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_3_one_shot','split_cold_dose&time_random_3_0.2_shot','split_cold_dose&time_random_3_0.3_shot','split_cold_dose&time_random_3_0.5_shot','split_cold_dose&time_random_3_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_4_one_shot','split_cold_dose&time_random_4_0.2_shot','split_cold_dose&time_random_4_0.3_shot','split_cold_dose&time_random_4_0.5_shot','split_cold_dose&time_random_4_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_5_one_shot','split_cold_dose&time_random_5_0.2_shot','split_cold_dose&time_random_5_0.3_shot','split_cold_dose&time_random_5_0.5_shot','split_cold_dose&time_random_5_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True


# cold-dose-time five-seeds pretrain-finetune
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_1_one_shot','split_cold_dose&time_random_1_0.2_shot','split_cold_dose&time_random_1_0.3_shot','split_cold_dose&time_random_1_0.5_shot','split_cold_dose&time_random_1_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_2_one_shot','split_cold_dose&time_random_2_0.2_shot','split_cold_dose&time_random_2_0.3_shot','split_cold_dose&time_random_2_0.5_shot','split_cold_dose&time_random_2_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_3_one_shot','split_cold_dose&time_random_3_0.2_shot','split_cold_dose&time_random_3_0.3_shot','split_cold_dose&time_random_3_0.5_shot','split_cold_dose&time_random_3_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_4_one_shot','split_cold_dose&time_random_4_0.2_shot','split_cold_dose&time_random_4_0.3_shot','split_cold_dose&time_random_4_0.5_shot','split_cold_dose&time_random_4_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_5_one_shot','split_cold_dose&time_random_5_0.2_shot','split_cold_dose&time_random_5_0.3_shot','split_cold_dose&time_random_5_0.5_shot','split_cold_dose&time_random_5_0.8_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --pretrained_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth


# cold-dose-time five-seeds zero-shot
# python train_xpert.py --model XPert --config config_l1000_cdt --drug_feat unimol --nfold 'split_cold_dose&time_random_1_one_shot,split_cold_dose&time_random_2_one_shot,split_cold_dose&time_random_3_one_shot,split_cold_dose&time_random_4_one_shot,split_cold_dose&time_random_5_one_shot'  --dataset l1000_mdmt --use_gradscaler True --include_cell_idx True --mode test --saved_model /root/data1/GY/gy/XPert/experiment_snn/pretrain_mdmt_new.pth
