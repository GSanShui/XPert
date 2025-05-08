# dataset: l1000_sdst

# output predicted profile

# python train_xpert_snn.py --model XPert_SNN --config config_snn --drug_feat unimol --dataset l1000_sdst --use_gradscaler True\ 
#     --mode test --output_profile True --include_cell_idx True\
#     --nfold split_1,split_cold_cell_1,split_cold_drug_1 \
#     --saved_model_path /root/data1/GY/gy/XPert/experiment_snn/20250106_142402_sdst_five_folds_新
