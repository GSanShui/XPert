
# To ouput cls_embed
# python train_xpert.py --model XPert --config config --drug_feat unimol --nfold split_1 --dataset l1000_sdst --use_gradscaler True --mode infer --include_cell_idx True --output_cls_embed True --saved_model 'saved_model/l1000_sdst_warm_split.pth'


# To output attention_score
# python train_xpert.py --model XPert --config config --drug_feat unimol --nfold split_1 --dataset l1000_sdst --use_gradscaler True --mode infer --include_cell_idx True --output_attention True --saved_model 'saved_model/l1000_sdst_warm_split.pth'

