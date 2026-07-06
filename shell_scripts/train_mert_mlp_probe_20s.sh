#!/bin/sh
for fold in 00 01 02 03; do
    python scripts/train_linear_probe.py \
      --feature-cache data/processed/jtd_group_cv_16/features/mert_20s_hop10s/mert_v1_95m_layermean_concat.pt \
      --fold-dir data/processed/jtd_group_cv_16/manifests_20s_hop10s/fold_${fold} \
      --classes-json data/processed/jtd_group_cv_16/classes.json \
      --results-dir results/final_mert_mlp_20s_no_pca/fold_${fold} \
      --head mlp \
      --hidden-dim 256 \
      --dropout 0.5 \
      --epochs 100 \
      --batch-size 64 \
      --learning-rate 1e-3 \
      --weight-decay 0.01 \
      --seed 1337 \
      --final-train \
      --device cuda \
      --wandb-project tipofmyear \
      --wandb-run-name final_mert_mlp_20s_fold${fold}
  done

python scripts/summarize_results.py \
    --results-root results/final_mert_mlp_20s_no_pca \
    --folds fold_00 fold_01 fold_02 fold_03 \
    --output results/final_mert_mlp_20s_no_pca/summary.json \
    --markdown-output results/final_mert_mlp_20s_no_pca/summary.md



