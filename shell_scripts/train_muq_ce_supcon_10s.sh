#!/bin/sh
for fold in 00 01 02 03; do
    python scripts/train_supcon_projection.py \
      --feature-cache data/processed/jtd_group_cv_16/features/muq_10s/muq_large_msd_iter_layermean_concat.pt \
      --fold-dir data/processed/jtd_group_cv_16/manifests/fold_${fold} \
      --classes-json data/processed/jtd_group_cv_16/classes.json \
      --results-dir results/final_muq_ce_supcon_knn_10s/fold_${fold} \
      --epochs 100 \
      --learning-rate 1e-3 \
      --weight-decay 1e-4 \
      --hidden-dim 512 \
      --projection-dim 128 \
      --dropout 0.1 \
      --lambda-supcon 0.2 \
      --temperature 0.07 \
      --classes-per-batch 8 \
      --performances-per-class 2 \
      --windows-per-performance 2 \
      --k 5 \
      --metric cosine \
      --weights distance \
      --seed 1337 \
      --final-train \
      --device cuda \
      --wandb-project tipofmyear \
      --wandb-run-name final_muq_ce_supcon_10s_fold${fold}
  done

python scripts/summarize_results.py \
    --results-root results/final_muq_ce_supcon_knn_10s \
    --folds fold_00 fold_01 fold_02 fold_03 \
    --output results/final_muq_ce_supcon_knn_10s/summary.json \
    --markdown-output results/final_muq_ce_supcon_knn_10s/summary.md



