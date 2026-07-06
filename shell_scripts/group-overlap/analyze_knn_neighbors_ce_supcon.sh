#!/bin/sh
for fold in 00 01 02 03; do
  python scripts/analyze_knn_group_overlap.py \
    --result-dir results/final_mert_ce_supcon_knn_10s/fold_${fold} \
    --query-split test \
    --use-existing-neighbors \
    --k 5 \
    --output-dir results/final_mert_ce_supcon_knn_10s/fold_${fold}/group_overlap_test
done
