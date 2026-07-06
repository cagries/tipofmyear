#!/bin/sh
python scripts/summarize_results.py \
    --results-root results/final_mert_ce_supcon_knn_10s \
    --folds fold_00 fold_01 fold_02 fold_03 \
    --metrics-path group_overlap_test/group_overlap_metrics.json \
    --metric-prefix metrics \
    --metric-keys same_group_neighbor_rate_at_k query_windows_with_any_same_group_neighbor_at_k \
    --output results/final_mert_ce_supcon_knn_10s/group_overlap_summary.json \
    --markdown-output results/final_mert_ce_supcon_knn_10s/group_overlap_summary.md

