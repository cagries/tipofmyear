for fold in 00 01 02 03; do
    python scripts/eval_knn_probe.py \
      --feature-cache data/processed/jtd_group_cv_16/features/mert_10s/mert_v1_95m_layermean_concat.pt \
      --fold-dir data/processed/jtd_group_cv_16/manifests/fold_${fold} \
      --classes-json data/processed/jtd_group_cv_16/classes.json \
      --results-dir results/final_mert_knn_10s_k5/fold_${fold} \
      --k 5 \
      --metric cosine \
      --weights distance
  done

python scripts/summarize_results.py \
    --results-root results/final_mert_knn_10s_k5 \
    --folds fold_00 fold_01 fold_02 fold_03 \
    --output results/final_mert_knn_10s_k5/summary.json \
    --markdown-output results/final_mert_knn_10s_k5/summary.md

