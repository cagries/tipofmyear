for fold in 00 01 02 03; do
    python scripts/train_hcnn.py \
      --fold-dir data/processed/jtd_group_cv_16/manifests/fold_${fold} \
      --classes-json data/processed/jtd_group_cv_16/classes.json \
      --results-dir results/final_hcnn_10s \
      --epochs 20 \
      --batch-size 16 \
      --learning-rate 1e-3 \
      --seed 1337 \
      --final-train \
      --device cuda \
      --wandb-project tipofmyear \
      --wandb-run-name final_hcnn_10s_fold${fold}
  done

python scripts/summarize_results.py \
    --results-root results/final_hcnn_10s \
    --folds fold_00 fold_01 fold_02 fold_03 \
    --output results/final_hcnn_10s/summary.json \
    --markdown-output results/final_hcnn_10s/summary.md



