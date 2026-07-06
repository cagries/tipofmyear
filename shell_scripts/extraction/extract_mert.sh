#!/bin/sh
python scripts/extract_mert.py \
    --manifest data/processed/jtd_group_cv_16/manifests/windows_10s_hop5s.csv \
    --output data/processed/jtd_group_cv_16/features/mert_10s/mert_v1_95m_layermean_concat.pt \
    --model-id m-a-p/MERT-v1-95M \
    --batch-size 4 \
    --device cuda

