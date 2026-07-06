#!/bin/sh
python scripts/extract_mert.py \
    --manifest data/processed/jtd_group_cv_16/manifests_20s_hop10s/windows_20s_hop10s.csv \
    --output data/processed/jtd_group_cv_16/features/mert_20s_hop10s/mert_v1_95m_layermean_concat.pt \
    --model-id m-a-p/MERT-v1-95M \
    --batch-size 4 \
    --device cuda

