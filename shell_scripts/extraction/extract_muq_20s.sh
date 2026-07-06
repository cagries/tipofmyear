#!/bin/sh
python scripts/extract_muq.py \
    --manifest data/processed/jtd_group_cv_16/manifests_20s_hop10s/windows_20s_hop10s.csv \
    --output data/processed/jtd_group_cv_16/features/muq_20s_hop10s/muq_large_msd_iter_layermean_concat.pt \
    --batch-size 4 \
    --device cuda

