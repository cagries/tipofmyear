#!/bin/sh
python3 scripts/extract_muq.py \
    --manifest data/processed/jtd_group_cv_16/manifests/windows_10s_hop5s.csv \
    --output data/processed/jtd_group_cv_16/features/muq_10s/muq_large_msd_iter_layermean_concat.pt \
    --model-id OpenMuQ/MuQ-large-msd-iter \
    --batch-size 8 \
    --device cuda

