#!/bin/bash
rm -rf /root/autodl-tmp/keyword_detect/output/dual_at_v4_text/
mkdir -p /root/autodl-tmp/keyword_detect/output/dual_at_v4_text/
cd /root/autodl-tmp/keyword_detect
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=baseline:. nohup python3 -u baseline/train_dual.py --name at_v4 --mode text --model-version 1 --unfreeze 2 --epochs 20 --bs 128 --lr 3e-4 --text-encoder char > output/dual_at_v4_text/train.log 2>&1 &
echo "PID=$!" | tee /root/autodl-tmp/keyword_detect/at_v4_pid.txt
