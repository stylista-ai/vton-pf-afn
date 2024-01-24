#!/bin/bash
python -u eval_PBAFN_viton.py --name=cloth-warp --resize_or_crop=none --batchSize=32 --gpu_ids=0 \
  --warp_checkpoint=checkpoints/warp_viton.pth --label_nc=13 --dataroot=/home/ext_rleifer_gmail_com/input \
  --fineSize=512 --unpaired
