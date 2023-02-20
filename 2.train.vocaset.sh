#########################################################################
# File Name: 2.train.vocaset.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Feb 20 11:22:29 2023
#########################################################################
#!/bin/bash

python -m ipdb main.py \
    --dataset vocaset \
    --vertice_dim 15069 \
    --feature_dim 64 \
    --period 30 \
    --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
    --val_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" \
    --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"
