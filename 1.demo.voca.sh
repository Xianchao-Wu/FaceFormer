#########################################################################
# File Name: 1.demo.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2023年02月19日 15時27分22秒
#########################################################################
#!/bin/bash

python -m ipdb demo.py \
	--model_name vocaset \
	--wav_path "demo/wav/test.wav" \
	--dataset vocaset \
	--vertice_dim 15069  \
	--feature_dim 64 \
	--period 30 \
	--fps 30 \
	--train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" \
	--test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" \
	--condition FaceTalk_170913_03279_TA \
	--subject FaceTalk_170809_00138_TA
