#########################################################################
# File Name: 1.demo.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2023年02月19日 15時27分22秒
#########################################################################
#!/bin/bash

python -m ipdb demo.py \
	--model_name biwi \
	--wav_path "demo/wav/test.wav" \
	--dataset BIWI \
	--vertice_dim 70110  \
	--feature_dim 128 \
	--period 25 \
	--fps 25 \
	--train_subjects "F2 F3 F4 M3 M4 M5" \
	--test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" \
	--condition M3 \
	--subject M1
