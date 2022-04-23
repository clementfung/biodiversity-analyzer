#!/usr/bin/env bash

set -euxo pipefail

layers=(1 2 3 4)
units=(4 8 16 32)
regs=('0.01' '0.05')

for ll in ${layers[@]}; do
	for uu in ${units[@]}; do
		for rr in ${regs[@]}; do
			
			python3 train_alti_model.py --gpus 0 --cnn_layers $ll --cnn_units $uu --cnn_reg $rr
			python3 train_ir_model.py --gpus 0 --cnn_layers $ll --cnn_units $uu --cnn_reg $rr
			python3 train_rgb_model.py --gpus 0 --cnn_layers $ll --cnn_units $uu --cnn_reg $rr
			# python3 train_cover_model.py --dnn_layers $ll --dnn_units $uu --dnn_reg $rr

		done
	done
done

