#!/usr/bin/env bash

set -euxo pipefail

layers=(1 2 3 4)
units=(4 8)
regs=('0.01' '0.05' '0.1' '0.5' '1')

for ll in ${layers[@]}; do
	for uu in ${units[@]}; do
		for rr in ${regs[@]}; do
			
			python3 train_ir_model.py --cnn_layers $ll --cnn_units $uu --cnn_reg $rr
			python3 train_rgb_model.py --cnn_layers $ll --cnn_units $uu --cnn_reg $rr
			python3 train_cover_model.py --dnn_layers $ll --dnn_units $uu --dnn_reg $rr

		done
	done
done

