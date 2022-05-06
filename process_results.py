import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
import os

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

def get_argparser():
	
	parser = argparse.ArgumentParser()

	### CNNs
	parser.add_argument("--units", 
		default=16,
		type=int,
		help="Number of units in hidden layers of the DNN")
	parser.add_argument("--layers", 
		default=2,
		type=int,
		help="Number of layers in the DNN")
	parser.add_argument("--reg", 
		default=0.1,
		type=float,
		help="Regularization weight of the DNN")

	parser.add_argument("--data_type", 
		default='rgb',
		type=str,
		help="Which data to evaluate on: (ir, rgb, cover)")

	return parser

if __name__ == '__main__':
	
	parser = get_argparser()
	args = parser.parse_args()
	data_type = args.data_type

	########################
	# Load and process data
	########################

	layers_arr = [1, 2, 3, 4]
	units_arr = [4, 8, 16, 32, 64]
	reg_arr = [0.001, 0.005, 0.01, 0.05, 0.1]

	plot_obj = np.zeros((len(layers_arr), len(units_arr), len(reg_arr)))

	for layers_idx in range(len(layers_arr)):
		for units_idx in range(len(units_arr)):
			for reg_idx in range(len(reg_arr)):

				layers = layers_arr[layers_idx]
				units = units_arr[units_idx]
				regularizer = reg_arr[reg_idx]
				kernel = 5

				model_name = f'CNN10-layers{layers}-units{units}-kernel{kernel}-reg{regularizer}-{data_type}'
				loss_obj = np.loadtxt(f'output-data/train-history-{model_name}.csv', delimiter=',')

				print(f'For model {model_name} Best val accuracy is {np.max(loss_obj[3])}')

				plot_obj[layers_idx, units_idx, reg_idx] = np.max(loss_obj[3])

	# Pick one regularization
	cut_plot_obj = plot_obj[:, :, 2]

	fig, ax = plt.subplots(1, 1, figsize=(10, 6))

	width = 0.15
	n_col = len(layers_arr)

	ax.set_title('Hyperparameter Search for IR models', fontsize=24)
	ax.set_ylim([0, 0.5])
	ax.bar(np.arange(n_col) - 2 * width, cut_plot_obj[:, 0], width, label='4 units/layer')
	ax.bar(np.arange(n_col) - 1 * width, cut_plot_obj[:, 1], width, label='8 units/layer')
	ax.bar(np.arange(n_col) + 0 * width, cut_plot_obj[:, 2], width, label='16 units/layer')
	ax.bar(np.arange(n_col) + 1 * width, cut_plot_obj[:, 3], width, label='32 units/layer')
	ax.bar(np.arange(n_col) + 2 * width, cut_plot_obj[:, 4], width, label='64 units/layer')

	ax.set_axisbelow(True)
	ax.grid(True, which='major', axis='y', linestyle = '--')

	ax.legend(loc='best', ncol=2, fontsize=24)

	ax.set_xticks(np.arange(n_col))
	ax.set_xticklabels(layers_arr, fontsize=16)
	ax.set_xlabel('Layers', fontsize=24)
	ax.set_ylabel('Validation Accuracy', fontsize=24)

	fig.tight_layout(rect=[0, 0, 1, 1])
	plt.savefig(f'plot-hyperparam-{data_type}.pdf')
