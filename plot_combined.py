import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
import os

import utils

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

def mixed_inputs():

	model_names = ['mixedinput', 'mixedlayer']
	plot_obj = np.zeros((2, 2))
	plot_i = 0

	for model_name in model_names:

		loss_obj = np.loadtxt(f'output-data/train-history-{model_name}.csv', delimiter=',')

		print(f'For model {model_name} Best train accuracy is {np.max(loss_obj[2])}')
		print(f'For model {model_name} Best val accuracy is {np.max(loss_obj[3])}')

		plot_obj[plot_i, 0] = np.max(loss_obj[2])
		plot_obj[plot_i, 1] = np.max(loss_obj[3])

		plot_i += 1

	fig, ax = plt.subplots(1, 1, figsize=(6, 5))

	width = 0.25

	ax.set_title(f'Combined CNN Performance', fontsize=24)
	ax.set_ylim([0, 1.05])
	ax.bar(np.arange(2) - width / 2, plot_obj[:, 0], width, label='Training')
	ax.bar(np.arange(2) + width / 2, plot_obj[:, 1], width, label='Validation')

	ax.set_axisbelow(True)
	ax.set_yticks(np.arange(0, 1.05, 0.1))
	ax.grid(True, which='major', axis='y', linestyle = '--')
	ax.legend(loc='best', fontsize=20)

	ax.set_xticks(np.arange(2))
	ax.set_xticklabels(['Mixed-input', 'Mixed-layer'], fontsize=16)
	ax.set_ylabel('Accuracy', fontsize=20)

	fig.tight_layout(rect=[0, 0, 1, 1])

	plt.savefig(f'plot-combined.pdf')

def final_cdf():

	rf_val_proba = np.load(f'RF7-valproba-simple.npy')
	rf_test_proba = np.load(f'RF7-testproba-simple.npy')
	rf_val = np.load(f'RF7-yval-simple.npy')
	rf_test = np.load(f'RF7-ytest-simple.npy')

	cnn_val_proba = np.load(f'CNN10-valproba-rgb.npy')
	cnn_test_proba = np.load(f'CNN10-testproba-rgb.npy')
	cnn_val_cat = np.load(f'CNN10-yval-rgb.npy')
	cnn_test_cat = np.load(f'CNN10-ytest-rgb.npy')
	cnn_val = np.argmax(cnn_val_cat, axis=1)
	cnn_test = np.argmax(cnn_test_cat, axis=1)

	rf_val_ranks = np.zeros(len(rf_val_proba))
	cnn_val_ranks = np.zeros(len(cnn_val_proba))
	rf_test_ranks = np.zeros(len(rf_test_proba))
	cnn_test_ranks = np.zeros(len(cnn_test_proba))

	for i in range(len(rf_val_proba)):
		rf_val_ranks[i] = utils.scores_to_rank(rf_val_proba[i], rf_val[i])
		cnn_val_ranks[i] = utils.scores_to_rank(cnn_val_proba[i], cnn_val[i])

	for i in range(len(rf_test_proba)):
		rf_test_ranks[i] = utils.scores_to_rank(rf_test_proba[i], rf_test[i])
		cnn_test_ranks[i] = utils.scores_to_rank(cnn_test_proba[i], cnn_test[i])

	n_classes = 10
	cdf_val_rf = utils.plot_cdf(rf_val_ranks, max_rank=n_classes)
	cdf_val_cnn = utils.plot_cdf(cnn_val_ranks, max_rank=n_classes)

	cdf_test_rf = utils.plot_cdf(rf_test_ranks, max_rank=n_classes)
	cdf_test_cnn = utils.plot_cdf(cnn_test_ranks, max_rank=n_classes)

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	ax.plot(np.arange(1, n_classes + 1), cdf_test_rf, lw=2, label='Random Forest')
	ax.plot(np.arange(1, n_classes + 1), cdf_test_cnn, lw=2, label='CNN')
	ax.plot(np.arange(1, n_classes+1), np.arange(1, n_classes+1) / n_classes, color='black', linestyle='--', lw=2, label='Baseline')

	ax.set_xticks(np.arange(n_classes + 1))
	ax.legend(fontsize=20)

	ax.set_ylim([0, 1.05])
	ax.set_yticks(np.arange(0, 1.05, 0.1))
	ax.set_axisbelow(True)
	ax.grid(True, which='major', axis='y', linestyle = '--')

	ax.set_ylabel('Top-K accuracy', fontsize=20)
	ax.set_xlabel('K (number of output classes)', fontsize=20)

	fig.tight_layout()
	plt.savefig(f'final_cdf.pdf')
	plt.close()

	print('RF')
	print(cdf_test_rf)
	
	print('CNN10')
	print(cdf_test_cnn)

	return

if __name__ == '__main__':
	
	parser = get_argparser()
	args = parser.parse_args()

	final_cdf()
