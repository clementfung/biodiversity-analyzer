import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
import os
import utils

import sklearn.neighbors
import sklearn.ensemble

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

def get_argparser():
	
	parser = argparse.ArgumentParser()

	### CNNs
	parser.add_argument("--model_type", 
		type=str,
		help="Type of Model")

	parser.add_argument("--data_type", 
		default='rgb',
		type=str,
		help="Which data to evaluate on: (ir, rgb, cover, alti)")

	return parser

if __name__ == '__main__':
	
	top10 = True
	
	if top10:
		n_classes = 10
		n_samples = 13240
	else:
		n_classes = 20
		n_samples = 14363
	
	data_amount = 'medium'

	parser = get_argparser()
	args = parser.parse_args()
	model_type = args.model_type
	data_type = args.data_type

	########################
	# Load and process data
	########################

	Xdata = np.load(f'X{data_type}_top{n_classes}.npy')		
	ydata = np.load(f'y_top{n_classes}.npy')
	ydata_cat = ydata

	#########################
	# Custom processing
	#########################

	# Add scaling on Xalti: scale to 0-1 range.
	if data_type == 'alti':
		Xdata = Xdata + np.abs(np.min(Xdata))
		Xdata = Xdata / np.abs(np.max(Xdata))

	if data_type == 'coverage':
		
		# Filter out coverages for USA
		column_idx = np.array([0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
		Xdata = Xdata[:, column_idx]

	#########################
	### Segment on n-axis
	#########################
	if data_amount == 'small':
		
		Xtrain = Xdata[0:100]
		ytrain = ydata_cat[0:100]
		
		Xval = Xdata[1000:1100]
		yval = ydata_cat[1000:1100]

		Xtest = Xdata[2000:2100]
		ytest = ydata_cat[2000:2100]

	else:

		# Performs a 60/20/20 split
		Xtrain1, Xtest, ytrain1, ytest = train_test_split(Xdata, ydata_cat, test_size=0.2, random_state=42)
		Xtrain, Xval, ytrain, yval = train_test_split(Xtrain1, ytrain1, test_size=0.25, random_state=42)

	if data_type == 'alti' or data_type == 'rgb' or data_type == 'ir':
		Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
		Xval = Xval.reshape(Xval.shape[0], -1)
		Xtest = Xtest.reshape(Xtest.shape[0], -1)

	########################
	# Define model
	########################
	
	models = []

	if model_type == 'KNN':
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=1))
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=3))
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=5))
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=7))
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=9))
		models.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=11))
		model_name = 'KNN'
	elif model_type == 'RF':
		model_name = 'RF'
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=3, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=7, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=9, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=11, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=13, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=15, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=17, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=19, n_estimators=200))
		models.append(sklearn.ensemble.RandomForestClassifier(max_depth=21, n_estimators=200))
	else:
		print(f'Unknown model {model_type}')
		exit(0)

	all_train_ranks = np.zeros((len(models), n_classes))
	all_val_ranks = np.zeros((len(models), n_classes))
	all_test_ranks = np.zeros((len(models), n_classes))

	for model_idx in range(len(models)):

		#########################
		# Train
		#########################
		
		model = models[model_idx]
		model.fit(Xtrain, ytrain)

		########################
		# Save and evaluate model
		########################
		ypred_train = model.predict(Xtrain)
		ypred_val = model.predict(Xval)
		ypred = model.predict(Xtest)

		########################
		# Evaluate: accuracy
		########################
		train_accuracy = np.mean(ypred_train == ytrain)
		print(f"final train accuracy is {train_accuracy}")

		val_accuracy = np.mean(ypred_val== yval)
		print(f"final val accuracy is {val_accuracy}")
		
		test_accuracy = np.mean(ypred == ytest)
		print(f"final test accuracy is {test_accuracy}")

		########################
		# Evaluate: top-K rank
		########################
		ytrain_proba = model.predict_proba(Xtrain)
		ytrain_ranks = np.zeros(len(ytrain_proba))

		yval_proba = model.predict_proba(Xval)
		yval_ranks = np.zeros(len(yval_proba))

		ytest_proba = model.predict_proba(Xtest)
		ytest_ranks = np.zeros(len(ytest_proba))

		for i in range(len(ytrain_proba)):
			ytrain_ranks[i] = utils.scores_to_rank(ytrain_proba[i], ytrain[i])

		for i in range(len(yval_proba)):
			yval_ranks[i] = utils.scores_to_rank(yval_proba[i], yval[i])

		for i in range(len(ytest_proba)):
			ytest_ranks[i] = utils.scores_to_rank(ytest_proba[i], ytest[i])

		# np.save(f'{model_name}{model_idx}-valproba-{data_type}.npy', yval_proba)
		# np.save(f'{model_name}{model_idx}-testproba-{data_type}.npy', ytest_proba)
		# np.save(f'{model_name}{model_idx}-yval-{data_type}.npy', yval)
		# np.save(f'{model_name}{model_idx}-ytest-{data_type}.npy', ytest)

		cdf_obj = utils.plot_cdf(ytrain_ranks, max_rank=n_classes)
		all_train_ranks[model_idx] = cdf_obj

		cdf_obj = utils.plot_cdf(yval_ranks, max_rank=n_classes)
		all_val_ranks[model_idx] = cdf_obj

		cdf_obj = utils.plot_cdf(ytest_ranks, max_rank=n_classes)
		all_test_ranks[model_idx] = cdf_obj

	####################################
	## Tr/Val/Test plots
	####################################

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	if model_type == 'RF':
		ax.set_title('Random Forest Top-1 Accuracy', fontsize=20)
		ax.plot(np.arange(3, 22, 2), all_train_ranks[:, 0], lw=2, label='Train')
		ax.plot(np.arange(3, 22, 2), all_val_ranks[:, 0], lw=2, label='Validation')
		ax.set_xticks(np.arange(3, 22, 2))
		ax.set_xticklabels(np.arange(3, 22, 2))
		ax.set_xlabel('Max Depth', fontsize=16)
	elif model_type == 'KNN':
		ax.set_title('KNN Top-1 Accuracy', fontsize=20)
		ax.plot(np.arange(1, 12, 2), all_train_ranks[:, 0], lw=2, label='Train')
		ax.plot(np.arange(1, 12, 2), all_val_ranks[:, 0], lw=2, label='Validation')
		ax.set_xticks(np.arange(1, 12, 2))
		ax.set_xticklabels(np.arange(1, 12, 2))
		ax.set_xlabel('# of Nearest Neighbors', fontsize=16)

	ax.set_ylim([0, 1.05])
	ax.set_yticks(np.arange(0, 1.05, 0.1))

	ax.set_axisbelow(True)
	ax.grid(True, which='major', axis='y', linestyle = '--')

	ax.legend(fontsize=20, loc='best')

	fig.tight_layout()
	plt.savefig(f'{model_name}_splits.pdf')
	plt.close()

	####################################
	## CDFs
	####################################
	
	max_rank = n_classes

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))

	ax.plot(np.arange(1, max_rank + 1), cdf_obj, lw=2)
	ax.plot(np.arange(max_rank+1), np.arange(max_rank+1) / max_rank, color='black', linestyle='--', lw=2)

	ax.set_xticks(np.arange(max_rank + 1))

	fig.tight_layout()
	plt.savefig(f'{model_name}_{data_type}_cdf.pdf')
	plt.close()
