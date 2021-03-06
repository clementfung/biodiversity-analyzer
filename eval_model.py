import argparse
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
from sklearn.model_selection import train_test_split

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
		help="Which data to evaluate on: (ir, rgb, cover, alti)")

	return parser

def get_grid_results(data_type, Xtrain, ytrain, Xtest, ytest):

	train_results = np.zeros((4, 5, 5))
	test_results = np.zeros((4, 5, 5))

	layers_arr = [1, 2, 3, 4]
	units_arr = [4, 8, 16, 32, 64]
	reg_arr = [0.001, 0.005, 0.01, 0.05, 0.1]

	for layers_idx in range(len(layers_arr)):
		for units_idx in range(len(units_arr)):
			for reg_idx in range(len(reg_arr)):

				layers = layers_arr[layers_idx]
				units = units_arr[units_idx]
				reg = reg_arr[reg_idx]

				if data_type == 'cover':
					model_name = f'DNN-layers{layers}-units{units}-reg{reg}-{data_type}'
				else:
					model_name = f'CNN10-layers{layers}-units{units}-kernel5-reg{reg}-{data_type}'
				
				model = load_model(f'{model_name}.h5')

				########################
				# Evaluate model
				########################

				print(f'For {model_name}')

				ypred = model.predict(Xtest, batch_size=32)
				test_accuracy = np.mean(np.argmax(ypred, axis=1) == np.argmax(ytest, axis=1))
				print(f"test accuracy is {test_accuracy}")

				# Sometimes train accuracy crashes OOM. Moved it to the end for now
				ytrain_pred = model.predict(Xtrain, batch_size=32)
				train_accuracy = np.mean(np.argmax(ytrain_pred, axis=1) == np.argmax(ytrain, axis=1))
				print(f"train accuracy is {train_accuracy}")

				train_results[layers_idx, units_idx, reg_idx] = train_accuracy
				test_results[layers_idx, units_idx, reg_idx] = test_accuracy

	np.save(f'CNN10-train-accuracy-{data_type}.npy', train_results)
	np.save(f'CNN10-test-accuracy-{data_type}.npy', test_results)

def get_proba(data_type, Xval, yval, Xtest, ytest, layers=4, units=32, reg=0.1):

	model_name = f'CNN10-layers{layers}-units{units}-kernel5-reg{reg}-{data_type}'			
	model = load_model(f'{model_name}.h5')

	########################
	# Evaluate model
	########################

	print(f'For {model_name}')

	ytest_proba = model.predict(Xtest, batch_size=32)
	yval_proba = model.predict(Xval, batch_size=32)
	
	np.save(f'CNN10-valproba-{data_type}.npy', yval_proba)
	np.save(f'CNN10-testproba-{data_type}.npy', ytest_proba)
	np.save(f'CNN10-yval-{data_type}.npy', yval)
	np.save(f'CNN10-ytest-{data_type}.npy', ytest)

if __name__ == '__main__':

	parser = get_argparser()
	args = parser.parse_args()
	data_type = args.data_type

	os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

	########################
	# Load and process data
	########################
	
	Xrgb = np.load(f'X{data_type}_top10.npy')
	yrgb = np.load('y_top10.npy')
	yrgb_cat = to_categorical(yrgb)

	if data_type == 'ir' or data_type == 'alti':
		Xrgb = np.expand_dims(Xrgb, axis=3)

	# Performs a 60/20/20 split
	Xtrain1, Xtest, ytrain1, ytest = train_test_split(Xrgb, yrgb_cat, test_size=0.2, random_state=42)
	Xtrain, Xval, ytrain, yval = train_test_split(Xtrain1, ytrain1, test_size=0.25, random_state=42)

	get_proba('rgb', Xval, yval, Xtest, ytest)	
