import argparse
import numpy as np
import tensorflow as tf
import pdb
import os

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

def create_model(n_classes=10, n_units=16, n_layers=2, reg_weight=0.1):
	""" Creates Keras CNN model.   """

	# retrieve params
	activation = 'relu'
	verbose = True

	input_shape = (17)

	input_layer = Input(shape=input_shape)	
	dnn_layer = Dense(n_units, kernel_regularizer=regularizers.l2(reg_weight))(input_layer)
	dnn_layer = LeakyReLU(alpha=0.2)(dnn_layer)

	for _ in range(n_layers - 1):
		dnn_layer = Dense(n_units, kernel_regularizer=regularizers.l2(reg_weight))(dnn_layer)
		dnn_layer = LeakyReLU(alpha=0.2)(dnn_layer)

	flatten = Flatten()(dnn_layer)
	dense_out = Dense(100, kernel_regularizer=regularizers.l2(reg_weight))(flatten)
	dense_out = LeakyReLU(alpha=0.2)(dense_out)
	dense_out = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(reg_weight))(dense_out)
	
	# Define the total model
	model = Model(input_layer, dense_out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	if verbose:
		print(model.summary())
		print(f'regularizer: {reg_weight}')

	# compile and return model
	return model

# Generic data generator object for feeding data to fit_generator
def data_generator(X, y, bs):
	
	i = 0
	while True:
		i += bs

		# Restart from beginning
		if i + bs > len(X):
			i = 0 

		X_batch = X[i:i+bs]
		y_batch = y[i:i+bs]
		yield (X_batch, y_batch)

def get_argparser():
	
	parser = argparse.ArgumentParser()

	### CNNs
	parser.add_argument("--dnn_units", 
		default=16,
		type=int,
		help="Number of units in hidden layers of the DNN")
	parser.add_argument("--dnn_layers", 
		default=2,
		type=int,
		help="Number of layers in the DNN")
	parser.add_argument("--dnn_reg", 
		default=0.1,
		type=float,
		help="Regularization weight of the DNN")

	parser.add_argument("--gpus", 
		default=-1,
		type=int,
		help="Which GPUs?")

	return parser

if __name__ == '__main__':
	
	n_classes = 10
	n_samples = 14363
	split_idx = 11000
	n_epochs = 25

	local = False

	parser = get_argparser()
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

	########################
	# Define CNN model
	########################
	units = args.dnn_units
	layers = args.dnn_layers
	regularizer = args.dnn_reg

	model_name = f'DNN-layers{layers}-units{units}-reg{regularizer}-cover'
	model = create_model(n_classes=n_classes, n_units=units, n_layers=layers, reg_weight=regularizer)
	
	########################
	# Load and process data
	########################
	Xrgb = np.load('Xcoverage_top10.npy')
	yrgb = np.load('y_top10.npy')
	yrgb_cat = to_categorical(yrgb)

	Xrgb = Xrgb[:, 17:]

	if local:
		Xtrain = Xrgb[0:500]
		ytrain = yrgb_cat[0:500]
		Xtest = Xrgb[1000:1100]
		ytest = yrgb_cat[1000:1100]
	else:
		Xtrain = Xrgb[:split_idx]
		ytrain = yrgb_cat[:split_idx]
		Xtest = Xrgb[split_idx:]
		ytest = yrgb_cat[split_idx:]

	########################
	# Train the model
	########################
	batch_size = 32
	epoch_steps = len(Xtrain) // batch_size
	val_steps = len(Xtest) // batch_size

	train_history = model.fit_generator(
		data_generator(Xtrain, ytrain, batch_size), 
		steps_per_epoch=epoch_steps,
				validation_steps=val_steps,
		validation_data=data_generator(Xtest, ytest, batch_size), 
		epochs=n_epochs)

	########################
	# Save and evaluate model
	########################
	model.save(model_name+'.h5')
	print(f'Keras model saved to {model_name+".h5"}')

	loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss'], train_history.history['acc'], train_history.history['val_acc']])
	np.savetxt(f'train-history-{model_name}.csv', loss_obj, delimiter=',', fmt='%.5f')

	ypred = model.predict(Xtest)
	test_accuracy = np.mean(np.argmax(ypred, axis=1) == np.argmax(ytest, axis=1))
	print(f"test accuracy is {test_accuracy}")

	# Sometimes train accuracy crashes OOM. Moved it to the end for now
	ytrain_pred = model.predict(Xtrain)
	train_accuracy = np.mean(np.argmax(ytrain_pred, axis=1) == np.argmax(ytrain, axis=1))
	print(f"train accuracy is {train_accuracy}")
