import argparse
import numpy as np
import tensorflow as tf
import pdb

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from keras.utils import np_utils

def create_model(n_classes=10, n_units=16, n_layers=2, kernel=5, reg_weight=0.1):
	""" Creates Keras CNN model.   """

	# retrieve params
	activation = 'relu'
	verbose = True

	input_shape = (256, 256, 1)

	input_layer = Input(shape=input_shape)
	cnn_layer = Conv2D(filters=n_units, kernel_size=kernel, activation=activation, 
		kernel_regularizer=regularizers.l2(reg_weight))(input_layer)
	cnn_layer = BatchNormalization()(cnn_layer)

	for _ in range(n_layers - 1):
		cnn_layer = Conv2D(filters=n_units, kernel_size=kernel, activation=activation, 
			kernel_regularizer=regularizers.l2(reg_weight))(cnn_layer)
		cnn_layer = BatchNormalization()(cnn_layer)

	flatten = Flatten()(cnn_layer)
	dense_out = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(reg_weight))(flatten)
	
	# Define the total model
	model = Model(input_layer, dense_out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	if verbose:
		print(model.summary())

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
	parser.add_argument("--cnn_units", 
		default=16,
		type=int,
		help="Number of units in hidden layers of the CNN")
	parser.add_argument("--cnn_layers", 
		default=2,
		type=int,
		help="Number of layers in the CNN")
	parser.add_argument("--cnn_kernel", 
		default=5,
		type=int,
		help="Kernel Size of the CNN")
	parser.add_argument("--cnn_reg", 
		default=0.1,
		type=float,
		help="Regularization weight of the CNN")

	return parser

if __name__ == '__main__':
	
	n_classes = 20
	n_samples = 14363
	split_idx = 11000
	local = False

	parser = get_argparser()
	args = parser.parse_args()

	########################
	# Define CNN model
	########################
	units = args.cnn_units
	layers = args.cnn_layers
	kernel = args.cnn_kernel
	regularizer = args.cnn_reg

	model_name = f'CNN-layers{layers}-units{units}-kernel{kernel}-reg{regularizer}-ir'
	model = create_model(n_classes=n_classes, n_units=units, n_layers=layers, kernel=kernel, reg_weight=regularizer)
	
	########################
	# Load and process data
	########################
	Xrgb = np.load('Xir_top20.npy')
	yrgb = np.load('y_top20.npy')

	yrgb_cat = np_utils.to_categorical(yrgb)
	
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
		epochs=50)

	########################
	# Save and evaluate model
	########################
	model.save(model_name+'.h5')
	print(f'Keras model saved to {model_name+".h5"}')

	loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss']])
	np.savetxt(f'train-history-{model_name}.csv', loss_obj, delimiter=',', fmt='%.5f')

	ypred = model.predict(Xtest)
	test_accuracy = np.mean(np.argmax(ypred, axis=1) == np.argmax(ytest, axis=1))
	print(f"test accuracy is {test_accuracy}")

	# Sometimes train accuracy crashes OOM. Moved it to the end for now
	ytrain_pred = model.predict(Xtrain)
	train_accuracy = np.mean(np.argmax(ytrain_pred, axis=1) == np.argmax(ytrain, axis=1))
	print(f"train accuracy is {train_accuracy}")
