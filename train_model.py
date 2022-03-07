import numpy as np
import tensorflow as tf
import pdb

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

def create_model(n_classes=10):
	""" Creates Keras CNN model.   """

	# retrieve params
	units = 32
	layers = 2
	activation = 'relu'
	kernel_size = 5
	verbose = True

	input_shape = (256, 256, 3)

	input_layer = Input(shape=input_shape)
	cnn_layer1 = Conv2D(filters=units, kernel_size=kernel_size, activation=activation)(input_layer)
	cnn_layer1 = BatchNormalization()(cnn_layer1)

	flatten = Flatten()(cnn_layer1)
	dense_out = Dense(n_classes)(flatten)
	
	# Define the total model
	model = Model(input_layer, dense_out)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

	if verbose:
		print(model.summary())

	# compile and return model
	return model

if __name__ == '__main__':
	
	### A minimal working example of a trained CNN on some fake data.

	batch_size = 32
	n_classes = 10

	model = create_model(n_classes=n_classes)
	Xrgb = np.load('Xrgb.npy')

	Xtrain = Xrgb[:batch_size]
	ytrain = np.floor(np.random.rand(batch_size) * n_classes)
	model.fit(Xtrain, ytrain, epochs=10)

