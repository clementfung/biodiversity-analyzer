import numpy as np
import tensorflow as tf
import pdb

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from keras.utils import np_utils

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
	dense_out = Dense(n_classes, activation='softmax')(flatten)
	
	# Define the total model
	model = Model(input_layer, dense_out)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	if verbose:
		print(model.summary())

	# compile and return model
	return model

if __name__ == '__main__':
	
	### A minimal working example of a trained CNN on some fake data.

	n_classes = 20

	n_samples = 8248
	split_idx = 7000

	model = create_model(n_classes=n_classes)
	
	Xrgb = np.load('Xrgb_filtered.npy')
	yrgb = np.load('yrgb_filtered.npy')

	yrgb_cat = np_utils.to_categorical(yrgb)
	Xtrain = Xrgb[:split_idx]
	ytrain = yrgb_cat[:split_idx]

	Xtest = Xrgb[split_idx:]
	ytest = yrgb_cat[split_idx:]

	train_history = model.fit(Xtrain, ytrain, epochs=5)
	loss_obj = train_history.history['loss']
	np.save(f'history.npy', loss_obj)

	ypred = model.predict(Xtest)
	test_accuracy = np.mean(np.argmax(ypred, axis=1) == np.argmax(ytest, axis=1))
	print(f"test accuracy is {test_accuracy}")

	pdb.set_trace()

	# Sometimes train accuracy crashes OOM. Moved it to the end for now
	ytrain_pred = model.predict(Xtrain)
	train_accuracy = np.mean(np.argmax(ytrain_pred, axis=1) == np.argmax(ytrain, axis=1))
	print(f"train accuracy is {train_accuracy}")

	# Add breakpoint in case some checks are needed
	pdb.set_trace()

