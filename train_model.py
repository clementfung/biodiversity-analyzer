import numpy as np
import tensorflow as tf
import pdb

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
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
	cnn_layer1 = Conv2D(filters=units, kernel_size=kernel_size, activation=activation, 
		kernel_regularizer=regularizers.l2(1))(input_layer)
	cnn_layer1 = BatchNormalization()(cnn_layer1)
	
	cnn_layer2 = Conv2D(filters=units, kernel_size=kernel_size, activation=activation, 
		kernel_regularizer=regularizers.l2(1))(cnn_layer1)
	cnn_layer2 = BatchNormalization()(cnn_layer2)

	cnn_layer3 = Conv2D(filters=units, kernel_size=kernel_size, activation=activation, 
		kernel_regularizer=regularizers.l2(1))(cnn_layer2)
	cnn_layer3 = BatchNormalization()(cnn_layer3)

	flatten = Flatten()(cnn_layer3)
	dense_out = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(1))(flatten)
	
	# Define the total model
	model = Model(input_layer, dense_out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	if verbose:
		print(model.summary())

	# compile and return model
	return model

if __name__ == '__main__':
	
	### A minimal working example of a trained CNN on some fake data.

	n_classes = 20

	n_samples = 14363
	split_idx = 11000
	local = True

	model = create_model(n_classes=n_classes)
	
	Xrgb = np.load('Xrgb_filtered.npy')
	yrgb = np.load('yrgb_filtered.npy')

	# Replace labels with 0-19
	label_mapping = np.unique(yrgb).astype(int)
	yrgb_20 = np.zeros_like(yrgb)
	for i in range(len(label_mapping)):
		old_label = label_mapping[i]
		yrgb_20[np.where(yrgb == old_label)] = i

	yrgb_cat = np_utils.to_categorical(yrgb_20)
	
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

	train_history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=5)
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

