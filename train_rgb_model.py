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

def create_model(n_classes=10, n_units=16, n_layers=2, kernel=5, reg_weight=0.1):
	""" Creates Keras CNN model.   """

	# retrieve params
	activation = 'relu'
	verbose = True

	input_shape = (256, 256, 3)

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

	parser.add_argument("--gpus", 
		default=-1,
		type=int,
		help="Which GPUs?")

	return parser

if __name__ == '__main__':
	
	top10 = True
	
	if top10:
		n_classes = 10
		n_samples = 13240
	else:
		n_classes = 20
		n_samples = 14363
	
	n_epochs = 25

	data_amount = 'medium'

	parser = get_argparser()
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

	########################
	# Define CNN model
	########################
	units = args.cnn_units
	layers = args.cnn_layers
	kernel = args.cnn_kernel
	regularizer = args.cnn_reg

	model_name = f'CNN10-layers{layers}-units{units}-kernel{kernel}-reg{regularizer}-rgb'
	model = create_model(n_classes=n_classes, n_units=units, n_layers=layers, kernel=kernel, reg_weight=regularizer)
	
	########################
	# Load and process data
	########################
	if top10:
		Xrgb = np.load('Xrgb_top10.npy')
		yrgb = np.load('y_top10.npy')
	else:
		Xrgb = np.load('Xrgb_top20.npy')
		yrgb = np.load('y_top20.npy')

	yrgb_cat = to_categorical(yrgb)
	
	if data_amount == 'full':
		split_idx = 11000
		
		Xtrain = Xrgb[:split_idx]
		ytrain = yrgb_cat[:split_idx]
		
		Xtest = Xrgb[split_idx:]
		ytest = yrgb_cat[split_idx:]

	elif data_amount == 'medium':
		
		Xtrain = Xrgb[:7000]
		ytrain = yrgb_cat[:7000]

		Xtest = Xrgb[7000:10000]
		ytest = yrgb_cat[7000:10000]

	elif data_amount == 'small':
		
		Xtrain = Xrgb[0:500]
		ytrain = yrgb_cat[0:500]
		
		Xtest = Xrgb[1000:1100]
		ytest = yrgb_cat[1000:1100]

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
	save = True
	
	if save:

		model.save(model_name+'.h5')
		print(f'Keras model saved to {model_name+".h5"}')

		loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss']])
		np.savetxt(f'train-history-{model_name}.csv', loss_obj, delimiter=',', fmt='%.5f')

		fig, ax = plt.subplots(1, 1, figsize=(10, 8))
		ax.hist(np.argmax(ypred, axis=1))
		
		if top10:
			ax.set_xticks(np.arange(10))
			ax.set_xticklabels(np.arange(10))
		else:
			ax.set_xticks(np.arange(20))
			ax.set_xticklabels(np.arange(20))

		plt.savefig(f'{model_name}-hist.pdf')

