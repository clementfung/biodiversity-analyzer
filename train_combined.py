import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

def create_model(n_classes=10, rgb_units=16, rgb_layers=2, ir_units=16, ir_layers=2, kernel=5, reg_weight=0.1):
	""" Creates Keras CNN model.   """

	# retrieve params
	verbose = True

	#### RGB MODEL
	rgb_input_shape = (256, 256, 3)
	rgb_input_layer = Input(shape=rgb_input_shape)
	rgb_cnn_layer = Conv2D(filters=rgb_units, kernel_size=kernel, kernel_regularizer=regularizers.l2(reg_weight))(rgb_input_layer)
	rgb_cnn_layer= LeakyReLU(alpha=0.2)(rgb_cnn_layer)
	rgb_cnn_layer = BatchNormalization()(rgb_cnn_layer)

	for _ in range(rgb_layers - 1):
		rgb_cnn_layer = Conv2D(filters=rgb_units, kernel_size=kernel, kernel_regularizer=regularizers.l2(reg_weight))(rgb_cnn_layer)
		rgb_cnn_layer= LeakyReLU(alpha=0.2)(rgb_cnn_layer)
		rgb_cnn_layer = BatchNormalization()(rgb_cnn_layer)

	rgb_flatten = Flatten()(rgb_cnn_layer)
	rgb_dense = Dense(50, kernel_regularizer=regularizers.l2(reg_weight))(rgb_flatten)
	rgb_out = LeakyReLU(alpha=0.2)(rgb_dense)

	#### IR Model
	ir_input_shape = (256, 256, 1)
	ir_input_layer = Input(shape=ir_input_shape)
	ir_cnn_layer = Conv2D(filters=ir_units, kernel_size=kernel, kernel_regularizer=regularizers.l2(reg_weight))(ir_input_layer)
	ir_cnn_layer= LeakyReLU(alpha=0.2)(ir_cnn_layer)
	ir_cnn_layer = BatchNormalization()(ir_cnn_layer)

	for _ in range(ir_layers - 1):
		ir_cnn_layer = Conv2D(filters=ir_units, kernel_size=kernel, kernel_regularizer=regularizers.l2(reg_weight))(ir_cnn_layer)
		ir_cnn_layer= LeakyReLU(alpha=0.2)(ir_cnn_layer)
		ir_cnn_layer = BatchNormalization()(ir_cnn_layer)

	ir_flatten = Flatten()(ir_cnn_layer)
	ir_dense = Dense(50, kernel_regularizer=regularizers.l2(reg_weight))(ir_flatten)
	ir_out = LeakyReLU(alpha=0.2)(ir_dense)

	combined = concatenate([rgb_out, ir_out])
	total_out = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(reg_weight))(combined)

	# Define the total model
	model = Model(inputs=[rgb_input_layer, ir_input_layer], outputs=total_out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	if verbose:
		print(model.summary())
		print(f'regularizer: {reg_weight}')

	# compile and return model
	return model

# Generic data generator object for feeding data to fit_generator
def data_generator_mixed(X1, X2, y, bs):
	
	i = 0
	while True:
		i += bs

		# Restart from beginning
		if i + bs > len(X1):
			i = 0 

		X1_batch = X1[i:i+bs]
		X2_batch = X2[i:i+bs]
		y_batch = y[i:i+bs]

		yield ([X1_batch, X2_batch], y_batch)

def get_argparser():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_epochs", 
		default=25,
		type=int,
		help="How many epochs?")

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
	
	data_amount = 'large'

	parser = get_argparser()
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

	########################
	# Define CNN model
	########################

	n_epochs = args.n_epochs

	#model_name = f'CNN10-layers{layers}-units{units}-kernel{kernel}-reg{regularizer}-rgb'
	model = create_model(n_classes=n_classes, kernel=5, reg_weight=0.01,
		rgb_units=32, rgb_layers=3, ir_units=32, ir_layers=3)
	
	########################
	# Load and process data
	########################
	if top10:
		Xrgb = np.load('Xrgb_top10.npy')
		Xir = np.load('Xir_top10.npy')
		Xir = np.expand_dims(Xir, axis=3)

		yfull = np.load('y_top10.npy')
	else:
		Xrgb = np.load('Xrgb_top20.npy')
		Xir = np.load('Xir_top20.npy')
		Xir = np.expand_dims(Xir, axis=3)

		yfull = np.load('y_top20.npy')

	yfull_cat = to_categorical(yfull)

	if data_amount == 'small':
		
		Xtrain_rgb = Xrgb[0:500]
		Xtrain_ir = Xir[0:500]
		Xtest_rgb = Xrgb[1000:1100]
		Xtest_ir = Xir[1000:1100]
		
		ytrain = yfull_cat[0:500]	
		ytest = yfull_cat[1000:1100]

	else:

		shff_idx = np.random.permutation(Xrgb.shape[0])

		Xtrain_rgb = Xrgb[shff_idx[0:8000]]
		Xval_rgb = Xrgb[shff_idx[8000:10600]]
		Xtest_rgb = Xrgb[shff_idx[10600:]]

		Xtrain_ir = Xir[shff_idx[0:8000]]
		Xval_ir = Xir[shff_idx[8000:10600]]
		Xtest_ir = Xir[shff_idx[10600:]]

		ytrain = yfull_cat[shff_idx[0:8000]]
		yval = yfull_cat[shff_idx[8000:10600]]
		ytest = yfull_cat[shff_idx[10600:]]

	########################
	# Train the model
	########################
	batch_size = 32
	epoch_steps = len(Xtrain_rgb) // batch_size
	val_steps = len(Xval_rgb) // batch_size

	train_history = model.fit_generator(
		data_generator_mixed(Xtrain_rgb, Xtrain_ir, ytrain, batch_size), 
		validation_data=data_generator_mixed(Xval_rgb, Xval_ir, yval, batch_size),
		callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0,  min_delta=0, mode='auto', restore_best_weights=True)],
		steps_per_epoch=epoch_steps,
		validation_steps=val_steps,
		epochs=n_epochs)

	########################
	# Save and evaluate model
	########################
	save = True
	
	if save:

		model_name = 'combined'

		model.save(model_name+'.h5')
		print(f'Keras model saved to {model_name+".h5"}')

		loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss']])
		np.savetxt(f'train-history-{model_name}.csv', loss_obj, delimiter=',', fmt='%.5f')

		ypred_val = model.predict([Xval_rgb, Xval_ir])
		val_accuracy = np.mean(np.argmax(ypred_val, axis=1) == np.argmax(yval, axis=1))
		print(f"final val accuracy is {val_accuracy}")

		ypred = model.predict([Xtest_rgb, Xtest_ir])
		test_accuracy = np.mean(np.argmax(ypred, axis=1) == np.argmax(ytest, axis=1))
		print(f"final test accuracy is {test_accuracy}")

		fig, ax = plt.subplots(2, 1, figsize=(10, 8))
		
		ax[0].hist(np.argmax(ypred_val, axis=1))
		ax[1].hist(np.argmax(ypred, axis=1))
		
		if top10:
			ax[0].set_xticks([])
			ax[0].set_xticklabels([])
			ax[1].set_xticks(np.arange(10))
			ax[1].set_xticklabels(np.arange(10))
		else:
			ax[0].set_xticks([])
			ax[0].set_xticklabels([])
			ax[1].set_xticks(np.arange(20))
			ax[1].set_xticklabels(np.arange(20))

		plt.savefig(f'{model_name}-hist.pdf')

