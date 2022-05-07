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

	return parser

def concat_full():

	n_classes = 10
	n_samples = 13240

	########################
	# Load and process data
	########################

	Xrgb = np.load(f'Xrgb_top{n_classes}.npy')
	Xir = np.load(f'Xir_top{n_classes}.npy')
	Xalti = np.load(f'Xalti_top{n_classes}.npy')

	#########################
	# Custom processing
	#########################

	# Add scaling on Xalti: scale to 0-1 range.	
	Xalti = Xalti + np.abs(np.min(Xalti))
	Xalti = Xalti / np.abs(np.max(Xalti))

	Xfull = np.zeros((n_samples, 256, 256, 5))
	Xfull[:,:,:, 0:3] = Xrgb
	Xfull[:,:,:, 3] = Xir
	Xfull[:,:,:, 4] = Xalti

	return Xfull

def concat_simple():

	n_classes = 10
	n_samples = 13240	

	Xir = np.load(f'Xir_top{n_classes}.npy')
	Xalti = np.load(f'Xalti_top{n_classes}.npy')
	Xcover = np.load(f'Xcoverage_top{n_classes}.npy')		

	Xdata = np.zeros((n_samples, 19))

	# First feature: average altitude

	# Add scaling on Xalti: scale to 0-1 range.	
	Xalti = Xalti + np.abs(np.min(Xalti))
	Xalti = Xalti / np.abs(np.max(Xalti))
	Xdata[:, 0] = np.mean(Xalti, axis=(1, 2))

	# Second feature: average Ir
	Xdata[:, 1] = np.mean(Xir, axis=(1, 2))

	# Filter out coverages for USA
	cover_column_idx = np.array([0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
	Xdata[:, 2:] = Xcover[:, cover_column_idx]

	np.save('Xsimple.npy', Xdata)

if __name__ == '__main__':
	
	concat_simple()
