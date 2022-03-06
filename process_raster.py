import rasterio
import matplotlib.pyplot as plt

import pdb

def plot_raster(raster, name):

	plt.imshow(raster)
	plt.savefig(f'{name}.png')
	plt.close()

def load_raster(file):

	src = rasterio.open(file)
	return src.read(1)

if __name__ == '__main__':
	
	# Clement: Still mostly incomplete. It's not clear how useful the rasters will be....

	for i in range(1, 5):

		dataset = load_raster(f'bio_{i}/bio_{i}_FR.tif')
		plot_raster(dataset, f'dataset{i}')

