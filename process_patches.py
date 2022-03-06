import rasterio
import matplotlib.pyplot as plt
import numpy as np

import glob
import pdb

def int_to_id(index):

	if index >= 10:
		return str(index)
	elif index > 0:
		return f'0{index}'
	else:
		return f'00'

if __name__ == '__main__':

	channel_file_index = 0
	alti_file_index = 0

	# Clement: For now, just doing 10 subdirectories each in the first 10 directories
	n_files = 8300

	altitude_input = np.zeros((n_files, 256, 256))
	rgb_channel_input = np.zeros((n_files, 256, 256, 3))
	ir_channel_input = np.zeros((n_files, 256, 256))
	cover_input = np.zeros((n_files, 256, 256))

	for value in range(10):

		patches_dir = (value // 5 + 1)
		filepath  = f'patches_fr_{int_to_id(patches_dir)}/{int_to_id(value)}'

		for file_index in range(10):

			datafiles = glob.glob(f'{filepath}/{int_to_id(file_index)}/*.npy')

			# Clement: slightly hacky indexing here, but assuming we trust the sorting alg, it looks like it all lines up.
			for file in sorted(datafiles):

				print(f'Processing {file}')

				## Is an altitude file
				if file[-8:] == 'alti.npy':
					
					# Clement: Add any scaling here?
					altitude_input[alti_file_index] = np.load(file)
					alti_file_index += 1
					
				else:

					area_data = np.load(file)
					rgb_channel_input[channel_file_index] = area_data[:, :, 0:3] / 256
					ir_channel_input[channel_file_index] = area_data[:, :, 3] / 256
					cover_input[channel_file_index] = area_data[:, :, 4]
					channel_file_index += 1
				

	print(f'Processed {alti_file_index} altitude files')
	print(f'Processed {channel_file_index} RGB-IR files')

	np.save('Xrgb.npy', rgb_channel_input)
	np.save('Xir.npy', ir_channel_input)
	np.save('Xaltitude.npy', altitude_input)
	np.save('Xcover.npy', cover_input)

	pdb.set_trace()


