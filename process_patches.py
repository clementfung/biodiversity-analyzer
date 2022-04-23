import rasterio
import matplotlib.pyplot as plt
import numpy as np

import json
import glob
import pdb

def int_to_id(index):

	if index >= 10:
		return str(index)
	elif index > 0:
		return f'0{index}'
	else:
		return f'00'

def process_demo():

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

def count_eligible_files(dir_limit=20):

	channel_file_index = 0

	for value in range(dir_limit):

		patches_dir = (value // 5 + 1)
		filepath  = f'patches/patches_us_{int_to_id(patches_dir)}/{int_to_id(value)}'

		datafiles = glob.glob(f'{filepath}/*/*.npy')

		# Clement: slightly hacky indexing here, but assuming we trust the sorting alg, it looks like it all lines up.
		for file in sorted(datafiles):

			if file[-8:] == 'alti.npy':
				continue

			print(f'Processing {file}')
			channel_file_index += 1

	print(f'Processed {channel_file_index} RGB-IR files')
	return channel_file_index

def process_filtered_rgb(nfiles, dir_limit=20):

	channel_file_index = 0

	# Clement: Limit data to k subdirectories 
	rgb_channel_input = np.zeros((nfiles, 256, 256, 3))

	for value in range(dir_limit):

		patches_dir = (value // 5 + 1)
		filepath  = f'patches/patches_us_{int_to_id(patches_dir)}/{int_to_id(value)}'

		datafiles = glob.glob(f'{filepath}/*/*.npy')

		# Clement: slightly hacky indexing here, but assuming we trust the sorting alg, it looks like it all lines up.
		for file in sorted(datafiles):

			if file[-8:] == 'alti.npy':
				continue

			print(f'Processing {file}')

			area_data = np.load(file)
			rgb_channel_input[channel_file_index] = area_data[:, :, 0:3] / 256
			channel_file_index += 1

	print(f'Processed {channel_file_index} RGB-IR files')
	np.save('Xrgb_top10.npy', rgb_channel_input)

def process_filtered_ir_coverage(nfiles, dir_limit=20):

	channel_file_index = 0

	# Clement: Limit data to k subdirectories 
	altitude_input = np.zeros((nfiles, 256, 256))
	ir_channel_input = np.zeros((nfiles, 256, 256))
	coverage_input = np.zeros((nfiles, 256, 256))

	for value in range(dir_limit):

		patches_dir = (value // 5 + 1)
		filepath  = f'patches/patches_us_{int_to_id(patches_dir)}/{int_to_id(value)}'

		datafiles = glob.glob(f'{filepath}/*/*.npy')

		# Clement: slightly hacky indexing here, but assuming we trust the sorting alg, it looks like it all lines up.
		for file in sorted(datafiles):

			if file[-8:] == 'alti.npy':
				continue

			print(f'Processing {file}')

			area_data = np.load(file)
			ir_channel_input[channel_file_index] = area_data[:, :, 3] / 256
			coverage_input[channel_file_index] = area_data[:, :, 4]

			channel_file_index += 1

	print(f'Processed {channel_file_index} IR-coverage files')
	np.save('Xir_top10.npy', ir_channel_input)
	np.save('Xcoverage_top10.npy', summarize_coverage(coverage_input))

def process_filtered_alti_coverage(nfiles, dir_limit=20):

	alti_file_index = 0

	# Clement: Limit data to k subdirectories 
	altitude_input = np.zeros((nfiles, 256, 256))

	for value in range(dir_limit):

		patches_dir = (value // 5 + 1)
		filepath  = f'patches/patches_us_{int_to_id(patches_dir)}/{int_to_id(value)}'

		datafiles = glob.glob(f'{filepath}/*/*.npy')

		# Clement: slightly hacky indexing here, but assuming we trust the sorting alg, it looks like it all lines up.
		for file in sorted(datafiles):

			if file[-8:] == 'alti.npy':
					
				print(f'Processing {file}')

				# Clement: Add any scaling here?
				altitude_input[alti_file_index] = np.load(file)
				alti_file_index += 1

	print(f'Processed {alti_file_index} altitude files')
	np.save('Xalti_top10.npy', altitude_input)

def summarize_coverage(coverage_input):

	num_coverage_categories = 34
	n_files = len(coverage_input)
	coverage_out = np.zeros((n_files, num_coverage_categories))

	for i in range(n_files):
		for j in range(num_coverage_categories):
			coverage_out[i, j] = np.mean(coverage_input[i] == j)

	return coverage_out

def process_filtered_labels(nfiles, dir_limit=20):

	y_idx = 0
	y_obj = np.zeros(nfiles)
	big_obj = dict()

	with open(f'annotations_train_top10_us_parsed.json', "r") as f:
		
		x = json.load(f)

		for item in x:
			if int(item["file_name"][:2]) < dir_limit:
				big_obj[item["file_name"]] = item["category"]

	for key in sorted(big_obj.keys()):
		y_obj[y_idx] = big_obj[key]
		y_idx += 1

	print(f'Processed {y_idx} labels')

	# Replace labels with 0-19
	label_mapping = np.unique(y_obj).astype(int)
	y_topk = np.zeros_like(y_obj)
	for i in range(len(label_mapping)):
		old_label = label_mapping[i]
		y_topk[np.where(y_obj == old_label)] = i

	np.save('y_top10.npy', y_topk)

if __name__ == '__main__':

	dir_limit = 30
	n_files = count_eligible_files(dir_limit)

	process_filtered_alti_coverage(n_files, dir_limit)
	process_filtered_rgb(n_files, dir_limit)
	process_filtered_ir_coverage(n_files, dir_limit)
	process_filtered_labels(n_files, dir_limit)
	
	pdb.set_trace()


