import matplotlib.pyplot as plt
import numpy as np

def scores_to_rank(scores, true_idx):

	rank = len(scores) - np.where(np.argsort(scores) == true_idx)[0][0]
	return rank

def plot_cdf(ranks, max_rank):

	cdf_obj = np.zeros(max_rank)

	for i in range(max_rank):
		cdf_obj[i] = np.mean(ranks <= i+1)

	return cdf_obj
