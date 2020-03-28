import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
import argparse
import time
import random

parser = argparse.ArgumentParser(description='Visualize sampler results')
parser.add_argument('-l', '--last', action='store_true',
                   help='show last 2*N_gt sources in run')
parser.add_argument('stats_dir',
                   help='relative path to stats directory')

args = parser.parse_args()
last = args.last

data_dir = os.path.join(os.getcwd(), args.stats_dir)
for file in os.listdir(data_dir):
	if ".npz" in file:
		posterior_file = os.path.join(data_dir, file)
	else:
		acceptance_file = os.path.join(data_dir, file)

PSF_PIXEL_SIZE = 2.9793119397393605e-06
window_min, window_max = -1300/2, 1300/2

metropolis_data = np.load(posterior_file)

posterior = metropolis_data["posterior"]
ground_truth = metropolis_data["gt"]
init = metropolis_data["init"] if "init" in metropolis_data else False

with open(acceptance_file, "r") as f:
	stats = json.load(f)

move_stats = stats.pop("stats by move type")
n_sources_counts = stats.pop('n_sources_counts')
has_mus = "mus" in stats
if has_mus:
	mus = stats.pop("mus")
a_rates = "acceptance_rates" in stats
if a_rates:
	acceptance_rates = stats.pop("acceptance_rates")

print("\n======acceptance stats======")
for stat in stats:
	print(stat, ":", stats[stat])

for move in move_stats:
	print(move)
	proposed = 0
	for move_stat in move_stats[move]:
		print("\t", move_stat, ":", move_stats[move][move_stat])
		if move_stat == "proposed":
			proposed = move_stats[move][move_stat]
		elif move_stat == "accepted":
			if proposed != 0:
				print("\t", "acceptance rate", ":", move_stats[move][move_stat]/proposed)

print("============================\n")

# if init is not False:
# 	print("n sources init: ", init.shape[1])
# print("n sources gt: ", ground_truth.shape[1])
# print("======n source counts======")
# for count in n_sources_counts:
# 	print(count, ":", n_sources_counts[count])
# print("===========================\n")

# print(ground_truth.shape)
# print(posterior.shape)

# print(max(ground_truth[0]), min(ground_truth[0]))
# print(max(ground_truth[1]), min(ground_truth[1]))
# print(max(ground_truth[2]), min(ground_truth[2]))
# print(max(posterior[0]), min(posterior[0]))
# print(max(posterior[1]), min(posterior[1]))
# print(max(posterior[2]), min(posterior[2]))


# TO PRINT GT
# gt = [(ground_truth[0][i], ground_truth[1][i], ground_truth[2][i]) for i in range(len(ground_truth[0]))]
# print(gt)

def percent_outside(sources_x, sources_y):
	sources_x = sources_x.reshape(-1, 1)
	sources_y = sources_y.reshape(-1, 1)
	sources = np.hstack((sources_x, sources_y))
	outside = np.where((np.abs(sources[:,0]) > window_max) | (np.abs(sources[:,1]) > window_max))
	return outside[0].shape[0] / len(sources_x)


gt_x, gt_y, gt_b = ground_truth[0]/PSF_PIXEL_SIZE, ground_truth[1]/PSF_PIXEL_SIZE, np.exp(ground_truth[2])
# print("percent under 50:", gt_b[gt_b<54].shape[0]/200)

if init is not False:
	i_x, i_y, i_b = init[0]/PSF_PIXEL_SIZE, init[1]/PSF_PIXEL_SIZE, np.exp(init[2])

p_x, p_y, p_b = posterior[0]/PSF_PIXEL_SIZE, posterior[1]/PSF_PIXEL_SIZE, np.exp(posterior[2])

print("expected percent outside: .174")
print("percent outside gt:", percent_outside(gt_x, gt_y))
print("percent outside post:", percent_outside(p_x, p_y))

last_n_x, last_n_y, last_n_b = p_x[len(p_x)-2*len(gt_x):len(p_x)], p_y[len(p_x)-2*len(gt_x):len(p_x)], p_b[len(p_x)-2*len(gt_x):len(p_x)]

plt.hist2d(x=p_x, y=p_y, range=[[-1000, 1000], [-1000, 1000]], bins=128)#, weights=p_b)
plt.scatter(x=gt_x, y=gt_y, c=gt_b, s=10, edgecolors='black')
if last:
	plt.scatter(x=last_n_x, y=last_n_y, c='r', marker='+')
if init is not False:
	plt.scatter(x=i_x, y=i_y, s=10, c='r', edgecolors='black')
plt.gca().add_patch(Rectangle((window_min,window_min),2*window_max,2*window_max,linewidth=.5,edgecolor='r',facecolor='none'))
plt.gca().add_patch(Rectangle((1.1*window_min,1.1*window_min),1.1*2*window_max,1.1*2*window_max,linewidth=.5,edgecolor='black',facecolor='none'))
plt.show()

if has_mus:
	plt.hist(x=mus, bins=25, color='y', edgecolor='k')
	plt.title("Mu Posterior")
	plt.show()


source_count_tuples = sorted(list(zip(n_sources_counts.keys(), n_sources_counts.values())), key=lambda tup: int(tup[0]))
n_sources = [int(tup[0])for tup in source_count_tuples]
count = [tup[1] for tup in source_count_tuples]
plt.bar(n_sources, count, color='g', edgecolor='k')
plt.title("N Posterior")
plt.axvline(len(gt_x), color='k', linestyle='dashed', linewidth=1)
plt.show()



gt_b_sort = np.sort(gt_b)
pr_b = np.linspace(1, 0, np.size(gt_b_sort))
plt.scatter(x=gt_b_sort, y=pr_b)
plt.title("CDF B GT")
plt.show()

posterior_b_sample = np.array(random.sample(list(p_b), 2000))
posterior_b_sort = np.sort(posterior_b_sample)
pr_b = np.linspace(1, 0, np.size(posterior_b_sort))
plt.scatter(x=posterior_b_sort, y=pr_b)
plt.title("CDF B POSTERIOR")
plt.show()

if a_rates:
	plt.scatter(x=[(i+1)*1000 for i in range(len(acceptance_rates))], y=acceptance_rates)
	plt.title("Acceptance_rate per 1000 iterations")
	plt.show()

# X = np.array([p_x, p_y]).T
# init = np.array([gt_x, gt_y]).T
# print(X.shape, init.shape)
# kmeans = KMeans(n_clusters=ground_truth.shape[1], random_state=0).fit(X)

# print(max(p_x), max(p_y))
# print(min(p_x), min(p_y))
# centers = kmeans.cluster_centers_
# print(init)
# print(centers)

# maps = np.load("inf_ratio.npz")
# sample_map = maps["sampled_img"]
#
#
# accepted = maps["accepted_img"]
# previous = maps["previous_img"]
#
# plt.matshow(previous)
#
# plt.show()
