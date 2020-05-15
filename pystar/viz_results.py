import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from nustar_constants import *

parser = argparse.ArgumentParser(description='Visualize sampler results')
parser.add_argument(
    'stats_dir',
    help='relative path to experiment directory containing a posterior npz file and pickled stats dictionary file'
)
args = parser.parse_args()

data_dir = os.path.join(os.getcwd(), args.stats_dir)
posterior_file, stats_obj_file = None, None
for file in os.listdir(data_dir):
    if ".npz" in file:
        posterior_file = os.path.join(data_dir, file)
    else:
        stats_obj_file = os.path.join(data_dir, file)

window_min, window_max = -PSF_IMAGE_LENGTH/2, PSF_IMAGE_LENGTH/2

posterior_data = np.load(posterior_file)

posterior = posterior_data["posterior"]
ground_truth = posterior_data["ground_truth"]

with open(stats_obj_file, "rb") as sf:
    stats = pickle.load(sf)

move_stats = stats.pop(STATS_BY_MOVE)
n_posterior = stats.pop(N_POSTERIOR)
mu_posterior = stats.pop(MU_POSTERIOR)
acceptance_rates = stats.pop(BATCH_ACCEPTANCE_RATES)
r_hat = stats.pop(R_HAT)

print("\n======acceptance stats======")

for stat in stats:
    print(stat, ":", stats[stat])

for move in move_stats:
    print(move)
    proposed = 0
    for move_stat in move_stats[move]:
        print("\t", move_stat, ":", move_stats[move][move_stat])
        if move_stat == PROPOSED:
            proposed = move_stats[move][move_stat]
        elif move_stat == ACCEPTED:
            if proposed != 0:
                print("\t", ACCEPTANCE_RATE, ":", move_stats[move][move_stat]/proposed)

print("============================\n")


def percent_outside(sources_x, sources_y):
    sources_x = sources_x.reshape(-1, 1)
    sources_y = sources_y.reshape(-1, 1)
    sources = np.hstack((sources_x, sources_y))
    outside = np.where((np.abs(sources[:,0]) > window_max) | (np.abs(sources[:,1]) > window_max))
    return outside[0].shape[0] / len(sources_x)


# get last for inspection
last_x, last_y, last_b = posterior[-1]
last_x, last_y, last_b = last_x[last_b != 0]/PSF_PIXEL_SIZE, last_y[last_b != 0]/PSF_PIXEL_SIZE, last_b[last_b != 0]
print('N last sample:', len(last_x))

# posterior is shape (samples, 3, N_MAX)
p_x = np.concatenate(posterior[:, 0, :])
p_y = np.concatenate(posterior[:, 1, :])
p_b = np.concatenate(posterior[:, 2, :])

p_x = p_x[p_b != 0]
p_y = p_y[p_b != 0]
p_b = p_b[p_b != 0]

total_sources = np.sum([n * n_posterior[n] for n in n_posterior])

# assert len(p_x) == total_sources, f"size of posterior sources ({len(p_x)}) not consistent with N posterior total ({total_sources})"

gt_x, gt_y, gt_b = ground_truth[0]/PSF_PIXEL_SIZE, ground_truth[1]/PSF_PIXEL_SIZE, ground_truth[2]
p_x, p_y = p_x/PSF_PIXEL_SIZE, p_y/PSF_PIXEL_SIZE

# we expect 17.4 of sources to be outside FOV but in prior window
print(f"expected percent outside: {100 * (1 - 1/WINDOW_SCALE**2)}")
print("percent outside gt:", 100 * percent_outside(gt_x, gt_y))
print("percent outside post:", 100 * percent_outside(p_x, p_y))
print("percent outside last:", 100 * percent_outside(last_x, last_y))


# histogram of r_hat statistic over 64x64 nustar image
if r_hat is not None:
    r_hat = r_hat.reshape(r_hat.shape[0] * r_hat.shape[1])
    r_hat = sorted(r_hat)
    plt.hist(r_hat, ec='black')
    plt.show()


# plot 2d histogram of sources
plt.hist2d(x=p_x, y=p_y, range=[[-800, 800], [-800, 800]], bins=64, weights=p_b)
plt.scatter(x=gt_x, y=gt_y, c=gt_b, s=30, edgecolors='black')
plt.scatter(x=last_x, y=last_y, c=last_b, s=30, edgecolors='red')
plt.gca().add_patch(Rectangle((window_min,window_min),2*window_max,2*window_max,linewidth=.5,edgecolor='r',facecolor='none'))
plt.gca().add_patch(Rectangle((WINDOW_SCALE*window_min,WINDOW_SCALE*window_min),WINDOW_SCALE*2*window_max,WINDOW_SCALE*2*window_max,linewidth=.5,edgecolor='black',facecolor='none'))
plt.show()

# plot histogram of mus
plt.hist(x=list(mu_posterior.keys()), weights=list(mu_posterior.values()), bins=25, color='y', edgecolor='k')
plt.title("Mu Posterior")
plt.axvline(len(gt_x), color='k', linestyle='dashed', linewidth=1)
plt.show()

# plot bar graph of ns
source_count_tuples = sorted(list(zip(n_posterior.keys(), n_posterior.values())), key=lambda tup: int(tup[0]))
n_sources = [int(tup[0])for tup in source_count_tuples]
count = [tup[1] for tup in source_count_tuples]
plt.bar(n_sources, count, color='g', edgecolor='k')
plt.title("N Posterior")
plt.axvline(len(gt_x), color='k', linestyle='dashed', linewidth=1)
plt.show()

# plot cdf over source brightness for ground truth
gt_b_sort = np.sort(gt_b)
pr_b = np.linspace(1, 0, np.size(gt_b_sort))
plt.scatter(x=gt_b_sort, y=pr_b)
print('min b gt:', np.min(gt_b_sort))
plt.title("CDF B GT")
plt.show()

# plot cdf over source brightness for posterior
posterior_b_sample = np.random.choice(p_b, size=2000, replace=False)
posterior_b_sort = np.sort(posterior_b_sample)
pr_b = np.linspace(1, 0, np.size(posterior_b_sort))
plt.scatter(x=posterior_b_sort, y=pr_b)
plt.title("CDF B POSTERIOR")
plt.show()

plt.hist(x=gt_b)
plt.title("gt b")
plt.show()

plt.hist(x=p_b)
plt.title("post b")
plt.show()

# plot acceptance rate over time
batch_size = stats[BATCH_SIZE]
plt.scatter(x=[(i+1)*batch_size for i in range(len(acceptance_rates))], y=acceptance_rates)
plt.title(f"Acceptance_rate per {batch_size} iterations")
plt.show()

