import pickle

import jax.numpy as np
import numpy as onp
from jax import random
from scipy.stats import truncnorm

from mcmc_configs import *


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def random_sources(rng_key, n):
    key1, key2, key3 = random.split(rng_key, 3)
    sources_x = random.uniform(key1, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    sources_y = random.uniform(key2, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    sources_b = random.uniform(key3, shape=(n,), minval=B_MIN, maxval=B_MAX)
    return sources_x, sources_y, sources_b


def random_sources_faint(rng_key, n, mu, sigma):
    key1, key2, key3 = random.split(rng_key, 3)
    b_dist = get_truncated_normal(mean=mu, sd=sigma, low=B_MIN, upp=B_MAX)
    sources_b = np.array(b_dist.rvs(n))
    sources_x = random.uniform(key1, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    sources_y = random.uniform(key2, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    return sources_x, sources_y, sources_b


def random_source(rng_key):
    key1, key2, key3 = random.split(rng_key, 3)
    source_x = random.uniform(key1, minval=XY_MIN, maxval=XY_MAX)
    source_y = random.uniform(key2, minval=XY_MIN, maxval=XY_MAX)
    source_b = random.uniform(key3, minval=B_MIN, maxval=B_MAX)
    return source_x, source_y, source_b


def write_results(sources_x, sources_y, sources_b, posterior_sources, stats, posterior_file, stats_file):
    gt = onp.array([sources_x, sources_y, sources_b])  # shape = (3 x n_sources_truth)
    onp.savez(posterior_file, ground_truth=gt, posterior=posterior_sources)
    with open(stats_file, 'wb') as sf:
        pickle.dump(stats, sf)
