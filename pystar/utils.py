from nustar_constants import *
from mcmc_configs import *

import jax.numpy as np
from jax import grad, jit, vmap, random, scipy


def random_sources(rng_key, n):
    key1, key2, key3 = random.split(rng_key, 3)
    sources_x = random.uniform(key1, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    sources_y = random.uniform(key2, shape=(n,), minval=XY_MIN, maxval=XY_MAX)
    sources_b = random.uniform(key3, shape=(n,), minval=B_MIN, maxval=B_MAX)
    return sources_x, sources_y, sources_b


@jit
def random_source(rng_key):
    key1, key2, key3 = random.split(rng_key, 3)
    source_x = random.uniform(key1, minval=XY_MIN, maxval=XY_MAX)
    source_y = random.uniform(key2, minval=XY_MIN, maxval=XY_MAX)
    source_b = random.uniform(key3, minval=B_MIN, maxval=B_MAX)
    return source_x, source_y, source_b


def write_results(file):
    pass
