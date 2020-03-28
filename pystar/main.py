from mcmc_configs import *

from model import NuSTARModel, ParameterSample
from sampler import NuSTARSampler
from utils import random_sources

import jax.numpy as np
from jax import random
from timeit import default_timer as timer

# set random seed
key = random.PRNGKey(5)
print(key)
key, sub_key = random.split(key)
print(key)

# generate ground truth observation
sources_xt, sources_yt, sources_bt = random_sources(sub_key, N_SOURCES_TRUTH)
mean_image = NuSTARModel.mean_emission_map(sources_xt, sources_yt, sources_bt)
observed_image = NuSTARModel.sample_image(mean_image)

model = NuSTARModel(observed_image)
sampler = NuSTARSampler(
    model, key, burn_in=BURN_IN_STEPS, samples=SAMPLES, jump_rate=JUMP_RATE, hyper_rate=HYPER_RATE,
    proposal_width_xy=PROPOSAL_WIDTH_XY, proposal_width_b=PROPOSAL_WIDTH_B, proposal_width_mu=PROPOSAL_WIDTH_MU,
    proposal_width_split=SPLIT_PROPOSAL_WIDTH
)
print(sampler.head.sources_x.shape)
start = timer()
sampler.sample()
end = timer()
print("Sampling time:", end - start)
print(sampler.rng_key)
print(sampler.head.sources_x.shape)

# TODO: write results
