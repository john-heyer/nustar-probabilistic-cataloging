from timeit import default_timer as timer

from jax import random

from mcmc_configs import *
from model import NuSTARModel
from sampler import NuSTARSampler
from utils import random_sources, write_results

# set random seed
key = random.PRNGKey(5)
key, sub_key = random.split(key)

# generate ground truth observation
sources_xt, sources_yt, sources_bt = random_sources(sub_key, N_SOURCES_TRUTH)
mean_image = NuSTARModel.mean_emission_map(sources_xt, sources_yt, sources_bt)
observed_image = NuSTARModel.sample_image(mean_image)

model = NuSTARModel(observed_image)
sampler = NuSTARSampler(
    model, key, burn_in_steps=BURN_IN_STEPS, samples=SAMPLES, jump_rate=JUMP_RATE, hyper_rate=HYPER_RATE,
    proposal_width_xy=PROPOSAL_WIDTH_XY, proposal_width_b=PROPOSAL_WIDTH_B, proposal_width_mu=PROPOSAL_WIDTH_MU,
    proposal_width_split=PROPOSAL_WIDTH_SPLIT, sample_batch_size=SAMPLE_BATCH_SIZE
)
sampler.sample_with_burn_in()
posterior = sampler.get_posterior_sources()
stats = sampler.get_stats()

print("Writing results to disk")
write_results(sources_xt, sources_yt, sources_bt, posterior, stats, POSTERIOR_FILE, STATS_FILE)
print("DONE")
