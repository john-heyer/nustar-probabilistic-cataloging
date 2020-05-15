from timeit import default_timer as timer
import jax.numpy as np
from jax import random

from mcmc_configs import *
from model import NuSTARModel, ParameterSample
from sampler import NuSTARSampler
from utils import random_sources, random_sources_faint, write_results

# set random seed
key = random.PRNGKey(6)
key, sub_key = random.split(key)

# generate ground truth observation
sources_xt, sources_yt, sources_bt = random_sources(sub_key, N_SOURCES_TRUTH)
# sources_xt, sources_yt, sources_bt = random_sources_faint(sub_key, N_SOURCES_TRUTH)
mean_image = NuSTARModel.mean_emission_map(sources_xt, sources_yt, sources_bt)
observed_image = NuSTARModel.sample_image(mean_image)

pad = np.zeros(N_MAX - N_SOURCES_TRUTH)
params_gt = ParameterSample(
    sources_x=np.hstack((sources_xt, pad)),
    sources_y=np.hstack((sources_yt, pad)),
    sources_b=np.hstack((sources_bt, pad)),
    mask=(np.arange(N_MAX) < N_SOURCES_TRUTH),
    n=N_SOURCES_TRUTH,
    mu=float(N_SOURCES_TRUTH),
)

experiment_description = """
200 uniform sources gt b_min set to 20, init rand sources.
8 chains.
Only uses birth/death moves WITHOUT factors of n in the proposal ratio.
"""

model = NuSTARModel(observed_image)
sampler = NuSTARSampler(
    model, key, burn_in_steps=BURN_IN_STEPS, samples=SAMPLES, jump_rate=JUMP_RATE, hyper_rate=HYPER_RATE,
    proposal_width_xy=PROPOSAL_WIDTH_XY, proposal_width_b=PROPOSAL_WIDTH_B, proposal_width_mu=PROPOSAL_WIDTH_MU,
    proposal_width_split=PROPOSAL_WIDTH_SPLIT, sample_batch_size=SAMPLE_BATCH_SIZE, description=experiment_description,
    n_chains=N_CHAINS, compute_psrf=CHECK_CONVERGENCE, sample_interval=SAMPLE_INTERVAL
)
sampler.sample_with_burn_in()
posterior = sampler.get_posterior_sources()
stats = sampler.get_stats()

print("Writing results to disk...")
write_results(sources_xt, sources_yt, sources_bt, posterior, stats, POSTERIOR_FILE, STATS_FILE)
print("DONE")
