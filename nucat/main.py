import jax.numpy as np
from jax import random

from mcmc_configs import *
from model import NuSTARModel, ParameterSample
from sampler import NuSTARSampler
from utils import random_sources, random_sources_faint, write_results

import numpy as onp

# set random seeds
jax_seed = 24
np_seed = 1
onp.random.seed(np_seed)  # for drawing poisson numbers
key = random.PRNGKey(jax_seed)
key, sub_key = random.split(key)

# generate ground truth observation uniformly from prior or from faint b distribution
if USE_FAINT_GENERATIVE_DIST:
    sources_xt, sources_yt, sources_bt = random_sources_faint(sub_key, N_SOURCES_TRUTH, B_MU, B_STD)
else:
    sources_xt, sources_yt, sources_bt = random_sources(sub_key, N_SOURCES_TRUTH)

# generate observed image, using power_law approximation or true psf
if USE_POWER_LAW_PSF_ESTIMATE:
    mean_image = NuSTARModel.mean_emission_map_power_law(sources_xt, sources_yt, sources_bt)
else:
    mean_image = NuSTARModel.mean_emission_map(sources_xt, sources_yt, sources_bt, PSF_UP_SAMPLE_FACTOR)
observed_image = NuSTARModel.sample_image(mean_image)

pad = np.zeros(N_MAX - N_SOURCES_TRUTH)
# use to seed with gt for testing by passing to sampler as params_init
params_gt = ParameterSample(
    sources_x=np.hstack((sources_xt, pad)),
    sources_y=np.hstack((sources_yt, pad)),
    sources_b=np.hstack((sources_bt, pad)),
    mask=(np.arange(N_MAX) < N_SOURCES_TRUTH),
    n=N_SOURCES_TRUTH,
    mu=float(N_SOURCES_TRUTH),
)

# add optional description, e.g.
experiment_description = f"""
**ADD OPTIONAL EXPERIMENT DESCRIPTION **

Configs:
    {key=},
    {SAMPLES=},
    {N_SOURCES_TRUTH=},
    {N_MIN=}, {N_MAX=},
    {N_CHAINS=},
    {USE_FAINT_GENERATIVE_DIST=},
    {USE_POWER_LAW_PSF_ESTIMATE=},
    {B_MIN=}, {B_MAX=},
    {WINDOW_SCALE=},
    {BIRTH_DEATH_RATE=},
    {SPLIT_MERGE_RATE=},
    {HYPER_RATE=},
    {PROPOSAL_WIDTH_XY=},
    {PROPOSAL_WIDTH_B=},
    {PROPOSAL_WIDTH_MU=},
    {PROPOSAL_WIDTH_SPLIT=},
"""

model = NuSTARModel(observed_image, use_power_law=USE_POWER_LAW_PSF_ESTIMATE, up_sample=PSF_UP_SAMPLE_FACTOR)
sampler = NuSTARSampler(model, key,
                        burn_in_steps=BURN_IN_STEPS, samples=SAMPLES, n_chains=N_CHAINS,
                        use_power_law=USE_POWER_LAW_PSF_ESTIMATE, up_sample=PSF_UP_SAMPLE_FACTOR,
                        birth_death_rate=BIRTH_DEATH_RATE, split_merge_rate=SPLIT_MERGE_RATE, hyper_rate=HYPER_RATE,
                        proposal_width_xy=PROPOSAL_WIDTH_XY, proposal_width_b=PROPOSAL_WIDTH_B,
                        proposal_width_mu=PROPOSAL_WIDTH_MU, proposal_width_split=PROPOSAL_WIDTH_SPLIT,
                        sample_batch_size=SAMPLE_BATCH_SIZE, description=experiment_description,
                        compute_psrf=CHECK_CONVERGENCE, sample_interval=SAMPLE_INTERVAL,
                        )

sampler.sample_with_burn_in()
posterior = sampler.get_posterior_sources()
stats = sampler.get_stats()

print("Writing results to disk...")
write_results(sources_xt, sources_yt, sources_bt, posterior, stats, POSTERIOR_FILE, STATS_FILE)
print("DONE")
