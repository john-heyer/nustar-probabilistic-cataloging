import os
from nustar_constants import *

# min and max number of allowable sources
N_MIN = 25
N_MAX = 100

# true number of sources in mock data
N_SOURCES_TRUTH = 50

# how much to scale the length of the field of view for prior support
WINDOW_SCALE = 1.1
# spatial support (uniform)
XY_MIN = -WINDOW_SCALE * NUSTAR_IMAGE_LENGTH/2.0 * NUSTAR_PIXEL_SIZE
XY_MAX = - XY_MIN
# intensity support (uniform)
B_MIN, B_MAX = 20, 1000

# if true, generate mock sources from Normal(B_MU, B_STD^2) truncated at B_MIN AND B_MAX, else uniformly as prior
USE_FAINT_GENERATIVE_DIST = False
B_MU, B_STD = 200, 300

# σ_xy = PROPOSAL_WIDTH_XY * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_XY = 0.75
# σ_b = PROPOSAL_WIDTH_B * sqrt(1/n_sources)
PROPOSAL_WIDTH_B = 10.0
# σ_mu = PROPOSAL_WIDTH_MU * sqrt(n_sources)
PROPOSAL_WIDTH_MU = 2.0
# σ_split = PROPOSAL_WIDTH_SPLIT * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_SPLIT = 1.5*200*7

# produces floor(SAMPLES/SAMPLE_BATCH_SIZE/N_CHAINS) * SAMPLE_BATCH_SIZE * N_CHAINS samples, divided amongst N_CHAINS
N_CHAINS = 1
SAMPLES = 6000
BURN_IN_STEPS = 4000  # each chain goes through all BURN_IN_STEPS
SAMPLE_BATCH_SIZE = 1000 * N_CHAINS

# alternative move rates, divided evenly between birth/death and split/merge
BIRTH_DEATH_RATE = 0.10
SPLIT_MERGE_RATE = 0.00
HYPER_RATE = .01

# use to approximate psf with a power_law increasing speed drastically
USE_POWER_LAW_PSF_APPROXIMATION = True
# when using true PSF, up_sample from the (1300x1300) PSF by (up_sample*64 x up_sample*64) for increased accuracy
PSF_UP_SAMPLE_FACTOR = 4

# when true, use every (SAMPLE_INTERVAL)th sample from each batch from all n_chains to compute r_hat statistic, i.e.
# M = N_CHAINS, N = BURN_IN_STEPS//SAMPLE_INTERVAL
CHECK_CONVERGENCE = False
SAMPLE_INTERVAL = 500

# file names to save results, be careful not to overwrite
EXPERIMENT_DIR = "experiments/testing/parallel/2"
os.makedirs(os.path.join(os.getcwd(), EXPERIMENT_DIR), exist_ok=True)  # mkdir if doesn't exist
POSTERIOR_FILE = os.path.join(EXPERIMENT_DIR, "posterior.npz")
STATS_FILE = os.path.join(EXPERIMENT_DIR, "stats.dictionary")
