import os

# min and max number of allowable sources
N_MIN = 100
N_MAX = 400

# true number of sources in mock data
N_SOURCES_TRUTH = 200

# σ_xy = PROPOSAL_WIDTH_XY * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_XY = 0.75
# σ_b = PROPOSAL_WIDTH_B * sqrt(1/n_sources)
PROPOSAL_WIDTH_B = 10.0
# σ_mu = PROPOSAL_WIDTH_MU * sqrt(n_sources)
PROPOSAL_WIDTH_MU = 2.0
# σ_split = PROPOSAL_WIDTH_SPLIT * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_SPLIT = 1.5*200*7

# produces floor(SAMPLES/SAMPLE_BATCH_SIZE/N_CHAINS) * SAMPLE_BATCH_SIZE * N_CHAINS samples, divided amongst N_CHAINS
N_CHAINS = 8
SAMPLES = 100000
BURN_IN_STEPS = 500000  # each chain goes through all BURN_IN_STEPS
SAMPLE_BATCH_SIZE = 1000

# alternative move rates
JUMP_RATE = 0.10
HYPER_RATE = .01

# when true, will save last sample from each batch from all n_chains to compute r_hat statistic
CHECK_CONVERGENCE = True

# file names to save results
EXPERIMENT_DIR = "experiments/testing/true_psf/2"
os.makedirs(os.path.join(os.getcwd(), EXPERIMENT_DIR), exist_ok=True)  # mkdir if doesn't exist
POSTERIOR_FILE = os.path.join(EXPERIMENT_DIR, "posterior.npz")
STATS_FILE = os.path.join(EXPERIMENT_DIR, "stats.dictionary")
