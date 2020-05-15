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
N_CHAINS = 4
SAMPLES = 1000
BURN_IN_STEPS = 160000  # each chain goes through all BURN_IN_STEPS
SAMPLE_BATCH_SIZE = 1000 * N_CHAINS

# alternative move rates
JUMP_RATE = 0.10
HYPER_RATE = .01

# when true, will every 500th sample from each batch from all n_chains to compute r_hat statistic, i.e.
# M = N_CHAINS, N = BURN_IN_STEPS//SAMPLE_INTERVAL
CHECK_CONVERGENCE = False
SAMPLE_INTERVAL = 500

# file names to save results
EXPERIMENT_DIR = "experiments/testing/parallel/1"
os.makedirs(os.path.join(os.getcwd(), EXPERIMENT_DIR), exist_ok=True)  # mkdir if doesn't exist
POSTERIOR_FILE = os.path.join(EXPERIMENT_DIR, "posterior.npz")
STATS_FILE = os.path.join(EXPERIMENT_DIR, "stats.dictionary")
