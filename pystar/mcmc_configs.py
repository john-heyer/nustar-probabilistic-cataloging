import os

# min and max number of allowable sources
N_MIN = 100
N_MAX = 400

# true number of sources in mock data
N_SOURCES_TRUTH = 200

# σ_xy = PROPOSAL_WIDTH_XY * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_XY = 0.75
# σ_b = PROPOSAL_WIDTH_B * sqrt(1/n_sources)
PROPOSAL_WIDTH_B = 2.0
# σ_mu = PROPOSAL_WIDTH_MU * sqrt(n_sources)
PROPOSAL_WIDTH_MU = 1.0
# σ_split = PROPOSAL_WIDTH_SPLIT * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_SPLIT = 1.5*200*7

# will produce floor(SAMPLES/SAMPLE_BATCH_SIZE) * SAMPLE_BATCH_SIZE samples
SAMPLES = 100000
BURN_IN_STEPS = 1000000
SAMPLE_BATCH_SIZE = 1000

# alternative move rates
JUMP_RATE = 0.10
HYPER_RATE = .02

# when true, will save N_KEEP emission maps randomly from each chain
CHECK_CONVERGENCE = False
N_KEEP = 10

# file names to save results
EXPERIMENT_DIR = "experiments/split_merge_only/random/wide_q_2_split"
os.makedirs(os.path.join(os.getcwd(), EXPERIMENT_DIR), exist_ok=True)  # mkdir if doesn't exist
POSTERIOR_FILE = os.path.join(EXPERIMENT_DIR, "posterior.npz")
STATS_FILE = os.path.join(EXPERIMENT_DIR, "stats.dictionary")
