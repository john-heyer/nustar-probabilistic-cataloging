N_MIN = 100
N_MAX = 400

N_SOURCES_TRUTH = 200

# σ_xy = PROPOSAL_WIDTH_XY * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_XY = 0.75
# σ_b = PROPOSAL_WIDTH_B * sqrt(1/n_sources)
PROPOSAL_WIDTH_B = 2.0
# σ_mu = PROPOSAL_WIDTH_MU * sqrt(n_sources)
PROPOSAL_WIDTH_MU = 1.0
# σ_split = PROPOSAL_WIDTH_SPLIT * PSF_PIXEL_SIZE * sqrt(1/n_sources)
PROPOSAL_WIDTH_SPLIT = 200

# will produce floor(SAMPLES/SAMPLE_BATCH_SIZE) * SAMPLE_BATCH_SIZE samples
SAMPLES = 100000
SAMPLE_BATCH_SIZE = 1000
BURN_IN_STEPS = 0
JUMP_RATE = 0.10
HYPER_RATE = .02


CHECK_CONVERGENCE = False
N_KEEP = 10  # number of maps to save if checking for convergence
