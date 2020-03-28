NUSTAR_IMAGE_LENGTH = 64
PSF_IMAGE_LENGTH = 1300
IM_SCALE = (1300**2)/(64**2)

# In radians/pixel
NUSTAR_PIXEL_SIZE = 5.5450564776903175e-05
PSF_PIXEL_SIZE = 2.9793119397393605e-06

# prior boundaries
XY_MIN = -1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE
XY_MAX = - XY_MIN
B_MIN, B_MAX = 1, 1096.6331584284585

# strings
PROPOSED = "proposed"
ACCEPTED = "accepted"
ZERO_MOVES = "zero A moves"
INF_MOVES = "inf A moves"

# TODO: Fix string conventions for consistency, left for compatibility
ACCEPTANCE_RATE = "acceptance rate"
STATS_BY_MOVE = "stats by move type"
N_SOURCES_COUNTS = "n_sources_counts"
ACCEPTANCE_RATES = "acceptance_rates"
