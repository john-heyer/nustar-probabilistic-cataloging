import os

PSF_BY_ARC_MINUTE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "psf_by_arcminute.npy")

NUSTAR_IMAGE_LENGTH = 64
PSF_IMAGE_LENGTH = 1300
IM_SCALE = (1300**2)/(64**2)

# in radians/pixel
NUSTAR_PIXEL_SIZE = 5.5450564776903175e-05
PSF_PIXEL_SIZE = 2.9793119397393605e-06
RADIANS_PER_ARC_MINUTE = 0.000290888

# move definitions
NORMAL = 0
BIRTH = 1
DEATH = 2
SPLIT = 3
MERGE = 4
HYPER = 5

MOVES = {
    NORMAL: 'normal move',
    BIRTH: 'birth move',
    DEATH: 'death move',
    SPLIT: 'split move',
    MERGE: 'merge move',
    HYPER: 'hyper move'
}

# strings for recording stats
ACCEPTANCE_RATE = "acceptance rate"
STATS_BY_MOVE = "stats by move type"
N_POSTERIOR = "n posterior"
MU_POSTERIOR = "mu posterior"
BATCH_ACCEPTANCE_RATES = "acceptance rates"
R_HAT = "r hat"

# per move type strings
DESCRIPTION = "description"
PROPOSED = "proposed"
ACCEPTED = "accepted"
BURN_IN = "burn in"
BATCH_SIZE = "batch size"
ZERO_MOVES = "zero alpha moves"
INF_MOVES = "inf alpha moves"
WIND_SCALE_STR = "window scale"
