from nustar_constants import *
from mcmc_configs import *
from utils import random_sources, random_source
from model import NuSTARModel, ParameterSample

from collections import OrderedDict
from enum import Enum
from tqdm import tqdm

import jax.numpy as np
from jax import grad, jit, vmap, random, scipy, lax
from jax.ops import index_update
import numpy.random as nprand  # for drawing poisson random numbers
nprand.seed(1)

from functools import partial
from timeit import default_timer as timer


NORMAL = 0
BIRTH = 1
DEATH = 2
SPLIT = 3
MERGE = 4
HYPER = 5

MOVES = [NORMAL, BIRTH, DEATH, SPLIT, MERGE, HYPER]


class NuSTARSampler:

    def __init__(
            self, model, rng_key, params_init=None, burn_in_steps=0, samples=10000, jump_rate=.1, hyper_rate=.02,
            proposal_width_xy=.75, proposal_width_b=2.0, proposal_width_mu=1.0, proposal_width_split=200,
            sample_batch_size=1000
    ):
        self.model = model
        self.rng_key = rng_key

        # initial params
        if params_init is None:
            self.head = self.random_params(self.rng_key)
        else:
            self.head = params_init
        self.log_joint_head = model.log_joint(self.head)
        self.mean_emission_map = NuSTARModel.mean_emission_map(
            self.head.sources_x, self.head.sources_y, self.head.sources_b
        )
        self.chain = [self.head]
        NuSTARSampler.check_parameter_invariants(self.head)

        # sampler configs
        self.burn_in_steps = burn_in_steps
        self.samples = samples
        self.jump_rate = jump_rate
        self.hyper_rate = hyper_rate
        self.proposal_width_xy = proposal_width_xy
        self.proposal_width_b = proposal_width_b
        self.proposal_width_mu = proposal_width_mu
        self.proposal_width_split = proposal_width_split

        # stats
        self.move_stats = self.__new_move_stats()
        self.accepted = 0
        self.sample_batch_size = sample_batch_size
        self.acceptance_rates_per_period = []
        self.n_sources_counts = {}

    @staticmethod
    def check_parameter_invariants(params):
        # only used for testing as will not compile with jax
        assert params.n == np.sum(params.mask), "parameter n not matching mask"
        assert np.all(params.mask == (np.arange(N_MAX) < params.n)), "mask is not an array of ones with zero padding"
        assert np.all(params.sources_x[np.logical_not(params.mask)] == 0), "outside mask x values not zero"
        assert np.all(params.sources_y[np.logical_not(params.mask)] == 0), "outside mask y values not zero"
        assert np.all(params.sources_b[np.logical_not(params.mask)] == 0), "outside mask b values not zero"

    @staticmethod
    def random_params(rng_key):
        sub_key, sub_key_2 = random.split(rng_key, 2)
        mu_init = np.exp(random.uniform(sub_key, minval=np.log(N_MIN), maxval=np.log(N_MAX)))
        n_init = nprand.poisson(mu_init)
        sources_x_init, sources_y_init, sources_b_init = random_sources(sub_key_2, n_init)
        pad = np.zeros(N_MAX - n_init)
        return ParameterSample(
            sources_x=np.hstack((sources_x_init, pad)),
            sources_y=np.hstack((sources_y_init, pad)),
            sources_b=np.hstack((sources_b_init, pad)),
            mask=(np.arange(N_MAX) < n_init),
            mu=mu_init,
            n=n_init
        )

    def __new_move_stats(self):
        move_stats = OrderedDict()
        for move in MOVES:
            move_stats[move] = OrderedDict(
                {
                    PROPOSED: 0,
                    ACCEPTED: 0,
                    ZERO_MOVES: 0,
                    INF_MOVES: 0
                }
            )
        return move_stats

    def __record_move(self, move_type, alpha, accept):
        zero_alpha = (alpha == 0)
        inf_alpha = (alpha == np.inf)
        self.move_stats[move_type][PROPOSED] += 1
        self.move_stats[move_type][ACCEPTED] += accept
        self.move_stats[move_type][ZERO_MOVES] += zero_alpha
        self.move_stats[move_type][INF_MOVES] += inf_alpha

    @staticmethod
    @jit
    def __get_move_type(rng_key, jump_rate, hyper_rate):
        normal_rate = 1 - jump_rate - hyper_rate
        jump_p = jump_rate / 4
        logits = np.log(np.array([normal_rate, jump_p, jump_p, jump_p, jump_p, hyper_rate]))
        return random.categorical(rng_key, logits)

    @staticmethod
    @jit
    def birth(rng_key, params):
        source_x, source_y, source_b = random_source(rng_key)
        log_proposal_ratio = (2 * np.log(XY_MAX - XY_MIN) + np.log(B_MAX - B_MIN))
        mask_new_source = np.arange(N_MAX) == params.n
        sample_new = ParameterSample(
            sources_x=(params.sources_x + mask_new_source * source_x),
            sources_y=(params.sources_y + mask_new_source * source_y),
            sources_b=(params.sources_b + mask_new_source * source_b),
            mask=(np.arange(N_MAX) < (params.n + 1)),
            mu=params.mu,
            n=(params.n + 1)
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    def remove_and_shift(arr, i_remove):
        unmoved = arr * (np.arange(arr.shape[0]) < i_remove)
        to_shift = arr * (np.arange(arr.shape[0]) > i_remove)
        rolled_left = np.roll(to_shift, -1)
        return unmoved + rolled_left

    @staticmethod
    @jit
    def death(rng_key, params):
        i_source = random.randint(rng_key, shape=(), minval=0, maxval=params.n)
        log_proposal_ratio = -(2 * np.log(XY_MAX - XY_MIN) + np.log(B_MAX - B_MIN))
        sample_new = ParameterSample(
            sources_x=NuSTARSampler.remove_and_shift(params.sources_x, i_source),
            sources_y=NuSTARSampler.remove_and_shift(params.sources_y, i_source),
            sources_b=NuSTARSampler.remove_and_shift(params.sources_b, i_source),
            mask=(np.arange(N_MAX) < (params.n - 1)),
            mu=params.mu,
            n=(params.n - 1)
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    def utr_idx(N, i, j):
        return N * i - (i * (i + 1)) // 2 + (j - i - 1)

    @staticmethod
    @jit
    def distance_distribution(params):
        def distance(i, j):
            is_real_pair = (j != 0)
            d = (
                (params.sources_x[i] - params.sources_x[j]) ** 2 +
                (params.sources_y[i] - params.sources_y[j]) ** 2
            )
            return np.where(is_real_pair, 1.0/d, 0)
        n = params.sources_x.shape[0]
        sources_one, sources_two = np.triu_indices(n, 1)
        sources_two_mask = sources_two * (sources_two < params.n)  # set the padded indices to zero
        pairwise_distances = vmap(distance, in_axes=(0, 0))(sources_one, sources_two_mask)
        distance_dist = pairwise_distances/(np.sum(pairwise_distances))
        return np.vstack((sources_one, sources_two, distance_dist)).T

    @staticmethod
    @jit
    def split(rng_key, params, proposal_width_split):
        sigma_split = proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0/params.n)
        key1, key2, key3 = random.split(rng_key, 3)
        i_source = random.randint(key1, shape=(), minval=0, maxval=params.n)
        alpha = random.uniform(key2, minval=0, maxval=1)
        q_x, q_y = random.normal(key3, shape=(2,)) * sigma_split
        x, y, b = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]
        new_x1, new_x2 = x + q_x/2, x - q_x/2
        new_y1, new_y2 = y + q_y/2, y - q_y/2
        new_b1, new_b2 = b * alpha, b * (1-alpha)

        def split_padded_sources(arr, i_remove, n, new1, new2):
            source_removed = NuSTARSampler.remove_and_shift(arr, i_remove)
            mask_new1 = np.arange(N_MAX) == (n-1)
            mask_new2 = np.arange(N_MAX) == n
            return source_removed + new1 * mask_new1 + new2 * mask_new2

        sample_new = ParameterSample(
            sources_x=split_padded_sources(params.sources_x, i_source, params.n, new_x1, new_x2),
            sources_y=split_padded_sources(params.sources_y, i_source, params.n, new_y1, new_y2),
            sources_b=split_padded_sources(params.sources_b, i_source, params.n, new_b1, new_b2),
            mask=(np.arange(N_MAX) < (params.n + 1)),
            mu=params.mu,
            n=(params.n + 1)
        )
        # sum_0, sum_new = np.sum(params.sources_b), np.sum(sample_new.sources_b)
        # assert np.isclose(sum_0, sum_new), f"split brightness not conserved, {sum_0}, {sum_new}"
        # full proposal ratio: p(inverse_merge) / p(split) * |J| = b
        distance_dist = NuSTARSampler.distance_distribution(sample_new)
        pair_idx = NuSTARSampler.utr_idx(N_MAX, params.n-1, params.n)

        log_p_merge = np.log(distance_dist[pair_idx][2])  # new sources will be last pair in the distance distribution
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split = log_p_q - np.log(params.n)  # p(q) * 1/n for choice of source
        log_proposal_ratio = log_p_merge - log_p_split + np.log(b)
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def merge(rng_key, params, proposal_width_split):
        distance_dist = NuSTARSampler.distance_distribution(params)
        logits = np.log(distance_dist.T[2])
        i_pair = random.categorical(rng_key, logits)
        i_source, j_source, p_merge = distance_dist[i_pair]
        i_source, j_source = i_source.astype(np.int32), j_source.astype(np.int32)
        # assert i_source < params.n, "chose padded source i from distance distribution"
        # assert j_source < params.n, "chose padded source j from distance distribution"
        # assert i_source < j_source, "source j merge less than source i"
        x_i, y_i, b_i = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]
        x_j, y_j, b_j = params.sources_x[j_source], params.sources_y[j_source], params.sources_b[j_source]
        q_x, q_y = (x_i - x_j), (y_i - y_j)
        x_new, y_new = (x_i + x_j)/2, (y_i + y_j)/2
        b_new = b_i + b_j

        def merge_padded_sources(arr, i_remove, j_remove, n, new):
            source_j_removed = NuSTARSampler.remove_and_shift(arr, j_remove)  # j > i, so j removed first
            sources_removed = NuSTARSampler.remove_and_shift(source_j_removed, i_remove)
            mask_new = np.arange(N_MAX) == (n - 2)
            return sources_removed + new * mask_new

        sample_new = ParameterSample(
            sources_x=merge_padded_sources(params.sources_x, i_source, j_source, params.n, x_new),
            sources_y=merge_padded_sources(params.sources_y, i_source, j_source, params.n, y_new),
            sources_b=merge_padded_sources(params.sources_b, i_source, j_source, params.n, b_new),
            mask=(np.arange(N_MAX) < (params.n - 1)),
            mu=params.mu,
            n=(params.n - 1)
        )
        # sum_0, sum_new = np.sum(params.sources_b), np.sum(sample_new.sources_b)
        # assert np.isclose(sum_0, sum_new), f"merge brightness not conserved, {sum_0}, {sum_new}, {b_new}, {b_i}, {b_j}"
        # full proposal ratio: p(inverse_split) / p(merge) * 1/|J| = 1/b
        new_n = sample_new.n
        sigma_split = proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0 / new_n)
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split = log_p_q - np.log(new_n)
        log_p_merge = np.log(p_merge)
        log_proposal_ratio = log_p_split - log_p_merge - np.log(b_new)
        return sample_new, log_proposal_ratio

    @staticmethod
    def __birth_death(tup_args):
        rng_key, params, move_type = tup_args
        up = (move_type == BIRTH)
        birth_fn = lambda args: NuSTARSampler.birth(*args)
        death_fn = lambda args: NuSTARSampler.death(*args)
        sample_new, log_proposal_ratio = lax.cond(
            up,
            (rng_key, params),
            birth_fn,
            (rng_key, params),
            death_fn
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    def __split_merge(tup_args):
        rng_key, params, move_type, proposal_width_split = tup_args
        up = (move_type == SPLIT)
        sample_new, log_proposal_ratio = lax.cond(
            up,
            (rng_key, params, proposal_width_split),
            lambda args: NuSTARSampler.split(*args),
            (rng_key, params, proposal_width_split),
            lambda args: NuSTARSampler.merge(*args)
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    def __jump_proposal(tup_args):
        rng_key, params, move_type, proposal_width_split = tup_args
        birth_death = ((move_type == BIRTH) | (move_type == DEATH))
        sample_new, log_proposal_ratio = lax.cond(
            birth_death,
            (rng_key, params, move_type),
            NuSTARSampler.__birth_death,
            (rng_key, params, move_type, proposal_width_split),
            NuSTARSampler.__split_merge
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def normal_proposal(rng_key, params, proposal_width_xy, proposal_width_b):
        key1, key2, key3 = random.split(rng_key, 3)
        sigma_xy = proposal_width_xy * PSF_PIXEL_SIZE * np.sqrt(1.0/params.n)
        sigma_b = proposal_width_b * np.sqrt(1.0/params.n)
        q_x = random.normal(key1, shape=(N_MAX,)) * sigma_xy
        q_y = random.normal(key2, shape=(N_MAX,)) * sigma_xy
        q_b = random.normal(key3, shape=(N_MAX,)) * sigma_b
        sample_new = ParameterSample(
            sources_x=params.sources_x + q_x * params.mask,
            sources_y=params.sources_y + q_y * params.mask,
            sources_b=params.sources_b + q_b * params.mask,
            mask=params.mask,
            mu=params.mu,
            n=params.n
        )
        log_proposal_ratio = 0.0
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def hyper_proposal(rng_key, params, proposal_width_mu):
        sample_new = ParameterSample(
            sources_x=params.sources_x,
            sources_y=params.sources_y,
            sources_b=params.sources_b,
            mask=params.mask,
            mu=params.mu + (random.normal(rng_key) * proposal_width_mu * np.sqrt(params.n)),
            n=params.n
        )
        log_proposal_ratio = 0.0
        return sample_new, log_proposal_ratio

    @staticmethod
    def __normal_hyper(tup_args):
        rng_key, params, move_type, proposal_width_xy, proposal_width_b, proposal_width_mu = tup_args
        normal = (move_type == NORMAL)
        sample_new, log_proposal_ratio = lax.cond(
            normal,
            (rng_key, params, proposal_width_xy, proposal_width_b),
            lambda args: NuSTARSampler.normal_proposal(*args),
            (rng_key, params, proposal_width_mu),
            lambda args: NuSTARSampler.hyper_proposal(*args)
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    @partial(jit, static_argnums=(2,3,4,5,6))
    def __cond_proposal(rng_key, params, model, proposal_width_xy, proposal_width_b, proposal_width_split, proposal_width_mu, move_type):
        jump = ((move_type != NORMAL) & (move_type != HYPER))
        # start = timer()
        sample_new, log_proposal_ratio = lax.cond(
            jump,
            (rng_key, params, move_type, proposal_width_split),
            NuSTARSampler.__jump_proposal,
            (rng_key, params, move_type, proposal_width_xy, proposal_width_b, proposal_width_mu),
            NuSTARSampler.__normal_hyper
        )
        sample_log_joint = model.log_joint(sample_new)
        return sample_new, log_proposal_ratio, sample_log_joint

    @partial(jit, static_argnums=(0, 2))
    def sample_batch(self, rng_key, samples):

        def pack_sample(params):
            return np.array(
                [
                    params.sources_x,
                    params.sources_y,
                    params.sources_b,
                    params.mask * params.mu
                ]
            )

        def sampler_kernel(i, key, head, log_joint_head, chain, acceptances_i, moves_i, log_alphas_i):
            key1, key2, key3 = random.split(key, 3)
            move_type = NuSTARSampler.__get_move_type(key1, self.jump_rate, self.hyper_rate)
            sample_new, log_proposal_ratio, sample_log_joint = NuSTARSampler.__cond_proposal(
                key2, head, self.model,  self.proposal_width_xy, self.proposal_width_b, self.proposal_width_split, self.proposal_width_mu, move_type
            )
            log_alpha = sample_log_joint - log_joint_head + log_proposal_ratio
            accept = np.log(random.uniform(key3, minval=0, maxval=1)) < log_alpha
            new_head = lax.cond(accept, sample_new, lambda x: x, head, lambda x: x)
            new_log_joint = np.where(accept, sample_log_joint, log_joint_head)
            chain = index_update(chain, i, pack_sample(new_head))
            acceptances_i = index_update(acceptances_i, i, accept)
            moves_i = index_update(moves_i, i, move_type)
            log_alphas_i = index_update(log_alphas_i, i, log_alpha)
            return new_head, new_log_joint, chain, acceptances_i, moves_i, log_alphas_i

        def next_state(i, state):
            key, head, log_joint_head, chain, acceptances_i, moves_i, log_alphas_i = state
            _, key = random.split(key)
            new_head, new_log_joint, chain, acceptances_i, moves_i, log_alphas_i = sampler_kernel(
                i, key, head, log_joint_head, chain, acceptances_i, moves_i, log_alphas_i
            )
            return key, new_head, new_log_joint, chain, acceptances_i, moves_i, log_alphas_i

        all_samples = np.zeros((samples, 4, N_MAX))
        acceptances = np.zeros(samples)
        moves = np.zeros(samples)
        log_alphas = np.zeros(samples)
        state_init = (rng_key, self.head, self.log_joint_head, all_samples, acceptances, moves, log_alphas)
        rng_key, final_sample, final_log_joint, all_samples, acceptances, moves, log_alphas = lax.fori_loop(
            0, samples, next_state, state_init
        )
        return final_sample, final_log_joint, all_samples, acceptances, moves, np.exp(log_alphas)

    def sample_with_burnin(self):
        self.run_sampler(burn_in=True)
        self.run_sampler(burn_in=False)
        print("Done!")

    def __collect_stats(self, acceptances, moves, alphas):
        pass

    def __combine_samples(self, chains):
        pass

    def run_sampler(self, burn_in=False):
        if burn_in:
            batches = self.burn_in_steps // self.sample_batch_size
            print(f"Burning in for {batches * self.sample_batch_size} steps:")
        else:
            # sampling
            batches = self.samples // self.sample_batch_size
            print(f"Sampling for {batches * self.sample_batch_size} steps:")

        chains, acceptances, all_moves, all_alphas = [], [], [], []
        t = tqdm(total=(batches * self.sample_batch_size))
        for batch_i in range(batches):
            self.rng_key, key = random.split(self.rng_key)
            head, log_joint_head, chain, accepts, moves, alphas = self.sample_batch(key, self.sample_batch_size)
            self.head, self.log_joint_head = head, log_joint_head
            if not burn_in:
                chains.append(chain)
            acceptances.append(accepts)
            all_moves.append(moves)
            all_alphas.append(alphas)
            t.update(self.sample_batch_size)
        print("Recording stats from run...")
        self.__collect_stats(acceptances, all_moves, all_alphas)
        if not burn_in:
            print("Gathering posterior samples...")
            self.__combine_samples(chains)




    def sample2(self):
        accepted_recently = 0
        for i in tqdm(range(self.samples)):
            self.rng_key, sub_key = random.split(self.rng_key)
            if (i+1) % self.period == 0:
                self.acceptance_rates_per_period.append(accepted_recently/self.period)
                accepted_recently = 0
            move_type = self.__get_move_type()
            self.rng_key, key = random.split(self.rng_key)
            sample_new, log_proposal_ratio, sample_log_joint = self.__cond_proposal(key, self.head, move_type)
            log_alpha = sample_log_joint - self.log_joint_head + log_proposal_ratio
            accept = np.log(random.uniform(sub_key, minval=0, maxval=1)) < log_alpha
            if accept:
                self.head = sample_new
                self.log_joint_head = sample_log_joint
                self.accepted += 1
                accepted_recently += 1
            if i > self.burn_in:
                self.chain.append(self.head)
                n = self.head.sources_x.shape[0]
                self.n_sources_counts[n] = self.n_sources_counts.get(n, 0) + 1
            self.__record_move(move_type, np.exp(log_alpha), accept)

    def chain(self):
        return self.chain

    def stats(self):
        return OrderedDict(
            {
                PROPOSED: self.burn_in + self.samples,
                ACCEPTED: self.accepted,
                ACCEPTANCE_RATE: self.accepted / (self.burn_in + self.samples),
                STATS_BY_MOVE: self.move_stats,
                N_SOURCES_COUNTS: self.n_sources_counts
            }
        )
