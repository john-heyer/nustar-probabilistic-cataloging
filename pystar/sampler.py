from nustar_constants import *
from mcmc_configs import *
from utils import random_sources, random_source
from model import NuSTARModel, ParameterSample

from collections import OrderedDict
from enum import Enum
from tqdm import tqdm

import jax.numpy as np
from jax import grad, jit, vmap, random, scipy, lax
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
            self, model, rng_key, params_init=None, burn_in=0, samples=1000, jump_rate=.1, hyper_rate=.02, period=1000,
            proposal_width_xy=.75, proposal_width_b=2.0, proposal_width_mu=1.0, proposal_width_split=200
    ):
        self.model = model
        self.rng_key = rng_key

        # initial params
        if params_init is None:
            self.head = self.__init_params(self.rng_key)
        else:
            self.head = params_init
        self.log_joint_head = model.log_joint(self.head)
        self.mean_emission_map = NuSTARModel.mean_emission_map(
            self.head.sources_x, self.head.sources_y, self.head.sources_b
        )
        self.chain = [self.head]

        # sampler configs
        self.burn_in = burn_in
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
        self.period = period
        self.acceptance_rates_per_period = []
        self.n_sources_counts = {}

        # presample step
        # self.__cache_jits()

    def __cache_jits(self):
        start = timer()
        rng_key = random.PRNGKey(0)
        for n in range(180, 200):
            rng_key, key = random.split(rng_key)
            sources_x, sources_y, sources_b = random_sources(key, n)
            params = ParameterSample(
                sources_x=sources_x,
                sources_y=sources_y,
                sources_b=sources_b,
                mu=200
            )
            for move in MOVES:
                self.__proposal(key, params, move)
        end = timer()
        print("cache jits time:", end - start)


    def __init_params(self, rng_key):
        sub_key, sub_key_2 = random.split(rng_key, 2)
        mu_init = np.exp(random.uniform(sub_key, minval=np.log(N_MIN), maxval=np.log(N_MAX)))
        print(mu_init)
        n_init = nprand.poisson(mu_init)
        sources_x_init, sources_y_init, sources_b_init = random_sources(sub_key_2, n_init)
        return ParameterSample(sources_x_init, sources_y_init, sources_b_init, mu_init)

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

    def __get_move_type(self):
        self.rng_key, key1, key2, key3 = random.split(self.rng_key, 4)
        r = random.uniform(key1, minval=0, maxval=1)
        if r < self.hyper_rate:
            return HYPER

        elif r < (self.jump_rate + self.hyper_rate):
            split_merge = random.uniform(key2, minval=0, maxval=1) < 0.5
            up = random.uniform(key3, minval=0, maxval=1) < 0.5
            if split_merge:
                if up:
                    return SPLIT
                else:
                    return MERGE
            else:
                if up:
                    return BIRTH
                else:
                    return DEATH

        return NORMAL

    @staticmethod
    @jit
    def birth(rng_key, params):
        source_x, source_y, source_b = random_source(rng_key)
        log_proposal_ratio = - (2 * np.log(XY_MAX - XY_MIN) + np.log(B_MAX - B_MIN))
        sample_new = ParameterSample(
            sources_x=np.hstack((params.sources_x, source_x)),
            sources_y=np.hstack((params.sources_y, source_y)),
            sources_b=np.hstack((params.sources_b, source_b)),
            mu=params.mu
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def death(rng_key, params):
        n = params.sources_x.shape[0]
        i_source = random.randint(rng_key, shape=(), minval=0, maxval=2)
        log_proposal_ratio = (2 * np.log(XY_MAX - XY_MIN) + np.log(B_MAX - B_MIN))
        indices = np.arange(n - 1) + (np.arange(n - 1) >= i_source)
        sample_new = ParameterSample(
            sources_x=params.sources_x[indices],
            sources_y=params.sources_y[indices],
            sources_b=params.sources_b[indices],
            mu=params.mu
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def distance_distribution(params):
        def distance(i, j):
            return (
                (params.sources_x[i] - params.sources_x[j]) ** 2 +
                (params.sources_y[i] - params.sources_y[j]) ** 2
            )
        n = params.sources_x.shape[0]
        sources_one, sources_two = np.triu_indices(n, 1)
        pairwise_distances = vmap(distance, in_axes=(0, 0))(sources_one, sources_two)
        distance_dist = pairwise_distances/(np.sum(pairwise_distances))
        return np.vstack((sources_one, sources_two, distance_dist)).T

    @staticmethod
    @jit
    def split(rng_key, params, proposal_width_split):
        n = params.sources_x.shape[0]
        sigma_split = proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0/n)
        key1, key2, key3 = random.split(rng_key, 3)
        i_source = random.randint(key1, shape=(), minval=0, maxval=n)
        alpha = random.uniform(key2, minval=0, maxval=1)
        q_x, q_y = random.normal(key3, shape=(2,)) * sigma_split
        x, y, b = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]
        new_x = [x + q_x/2, x - q_x/2]
        new_y = [y + q_y/2, y - q_y/2]
        new_b = [b * alpha, b * (1-alpha)]
        indices = np.arange(n - 1) + (np.arange(n - 1) >= i_source)
        sample_new = ParameterSample(
            sources_x=np.hstack((params.sources_x[indices], new_x)),
            sources_y=np.hstack((params.sources_y[indices], new_y)),
            sources_b=np.hstack((params.sources_b[indices], new_b)),
            mu=params.mu
        )
        # full proposal ratio: p(inverse_merge) / p(split) * |J| = b
        distance_dist = NuSTARSampler.distance_distribution(sample_new)
        log_p_merge = np.log(distance_dist[-1][2])  # new sources will be last pair in the distance distribution
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split = log_p_q - np.log(n)  # p(q) * 1/n for choice of source
        log_proposal_ratio = log_p_merge - log_p_split + np.log(b)
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def merge(rng_key, params, proposal_width_split):
        distance_dist = NuSTARSampler.distance_distribution(params)
        cumulative_dist = np.cumsum(distance_dist[:, 2])
        v = random.uniform(rng_key, minval=0, maxval=1)
        i_pair = np.sum(np.less(cumulative_dist, v))
        i_source, j_source, p_merge = distance_dist[i_pair]
        i_source, j_source = i_source.astype(np.int32), j_source.astype(np.int32)
        x_i, y_i, b_i = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]
        x_j, y_j, b_j = params.sources_x[j_source], params.sources_y[j_source], params.sources_b[j_source]
        q_x, q_y = (x_i - x_j), (y_i - y_j)
        x_new, y_new = (x_i + x_j)/2, (y_i + y_j)/2
        b_new = b_i + b_j
        n = params.sources_x.shape[0]
        indices = np.arange(n-2) + (np.arange(n-2) >= i_source) + ((np.arange(n-2) + 1) >= j_source)
        sample_new = ParameterSample(
            sources_x=np.hstack((params.sources_x[indices], x_new)),
            sources_y=np.hstack((params.sources_y[indices], y_new)),
            sources_b=np.hstack((params.sources_b[indices], b_new)),
            mu=params.mu
        )
        # full proposal ratio: p(inverse_split) / p(merge) * 1/|J| = 1/b
        new_n = sample_new.sources_x.shape[0]
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
        split_fn = lambda args: NuSTARSampler.split(*args)
        merge_fn = lambda args: NuSTARSampler.split(*args)
        sample_new, log_proposal_ratio = lax.cond(
            up,
            (rng_key, params, proposal_width_split),
            split_fn,
            (rng_key, params, proposal_width_split),
            merge_fn
        )
        return sample_new, log_proposal_ratio

    @staticmethod
    def __cond_jump_proposal(rng_key, params, move_type, proposal_width_split):
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
    def __jump_proposal(rng_key, params, move_type, proposal_width_split):
        if move_type == Move.BIRTH:
            start = timer()
            sample_new, log_proposal_ratio = NuSTARSampler.birth(rng_key, params)
            end = timer()
            print("birth time", end - start)
        elif move_type == Move.DEATH:
            start = timer()
            sample_new, log_proposal_ratio = NuSTARSampler.death(rng_key, params)
            end = timer()
            print("deatg time", end - start)
        elif move_type == Move.SPLIT:
            start = timer()
            sample_new, log_proposal_ratio = NuSTARSampler.split(rng_key, params, proposal_width_split)
            end = timer()
            print("split time", end - start)
        else:
            start = timer()
            sample_new, log_proposal_ratio = NuSTARSampler.merge(rng_key, params, proposal_width_split)
            end = timer()
            print("merge time", end - start)
        return sample_new, log_proposal_ratio

    @staticmethod
    @jit
    def normal_proposal(rng_key, params, proposal_width_xy, proposal_width_b):
        key1, key2, key3 = random.split(rng_key, 3)
        n = params.sources_x.shape[0]
        sigma_xy = proposal_width_xy * PSF_PIXEL_SIZE * np.sqrt(1.0/n)
        sigma_b = proposal_width_b * np.sqrt(1.0/n)
        q_x = random.normal(key1, shape=(n,)) * sigma_xy
        q_y = random.normal(key2, shape=(n,)) * sigma_xy
        q_b = random.normal(key3, shape=(n,)) * sigma_b
        return ParameterSample(
            sources_x=params.sources_x + q_x,
            sources_y=params.sources_y + q_y,
            sources_b=params.sources_b + q_b,
            mu=params.mu
        )

    @staticmethod
    @jit
    def hyper_proposal(rng_key, params, proposal_width_mu):
        n = params.sources_x.shape[0]
        return ParameterSample(
            sources_x=params.sources_x,
            sources_y=params.sources_y,
            sources_b=params.sources_b,
            mu=params.mu + (random.normal(rng_key) * proposal_width_mu * np.sqrt(n))
        )



    def __proposal(self, key, params, move_type):
        if move_type == NORMAL:
            sample_new = NuSTARSampler.normal_proposal(key, params, self.proposal_width_xy, self.proposal_width_b)
            log_proposal_ratio = 0.0
            log_joint = self.model.log_joint(sample_new)
        elif move_type == HYPER:
            sample_new = NuSTARSampler.hyper_proposal(key, params, self.proposal_width_mu)
            log_proposal_ratio = 0.0
            log_joint = (
                    self.log_joint_head -
                    NuSTARModel.log_prior_mu(params.mu) +
                    NuSTARModel.log_prior_mu(sample_new.mu) -
                    NuSTARModel.log_prior_n(params.mu, params.sources_x.shape[0]) +
                    NuSTARModel.log_prior_n(sample_new.mu, sample_new.sources_x.shape[0])
            )
        else:
            # Jump move
            sample_new, log_proposal_ratio = NuSTARSampler.__cond_jump_proposal(key, params, move_type, self.proposal_width_split)
            log_joint = self.model.log_joint(sample_new)
            # log_joint = -np.inf

        return sample_new, log_proposal_ratio, log_joint

    def sample(self):
        accepted_recently = 0
        for i in tqdm(range(self.samples)):
            self.rng_key, sub_key = random.split(self.rng_key)
            if (i+1) % self.period == 0:
                self.acceptance_rates_per_period.append(accepted_recently/self.period)
                accepted_recently = 0
            move_type = self.__get_move_type()
            self.rng_key, key = random.split(self.rng_key)
            sample_new, log_proposal_ratio, sample_log_joint = self.__proposal(key, self.head, move_type)
            log_alpha = sample_log_joint - self.log_joint_head + log_proposal_ratio
            accept = np.log(random.uniform(sub_key, minval=0, maxval=1)) < log_alpha
            if accept:
                # if move_type != Move.HYPER and move_type != Move.NORMAL:
                    # print(sample_new.sources_x.shape)
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
