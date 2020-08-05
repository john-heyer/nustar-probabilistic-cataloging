import sys
from collections import OrderedDict, Counter
from functools import partial

import jax.numpy as np
import numpy as onp
from jax import jit, vmap, random, scipy, lax, pmap, device_count
from jax.ops import index_update
from tqdm import tqdm

from mcmc_configs import XY_MAX, XY_MIN, N_MIN, N_MAX, B_MIN, B_MAX, WINDOW_SCALE
from model import ParameterSample, NuSTARModel
from nustar_constants import *
from utils import random_sources, random_source
from psrf import compute_psrf

onp.random.seed(3)  # for drawing poisson numbers


class NuSTARSampler:

    def __init__(
            self, model, rng_key, params_init=None, burn_in_steps=0, samples=10000, use_power_law=True, up_sample=4,
            birth_death_rate=.05, split_merge_rate=.05, hyper_rate=.02,
            proposal_width_xy=.75, proposal_width_b=2.0, proposal_width_mu=1.0, proposal_width_split=200,
            sample_batch_size=1000, description="", n_chains=1, compute_psrf=False, sample_interval=500
    ):
        self.model = model
        self.rng_key = rng_key
        self.n_chains = n_chains

        # initial params
        if params_init is None:
            # self.head = [self.random_params(key) for key in random.split(self.rng_key, self.n_chains)]
            self.head = self.random_params_p(self.rng_key, self.n_chains)
        else:
            self.head = [params_init for _ in range(self.n_chains)]

        NuSTARSampler.check_parameter_invariants(self.head, self.n_chains)

        self.log_joint_head = vmap(model.log_joint)(self.head)

        # sampler configs
        self.burn_in_steps = burn_in_steps
        self.samples = samples
        self.use_power_law = use_power_law
        self.up_sample = up_sample
        self.sample_batch_size = sample_batch_size
        self.birth_death_rate = birth_death_rate
        self.split_merge_rate = split_merge_rate
        self.hyper_rate = hyper_rate
        self.proposal_width_xy = proposal_width_xy
        self.proposal_width_b = proposal_width_b
        self.proposal_width_mu = proposal_width_mu
        self.proposal_width_split = proposal_width_split

        # stats
        self.move_stats = None
        self._init_move_stats()
        self.proposals = 0
        self.accepted = 0
        self.batch_acceptance_rates = []
        self.description = description
        self.compute_psrf = compute_psrf
        self.sample_interval = sample_interval
        self.r_hat = None

        # posterior
        self.source_posterior = onp.zeros((0, 3, N_MAX))  # final shape should be (samples, 3, N_MAX)
        self.mu_posterior = Counter()
        self.n_posterior = Counter()

    @staticmethod
    def check_parameter_invariants(params, n_chains):
        # only used for testing as assertions will not compile with jax
        for i in range(n_chains):
            assert params.n[i] == np.sum(params.mask[i]), "parameter n not matching mask"
            assert np.all(params.mask[i] == (np.arange(N_MAX) < params.n[i])), \
                "mask is not an array of ones with zero padding"
            assert np.all(params.sources_x[i][np.logical_not(params.mask[i])] == 0), "outside mask x values not zero"
            assert np.all(params.sources_y[i][np.logical_not(params.mask[i])] == 0), "outside mask y values not zero"
            assert np.all(params.sources_b[i][np.logical_not(params.mask[i])] == 0), "outside mask b values not zero"
            print(f'Chain {i + 1} mu init: {params.mu[i]}')
            print(f'Chain {i + 1} n init: {params.n[i]}')

    @staticmethod
    def random_params_p(rng_key, n_chains):
        # will not compile as it relies on original numpy for poisson samples
        s_x, s_y, s_b, msk, mus, ns = [], [], [], [], [], []
        for i in range(n_chains):

            rng_key, sub_key, sub_key_2 = random.split(rng_key, 3)
            mu_init = np.exp(random.uniform(sub_key, minval=np.log(N_MIN), maxval=np.log(N_MAX)))
            n_init = 0
            while n_init < N_MIN or n_init > N_MAX:
                n_init = onp.random.poisson(mu_init)
            sources_x_init, sources_y_init, sources_b_init = random_sources(sub_key_2, n_init)
            pad = np.zeros(N_MAX - n_init)
            s_x.append(np.hstack((sources_x_init, pad))),
            s_y.append(np.hstack((sources_y_init, pad))),
            s_b.append(np.hstack((sources_b_init, pad))),
            msk.append(np.arange(N_MAX) < n_init),
            mus.append(mu_init)
            ns.append(n_init)

        return ParameterSample(
            sources_x=np.array(s_x),
            sources_y=np.array(s_y),
            sources_b=np.array(s_b),
            mask=np.array(msk),
            mu=np.array(mus),
            n=np.array(ns),
        )

    @staticmethod
    def random_params(rng_key):
        # will not compile as it relies on original numpy for poisson samples
        sub_key, sub_key_2 = random.split(rng_key, 2)
        mu_init = np.exp(random.uniform(sub_key, minval=np.log(N_MIN), maxval=np.log(N_MAX)))
        n_init = onp.random.poisson(mu_init)
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

    def _init_move_stats(self):
        move_stats = OrderedDict()
        for move in MOVES:
            move_stats[MOVES[move]] = OrderedDict(
                {
                    PROPOSED: 0,
                    ACCEPTED: 0,
                    ZERO_MOVES: 0,
                    INF_MOVES: 0
                }
            )
        self.move_stats = move_stats

    @partial(jit, static_argnums=(0,))
    def _get_move_type(self, rng_key):
        normal_rate = 1 - self.birth_death_rate - self.split_merge_rate - self.hyper_rate
        bd_rate = self.birth_death_rate/2
        sm_rate = self.split_merge_rate/2
        logits = np.log(np.array([normal_rate, bd_rate, bd_rate, sm_rate, sm_rate, self.hyper_rate]))
        return random.categorical(rng_key, logits)

    @partial(jit, static_argnums=(0,))
    def birth(self, rng_key, params):
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
        return sample_new, log_proposal_ratio #- np.log(sample_new.n)

    @staticmethod
    def remove_and_shift(arr, i_remove):
        unmoved = arr * (np.arange(arr.shape[0]) < i_remove)
        to_shift = arr * (np.arange(arr.shape[0]) > i_remove)
        rolled_left = np.roll(to_shift, -1)
        return unmoved + rolled_left

    @partial(jit, static_argnums=(0,))
    def death(self, rng_key, params):
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
        return sample_new, log_proposal_ratio #+ np.log(params.n)

    @staticmethod
    def utr_idx(N, i, j):
        return N * i - (i * (i + 1)) // 2 + (j - i - 1)

    @staticmethod
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

    @partial(jit, static_argnums=(0,))
    def split(self, rng_key, params):
        sigma_split = self.proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0/params.n)
        key1, key2, key3 = random.split(rng_key, 3)
        i_source = random.randint(key1, shape=(), minval=0, maxval=params.n)
        alpha = random.uniform(key2, minval=0, maxval=1)
        q_x, q_y = random.normal(key3, shape=(2,)) * sigma_split
        x, y, b = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]

        # preserves center of mass
        new_x1, new_x2 = x + q_x * (1 - alpha), x - q_x * alpha
        new_y1, new_y2 = y + q_y * (1 - alpha), y - q_y * alpha
        new_b1, new_b2 = b * alpha, b * (1-alpha)

        def split_padded_sources(arr, i_remove, n, new1, new2):
            source_removed = NuSTARSampler.remove_and_shift(arr, i_remove)
            mask_new1 = np.arange(N_MAX) == (n-1)
            mask_new2 = np.arange(N_MAX) == n
            return source_removed + (new1 * mask_new1) + (new2 * mask_new2)

        sample_new = ParameterSample(
            sources_x=split_padded_sources(params.sources_x, i_source, params.n, new_x1, new_x2),
            sources_y=split_padded_sources(params.sources_y, i_source, params.n, new_y1, new_y2),
            sources_b=split_padded_sources(params.sources_b, i_source, params.n, new_b1, new_b2),
            mask=(np.arange(N_MAX) < (params.n + 1)),
            mu=params.mu,
            n=(params.n + 1)
        )
        # uniform split merge distribution
        log_p_merge_to_p_split = np.log(1/(params.n + 1))  # p_m/p_s = 1/(n*(n+1)) / (1/n) = 1/(n+1)
        log_det_J = np.log(b)
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_proposal_ratio = log_p_merge_to_p_split - log_p_q + log_det_J  # = (p_m * J) / (p_s * p_q)
        """
        # full proposal ratio: p(inverse_merge) / p(split) * |J| = b
        distance_dist = NuSTARSampler.distance_distribution(sample_new)
        pair_idx = NuSTARSampler.utr_idx(N_MAX, params.n-1, params.n)
        n_c_2 = (sample_new.n ** 2 - sample_new.n) / 2
        log_p_merge = np.log(2) #- np.log(n_c_2) #np.log(distance_dist[pair_idx][2])
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        # print('sigma split', sigma_split)
        # print('qx, qy', q_x, q_y)
        # print('log p q', log_p_q)
        # print('log q 0', scipy.stats.norm.logpdf(0, scale=sigma_split))
        log_p_split = log_p_q #- np.log(params.n)  # p(q) * 1/n for choice of source
        log_proposal_ratio = log_p_merge - log_p_split + np.log(b)
        """  # TODO
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def merge(self, rng_key, params):
        # distance_dist = NuSTARSampler.distance_distribution(params)
        # logits = np.log(distance_dist.T[2])
        # i_pair = random.categorical(rng_key, logits)
        # i_source, j_source, p_merge = distance_dist[i_pair]
        # i_source, j_source = i_source.astype(np.int32), j_source.astype(np.int32)

        # uniform split merge distribution
        key1, key2 = random.split(rng_key, 2)
        i_source = random.randint(key1, shape=(), minval=0, maxval=params.n)
        j_source = random.randint(key2, shape=(), minval=0, maxval=params.n-1)
        j_source += (i_source == j_source)

        x_i, y_i, b_i = params.sources_x[i_source], params.sources_y[i_source], params.sources_b[i_source]
        x_j, y_j, b_j = params.sources_x[j_source], params.sources_y[j_source], params.sources_b[j_source]
        q_x, q_y = (x_i - x_j), (y_i - y_j)
        alpha = b_i / (b_i + b_j)
        x_new, y_new = x_i - (q_x * (1 - alpha)), y_i - (q_x * (1 - alpha))
        b_new = b_i + b_j

        def merge_padded_sources(arr, i_remove, j_remove, n, new):
            i_remove, j_remove = np.sort(np.array([i_remove, j_remove]))  # ensure j > i
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
        new_n = sample_new.n
        sigma_split = self.proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0 / new_n)
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split_to_p_merge = np.log(params.n)  # p_s/p_m = 1/(n-1) / (1/(n*(n-1)) = n
        log_det_J = np.log(b_new)
        log_proposal_ratio = log_p_split_to_p_merge + log_p_q - log_det_J  # = (p_s * p_q) / (p_m * J)
        """
        # full proposal ratio: p(inverse_split) / p(merge) * 1/|J| = 1/b
        new_n = sample_new.n
        sigma_split = self.proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0 / new_n)
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        # print('sigma split', sigma_split)
        # print('qx, qy', q_x, q_y)
        # print('log p q', log_p_q)
        log_p_split = log_p_q #- np.log(new_n)
        n_c_2 = (params.n**2 - params.n)/2
        log_p_merge = np.log(2) #- np.log(n_c_2) #np.log(p_merge)
        log_proposal_ratio = log_p_split - log_p_merge - np.log(b_new)
        """  # TODO
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def _birth_death(self, tup_args):
        rng_key, params, move_type = tup_args
        up = (move_type == BIRTH)
        sample_new, log_proposal_ratio = lax.cond(
            up,
            (rng_key, params),
            lambda args: self.birth(*args),
            (rng_key, params),
            lambda args: self.death(*args)
        )
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def _split_merge(self, tup_args):
        rng_key, params, move_type = tup_args
        up = (move_type == SPLIT)
        sample_new, log_proposal_ratio = lax.cond(
            up,
            (rng_key, params),
            lambda args: self.split(*args),
            (rng_key, params),
            lambda args: self.merge(*args)
        )
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def _jump_proposal(self, tup_args):
        rng_key, params, move_type = tup_args
        birth_death = ((move_type == BIRTH) | (move_type == DEATH))
        sample_new, log_proposal_ratio = lax.cond(
            birth_death,
            (rng_key, params, move_type),
            self._birth_death,
            (rng_key, params, move_type),
            self._split_merge
        )
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def normal_proposal(self, rng_key, params):
        key1, key2, key3 = random.split(rng_key, 3)
        sigma_xy = self.proposal_width_xy * PSF_PIXEL_SIZE * np.sqrt(1.0/params.n)
        sigma_b = self.proposal_width_b * np.sqrt(1.0/params.n)
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

    @partial(jit, static_argnums=(0,))
    def hyper_proposal(self, rng_key, params):
        sample_new = ParameterSample(
            sources_x=params.sources_x,
            sources_y=params.sources_y,
            sources_b=params.sources_b,
            mask=params.mask,
            mu=params.mu + (random.normal(rng_key) * self.proposal_width_mu * np.sqrt(params.n)),
            n=params.n
        )
        log_proposal_ratio = 0.0
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def _normal_hyper(self, tup_args):
        rng_key, params, move_type = tup_args
        normal = (move_type == NORMAL)
        sample_new, log_proposal_ratio = lax.cond(
            normal,
            (rng_key, params),
            lambda args: self.normal_proposal(*args),
            (rng_key, params),
            lambda args: self.hyper_proposal(*args)
        )
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def _cond_proposal(self, rng_key, params, move_type):
        jump = ((move_type != NORMAL) & (move_type != HYPER))
        sample_new, log_proposal_ratio = lax.cond(
            jump,
            (rng_key, params, move_type),
            self._jump_proposal,
            (rng_key, params, move_type),
            self._normal_hyper
        )
        sample_log_joint = self.model.log_joint(sample_new)
        return sample_new, log_proposal_ratio, sample_log_joint

    @partial(jit, static_argnums=(0, 4))
    def sample_batch(self, rng_key, head, log_joint_head, samples):

        def pack_sample(params):
            return np.array(
                [
                    params.sources_x,
                    params.sources_y,
                    params.sources_b,
                ]
            )

        def sampler_kernel(i, key, head, log_joint_head, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i):
            key1, key2, key3 = random.split(key, 3)
            move_type = self._get_move_type(key1)
            sample_new, log_proposal_ratio, sample_log_joint = self._cond_proposal(key2, head, move_type)

            # if move_type == SPLIT:  # (for debugging particular moves, disable jit; currently fails due to vmap)  TODO

            log_alpha = sample_log_joint - log_joint_head + log_proposal_ratio
            accept = np.log(random.uniform(key3, minval=0, maxval=1)) < log_alpha
            new_head = lax.cond(accept, sample_new, lambda x: x, head, lambda x: x)
            new_log_joint = np.where(accept, sample_log_joint, log_joint_head)
            chain = index_update(chain, i, pack_sample(new_head))
            mus_i = index_update(mus_i, i, new_head.mu)
            ns_i = index_update(ns_i, i, new_head.n)
            acceptances_i = index_update(acceptances_i, i, accept)
            moves_i = index_update(moves_i, i, move_type)
            log_alphas_i = index_update(log_alphas_i, i, log_alpha)
            return new_head, new_log_joint, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i

        def next_state(i, state):
            key, head, log_joint_head, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i = state
            _, key = random.split(key)
            new_head, new_log_joint, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i = sampler_kernel(
                i, key, head, log_joint_head, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i
            )
            return key, new_head, new_log_joint, chain, mus_i, ns_i, acceptances_i, moves_i, log_alphas_i

        all_samples = np.zeros((samples, 3, N_MAX))
        all_mus = np.zeros(samples)
        all_ns = np.zeros(samples)
        acceptances = np.zeros(samples)
        moves = np.zeros(samples)
        log_alphas = np.zeros(samples)
        state_init = (
            rng_key, head, log_joint_head, all_samples, all_mus, all_ns, acceptances, moves, log_alphas
        )

        rng_key, final_sample, final_log_joint, all_samples, all_mus, all_ns, acceptances, moves, log_alphas = \
            lax.fori_loop(0, samples, next_state, state_init)

        # equivalently (for debugging without jit):
        # state = state_init
        # for i in range(samples):
        #     state = next_state(i, state)
        # rng_key, final_sample, final_log_joint, all_samples, all_mus, all_ns, acceptances, moves, log_alphas = state

        return final_sample, final_log_joint, all_samples, all_mus, all_ns, acceptances, moves, np.exp(log_alphas)

    def sample_with_burn_in(self):
        print()
        if self.burn_in_steps//self.sample_batch_size > 0:
            self.run_sampler(burn_in=True)
        self.run_sampler(burn_in=False)
        print("Done!")

    def _collect_stats(self, batch_size, acceptances, moves, alphas):
        acceptances_arr = np.array(acceptances)
        moves_arr = np.array(moves)
        alpha_arr = np.array(alphas)

        @partial(jit, static_argnums=(0, 1))
        def compile_stats(batch_size, n_moves, moves, acceptances, alphas):
            batch_accepts = np.sum(acceptances, axis=1) / batch_size
            all_acceptances = np.hstack(acceptances)
            all_moves = np.hstack(moves)
            all_alphas = np.hstack(alphas)

            def stats_by_move(move, moves, accepts, alphas):
                proposals = np.sum(moves == move)
                accepts = np.sum(np.where(moves == move, accepts, np.zeros(accepts.shape[0])))
                move_alphas = np.where(moves == move, alphas, np.ones(alphas.shape[0]))
                zeros = np.sum(move_alphas == 0)
                infs = np.sum(move_alphas == np.inf)
                return (
                    proposals.astype(np.int32), accepts.astype(np.int32), zeros.astype(np.int32), infs.astype(np.int32)
                )

            all_move_stats = vmap(stats_by_move, in_axes=(0, None, None, None))
            move_stats = all_move_stats(np.arange(n_moves), all_moves, all_acceptances, all_alphas)

            # return total proposals, acceptances, acceptances by batch, stats by move
            return all_moves.shape[0], np.sum(all_acceptances), batch_accepts, move_stats

        total_proposals, total_accepts, batch_acceptance_rates, move_stats = compile_stats(
            batch_size, len(MOVES), moves_arr, acceptances_arr, alpha_arr
        )

        proposals_by_move, accepts_by_move, zeros_by_move, infs_by_move = move_stats
        for move in MOVES:
            self.move_stats[MOVES[move]][PROPOSED] += proposals_by_move[move].item()
            self.move_stats[MOVES[move]][ACCEPTED] += accepts_by_move[move].item()
            self.move_stats[MOVES[move]][ZERO_MOVES] += zeros_by_move[move].item()
            self.move_stats[MOVES[move]][INF_MOVES] += infs_by_move[move].item()

        self.proposals += total_proposals
        self.accepted += total_accepts.item()
        self.batch_acceptance_rates += list(batch_acceptance_rates)

    def _combine_samples(self, chains, mus, ns):
        source_posterior = onp.concatenate(chains, axis=0)
        unique_mus, counts_mu = onp.unique(mus, return_counts=True)
        unique_ns, counts_n = onp.unique(ns, return_counts=True)

        self.source_posterior = onp.concatenate((self.source_posterior, source_posterior), axis=0)
        self.mu_posterior += Counter(dict(zip(unique_mus, counts_mu)))
        self.n_posterior += Counter(dict(zip(unique_ns, counts_n)))

    def _compute_psrf(self, psrf_samples):
        psrf_samples = np.concatenate(psrf_samples, axis=1)  # shape = (n_chains, burn_in/sample_interval, 3, N_MAX)
        if self.use_power_law:
            v_mean_map = vmap(
                vmap(
                    lambda sample: self.model.mean_emission_map_power_law(sample[0], sample[1], sample[2]), in_axes=(0,)
                ), in_axes=(0,)
            )
        else:
            v_mean_map = vmap(
                vmap(
                    lambda sample: self.model.mean_emission_map(sample[0], sample[1], sample[2], self.up_sample),
                    in_axes=(0,)
                ),
                in_axes=(0,)
            )
        map_tensor = v_mean_map(psrf_samples)
        map_tensor = np.transpose(map_tensor, axes=(1, 0, 2, 3))
        r_hat = compute_psrf(map_tensor)
        self.r_hat = onp.array(r_hat)

    def run_sampler(self, burn_in=False):
        # Note: can be used to continue sampling
        if burn_in:
            batch_size = self.sample_batch_size
            batches = self.burn_in_steps // batch_size
            print(f"Burning in for {batches * self.sample_batch_size} steps:")
        else:
            # sampling
            batch_size = self.sample_batch_size // self.n_chains
            batches = (self.samples // self.n_chains) // batch_size
            print(f"Sampling for {batches * self.sample_batch_size} steps:")

        chains, all_mus, all_ns, acceptances, all_moves, all_alphas, psrf_samples = [], [], [], [], [], [], []
        t = tqdm(total=(batches * self.sample_batch_size), file=sys.stdout)

        if device_count() < self.n_chains or device_count() == 1:
            p_batch = vmap(
                self.sample_batch,
                in_axes=(0, 0, 0, None),
            )
            p_batch = jit(p_batch, static_argnums=(3,))
        else:
            p_batch = pmap(
                self.sample_batch,
                in_axes=(0, 0, 0, None),
                static_broadcasted_argnums=(3,)
            )
        for batch_i in range(batches):
            self.rng_key, *keys = random.split(self.rng_key, self.n_chains + 1)
            head, log_joint_head, chain, mus, ns, accepts, moves, alphas = p_batch(
                np.array(keys), self.head, self.log_joint_head, batch_size
            )
            self.head, self.log_joint_head = head, log_joint_head
            if not burn_in:
                chains.append(np.concatenate(chain, axis=0))
                all_mus.append(mus.flatten())
                all_ns.append(ns.flatten())
            if burn_in and self.compute_psrf:
                psrf_samples.append(chain[:, 0::self.sample_interval, :, :])

            acceptances.append(accepts.flatten())
            all_moves.append(moves.flatten())
            all_alphas.append(alphas.flatten())
            t.update(batch_size * (self.n_chains if not burn_in else 1))
        t.close()
        print("Recording stats from run...")
        self._collect_stats(batch_size * self.n_chains, acceptances, all_moves, all_alphas)
        if burn_in and self.compute_psrf:
            print("Computing psrf...")
            self._compute_psrf(psrf_samples)
        if not burn_in:
            print("Gathering posterior samples...")
            self._combine_samples(chains, all_mus, all_ns)

    def get_posterior_sources(self):
        return self.source_posterior

    def get_stats(self):
        return OrderedDict(
            {
                DESCRIPTION: self.description,
                PROPOSED: self.proposals,
                ACCEPTED: self.accepted,
                BURN_IN: self.burn_in_steps,
                BATCH_SIZE: self.sample_batch_size,
                ACCEPTANCE_RATE: self.accepted / self.proposals,
                STATS_BY_MOVE: self.move_stats,
                BATCH_ACCEPTANCE_RATES: self.batch_acceptance_rates,
                N_POSTERIOR: self.n_posterior,
                MU_POSTERIOR: self.mu_posterior,
                R_HAT: self.r_hat,
                WIND_SCALE_STR: WINDOW_SCALE,
            }
        )


if __name__ == '__main__':
    from mcmc_configs import N_SOURCES_TRUTH
    key = random.PRNGKey(0)
    key, sub_key = random.split(key)

    # generate ground truth observation
    sources_xt, sources_yt, sources_bt = random_sources(sub_key, N_SOURCES_TRUTH)
    mean_image = NuSTARModel.mean_emission_map(sources_xt, sources_yt, sources_bt)
    observed_image = NuSTARModel.sample_image(mean_image)

    pad = np.zeros(N_MAX - N_SOURCES_TRUTH)
    params = ParameterSample(
        sources_x=np.hstack((sources_xt, pad)),
        sources_y=np.hstack((sources_yt, pad)),
        sources_b=np.hstack((sources_bt, pad)),
        mask=(np.arange(N_MAX) < N_SOURCES_TRUTH),
        n=N_SOURCES_TRUTH,
        mu=float(N_SOURCES_TRUTH),
    )

    model = NuSTARModel(observed_image)
    sampler = NuSTARSampler(model, key)
    print('split')
    new, ratio = sampler.split(key, params)
    print(ratio)
    print('merge')
    new, ratio = sampler.merge(key, params)
    print(ratio)
