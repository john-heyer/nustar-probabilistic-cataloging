import sys
from collections import OrderedDict, Counter

import jax.numpy as np
import numpy as onp
from jax import jit, vmap, random, scipy, lax
from jax.ops import index_update
from tqdm import tqdm

from mcmc_configs import *
from model import ParameterSample
from nustar_constants import *
from utils import random_sources, random_source

onp.random.seed(1)  # for drawing poisson numbers

from functools import partial
from timeit import default_timer as timer


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
        NuSTARSampler.check_parameter_invariants(self.head)
        self.log_joint_head = model.log_joint(self.head)

        # sampler configs
        self.burn_in_steps = burn_in_steps
        self.samples = samples
        self.sample_batch_size = sample_batch_size
        self.jump_rate = jump_rate
        self.hyper_rate = hyper_rate
        self.proposal_width_xy = proposal_width_xy
        self.proposal_width_b = proposal_width_b
        self.proposal_width_mu = proposal_width_mu
        self.proposal_width_split = proposal_width_split

        # stats
        self.move_stats = None
        self.__init_move_stats()
        self.proposals = 0
        self.accepted = 0
        self.batch_acceptance_rates = []

        # posterior
        self.source_posterior = onp.zeros((0, 3, N_MAX))  # final shape should be (samples, 3, N_MAX)
        self.mu_posterior = Counter()
        self.n_posterior = Counter()

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

    def __init_move_stats(self):
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
    def __get_move_type(self, rng_key):
        normal_rate = 1 - self.jump_rate - self.hyper_rate
        jump_p = self.jump_rate / 4
        logits = np.log(np.array([normal_rate, jump_p, jump_p, jump_p, jump_p, self.hyper_rate]))
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
        return sample_new, log_proposal_ratio

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
        return sample_new, log_proposal_ratio

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
        # full proposal ratio: p(inverse_merge) / p(split) * |J| = b
        distance_dist = NuSTARSampler.distance_distribution(sample_new)
        pair_idx = NuSTARSampler.utr_idx(N_MAX, params.n-1, params.n)

        log_p_merge = np.log(distance_dist[pair_idx][2])  # new sources will be last pair in the distance distribution
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split = log_p_q - np.log(params.n)  # p(q) * 1/n for choice of source
        log_proposal_ratio = log_p_merge - log_p_split + np.log(b)
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def merge(self, rng_key, params):
        distance_dist = NuSTARSampler.distance_distribution(params)
        logits = np.log(distance_dist.T[2])
        i_pair = random.categorical(rng_key, logits)
        i_source, j_source, p_merge = distance_dist[i_pair]
        i_source, j_source = i_source.astype(np.int32), j_source.astype(np.int32)
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
        # full proposal ratio: p(inverse_split) / p(merge) * 1/|J| = 1/b
        new_n = sample_new.n
        sigma_split = self.proposal_width_split * PSF_PIXEL_SIZE * np.sqrt(1.0 / new_n)
        log_p_q = scipy.stats.norm.logpdf(q_x, scale=sigma_split) + scipy.stats.norm.logpdf(q_y, scale=sigma_split)
        log_p_split = log_p_q - np.log(new_n)
        log_p_merge = np.log(p_merge)
        log_proposal_ratio = log_p_split - log_p_merge - np.log(b_new)
        return sample_new, log_proposal_ratio

    @partial(jit, static_argnums=(0,))
    def __birth_death(self, tup_args):
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
    def __split_merge(self, tup_args):
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
    def __jump_proposal(self, tup_args):
        rng_key, params, move_type = tup_args
        birth_death = ((move_type == BIRTH) | (move_type == DEATH))
        sample_new, log_proposal_ratio = lax.cond(
            birth_death,
            (rng_key, params, move_type),
            self.__birth_death,
            (rng_key, params, move_type),
            self.__split_merge
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
    def __normal_hyper(self, tup_args):
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
    def __cond_proposal(self, rng_key, params, move_type):
        jump = ((move_type != NORMAL) & (move_type != HYPER))
        sample_new, log_proposal_ratio = lax.cond(
            jump,
            (rng_key, params, move_type),
            self.__jump_proposal,
            (rng_key, params, move_type),
            self.__normal_hyper
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
            move_type = self.__get_move_type(key1)
            sample_new, log_proposal_ratio, sample_log_joint = self.__cond_proposal(key2, head, move_type)
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

        return final_sample, final_log_joint, all_samples, all_mus, all_ns, acceptances, moves, np.exp(log_alphas)

    def sample_with_burn_in(self):
        print()
        if self.burn_in_steps//self.sample_batch_size > 0:
            self.run_sampler(burn_in=True)
        self.run_sampler(burn_in=False)
        print("Done!")

    def __collect_stats(self, acceptances, moves, alphas):
        acceptances_arr = np.array(acceptances)
        moves_arr = np.array(moves)
        alpha_arr = np.array(alphas)

        @partial(jit, static_argnums=(0, 1))
        def compile_stats(batch_size, n_moves, acceptances, moves, alphas):
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
            move_stats = all_move_stats(np.arange(n_moves), all_acceptances, all_moves, all_alphas)

            # return total proposals, acceptances, acceptances by batch, stats by move
            return all_moves.shape[0], np.sum(all_acceptances).astype(np.int32), batch_accepts, move_stats

        total_proposals, total_accepts, batch_acceptance_rates, move_stats = compile_stats(
            self.sample_batch_size, len(MOVES), moves_arr, acceptances_arr, alpha_arr
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

    def __combine_samples(self, chains, mus, ns):
        source_posterior = onp.concatenate(chains, axis=0)
        unique_mus, counts_mu = onp.unique(mus, return_counts=True)
        unique_ns, counts_n = onp.unique(ns, return_counts=True)

        self.source_posterior = onp.concatenate((self.source_posterior, source_posterior), axis=0)
        self.mu_posterior += Counter(dict(zip(unique_mus, counts_mu)))
        self.n_posterior += Counter(dict(zip(unique_ns, counts_n)))

    def run_sampler(self, burn_in=False):
        # Note: can be used to continue sampling
        start = timer()
        if burn_in:
            batches = self.burn_in_steps // self.sample_batch_size
            print(f"Burning in for {batches * self.sample_batch_size} steps:")
        else:
            # sampling
            batches = self.samples // self.sample_batch_size
            print(f"Sampling for {batches * self.sample_batch_size} steps:")

        chains, all_mus, all_ns, acceptances, all_moves, all_alphas = [], [], [], [], [], []
        t = tqdm(total=(batches * self.sample_batch_size), file=sys.stdout)
        for batch_i in range(batches):
            self.rng_key, key = random.split(self.rng_key)
            head, log_joint_head, chain, mus, ns, accepts, moves, alphas = self.sample_batch(
                key, self.head, self.log_joint_head, self.sample_batch_size
            )
            self.head, self.log_joint_head = head, log_joint_head
            if not burn_in:
                chains.append(chain)
                all_mus.append(mus)
                all_ns.append(ns)
            acceptances.append(accepts)
            all_moves.append(moves)
            all_alphas.append(alphas)
            t.update(self.sample_batch_size)
        t.close()
        end = timer()
        print("time sampling:", end - start)
        print("Recording stats from run...")
        self.__collect_stats(acceptances, all_moves, all_alphas)
        end2 = timer()
        print("time writing stats:", end2 - end)
        if not burn_in:
            print("Gathering posterior samples...")
            self.__combine_samples(chains, all_mus, all_ns)
            end3 = timer()
            print("time combining samples:", end3 - end2)

    def get_posterior_sources(self):
        return self.source_posterior

    def get_stats(self):
        return OrderedDict(
            {
                PROPOSED: self.proposals,
                ACCEPTED: self.accepted,
                BURN_IN: self.burn_in_steps,
                BATCH_SIZE: self.sample_batch_size,
                ACCEPTANCE_RATE: self.accepted / self.proposals,
                STATS_BY_MOVE: self.move_stats,
                BATCH_ACCEPTANCE_RATES: self.batch_acceptance_rates,
                N_POSTERIOR: self.n_posterior,
                MU_POSTERIOR: self.mu_posterior
            }
        )
