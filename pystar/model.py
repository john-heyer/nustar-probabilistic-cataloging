from collections import namedtuple
from functools import partial

import jax.numpy as np
import numpy as onp
from jax import jit, vmap, scipy, lax, ops, random

from mcmc_configs import *
from nustar_constants import *
from utils import random_sources

onp.random.seed(1)  # for drawing poisson numbers

import matplotlib.pyplot as plt
from timeit import default_timer as timer

ParameterSample = namedtuple('ParameterSample', ['sources_x', 'sources_y', 'sources_b', 'mask', 'mu', 'n'])

# TODO:
# [] check psfs zero edge
# [] parallelize
# [] inspect zeros and infs
# [] remove mask from parameter sample and construct based on n
# [] make non-normal moves likelihoods constant time
# [] update psrf
# [] use logging

PSF_BY_ARC_MINUTE = np.array(onp.load("psf_by_arcminute.npy"))
print(PSF_BY_ARC_MINUTE.shape)

class NuSTARModel:

    def __init__(self, observed_image):
        self.observed_image = observed_image

    @staticmethod
    def rebin(arr, shape):
        sh = shape[0], arr.shape[0] // shape[0], shape[1], arr.shape[1] // shape[1]
        return arr.reshape(sh).sum(-1).sum(1)

    @staticmethod
    def bilinear_interpolation(x, y, x_1, x_2, y_1, y_2, psf):
        f_11 = psf[y_1, x_1]
        f_12 = psf[y_2, x_1]
        f_21 = psf[y_1, x_2]
        f_22 = psf[y_2, x_2]
        # formula: https://en.wikipedia.org/wiki/Bilinear_interpolation
        interpolant = (
                1 / ((x_2 - x_1) * (y_2 - y_1)) *
                np.array([x_2 - x, x - x_1]) @ np.array([[f_11, f_12], [f_21, f_22]]) @ np.array([y_2 - y, y - y_1])
        )
        # nan occurs from divide by zero on edges, where x_1 == x_2 (or y_1 == y_2) due to index clipping
        return np.nan_to_num(interpolant)

    @staticmethod
    @partial(jit, static_argnums=(3,))
    def mean_emission_map_new(sources_x, sources_y, sources_b, up_sample_factor):

        def pixel_psf(x_im, y_im, source_x, source_y, theta, psf):
            # translated
            x_t = x_im - source_x
            y_t = y_im - source_y
            R = np.array(
                [[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]]
            )
            # scaled
            s = PSF_PIXEL_SIZE
            # rotated
            x_psf, y_psf = (1 / s * R @ np.array([x_t, y_t]))
            # center of image is zero
            x_psf_idx = PSF_IMAGE_LENGTH / 2 + x_psf - .5
            y_psf_idx = PSF_IMAGE_LENGTH / 2 - y_psf - .5
            # prevent wrap indexing from negative values
            x_psf_idx = np.max([x_psf_idx, 0])
            y_psf_idx = np.max([y_psf_idx, 0])
            # 4 nearest pixel indices
            x_1, x_2 = np.floor(x_psf_idx).astype(np.int32), np.ceil(x_psf_idx).astype(np.int32)
            y_1, y_2 = np.floor(y_psf_idx).astype(np.int32), np.ceil(y_psf_idx).astype(np.int32)
            return NuSTARModel.bilinear_interpolation(x_psf_idx, y_psf_idx, x_1, x_2, y_1, y_2, psf)

        # emission map in response to a single source
        def source_map(x_coordinates, y_coordinates, source_x, source_y):
            theta = (np.arctan2(source_y, source_x).squeeze())
            r = np.sqrt(source_x**2 + source_y**2)
            arc_minutes = r / RADIANS_PER_ARC_MINUTE
            arc_minutes = np.min([arc_minutes, 8])  # only 9 entries
            low, high = np.floor(arc_minutes).astype(np.int32), np.ceil(arc_minutes).astype(np.int32)
            psf_low, psf_high = PSF_BY_ARC_MINUTE[low], PSF_BY_ARC_MINUTE[high]
            psf_interpolated = (1 - (arc_minutes - low)) * psf_low + (arc_minutes - low) * psf_high
            map_source = vmap(
                vmap(pixel_psf, in_axes=(0, None, None, None, None, None)),
                in_axes=(None, 0, None, None, None, None)
            )
            return map_source(x_coordinates, y_coordinates, source_x, source_y, theta, psf_interpolated)

        # emission map in response to a set of sources
        map_all_sources = vmap(source_map, in_axes=(None, None, 0, 0))

        up_sampled_pixel_size = NUSTAR_PIXEL_SIZE / up_sample_factor
        n_pixels = NUSTAR_IMAGE_LENGTH * up_sample_factor

        x_coords = np.linspace(-n_pixels / 2 + .5, n_pixels / 2 - .5, n_pixels) * up_sampled_pixel_size
        y_coords = np.linspace(n_pixels / 2 - .5, -n_pixels / 2 + .5, n_pixels) * up_sampled_pixel_size
        all_sources_map = map_all_sources(x_coords, y_coords, sources_x, sources_y)
        rebin_all = vmap(NuSTARModel.rebin, in_axes=(0, None))

        pixel_area_ratio = up_sampled_pixel_size**2 / PSF_PIXEL_SIZE**2
        psf_all = rebin_all(all_sources_map, (NUSTAR_IMAGE_LENGTH, NUSTAR_IMAGE_LENGTH)) * pixel_area_ratio
        psf_all = psf_all * sources_b[:, np.newaxis, np.newaxis]  # scale by brightness of sources
        emission_map = np.sum(psf_all, axis=0)
        return emission_map

    @staticmethod
    @jit
    def mean_emission_map(sources_x, sources_y, sources_b):
        # psf by pixel (i, j) in response to a single source
        def pixel_psf_powerlaw(i, j, source_x, source_y):
            psf_half_length = NUSTAR_IMAGE_LENGTH / 2 / 2
            d = (
                    ((psf_half_length - i) - source_y / NUSTAR_PIXEL_SIZE / 2) ** 2 +
                    ((j - psf_half_length) - source_x / NUSTAR_PIXEL_SIZE / 2) ** 2
            )
            return 1 / (1 + .5 * d ** 2)

        # emission map psf in response to a single source
        image_psf = vmap(
            vmap(pixel_psf_powerlaw, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None)
        )
        # emission map psf in response to a set of sources
        all_sources_psf = vmap(image_psf, in_axes=(None, None, 0, 0))

        psf_all = all_sources_psf(np.linspace(.5, 31.5, 32), np.linspace(0.5, 31.5, 32), sources_x, sources_y)
        psf_all = psf_all / np.sum(psf_all, axis=(1, 2), keepdims=True)  # Normalize
        psf_all = psf_all * sources_b[:, np.newaxis, np.newaxis]  # Scale by brightness of sources and mask
        emission_map = np.sum(psf_all, axis=0)
        return emission_map

    @staticmethod
    def sample_image(mean_emission_map):
        # NOTE: Cannot draw Poisson numbers using JAX
        return np.array(
            [[onp.random.poisson(lam) for lam in row] for row in mean_emission_map]
        )

    @staticmethod
    @jit
    def log_poisson(mean, obs):
        mean = np.max([0, mean])
        return -mean + obs * np.log(mean) - scipy.special.gammaln(obs + 1)

    @staticmethod
    @jit
    def log_prior_mu(mu):
        zero = (mu > N_MAX) | (mu < N_MIN)
        log_p_mu = - np.log(mu * (np.log(N_MAX) - np.log(N_MIN)))
        return np.where(zero, -np.inf, log_p_mu)

    @staticmethod
    @jit
    def log_prior_n(mu, n):
        return NuSTARModel.log_poisson(mu, n)

    @staticmethod
    @jit
    def log_prior_sources(sources_x, sources_y, sources_b, mask, n):
        # Check if any sources lie outside prior support
        outside_x = np.any(np.logical_and(mask, np.greater(np.abs(sources_x), XY_MAX)))
        outside_y = np.any(np.logical_and(mask, np.greater(np.abs(sources_y), XY_MAX)))
        outside_b_min = np.any(np.logical_and(mask, np.less(sources_b, B_MIN)))
        outside_b_max = np.any(np.logical_and(mask, np.greater(sources_b, B_MAX)))
        outside_support = np.any([outside_x, outside_y, outside_b_min, outside_b_max])

        # Compute pdf
        source_inside_pdf = np.log((1.0 / (XY_MAX - XY_MIN) ** 2))
        source_b_pdf = np.log(1.0 / (B_MAX - B_MIN))
        return np.log(np.logical_not(outside_support)) + n * (source_inside_pdf + source_b_pdf)

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, sample_emission_map):
        # log poisson pdf over entire image
        map_log_poisson = vmap(vmap(self.log_poisson, in_axes=(0, 0)), in_axes=(0, 0))

        return np.sum(map_log_poisson(sample_emission_map, self.observed_image))

    @staticmethod
    @jit
    def log_prior(params):
        return (
                NuSTARModel.log_prior_mu(params.mu) +
                NuSTARModel.log_prior_n(params.mu, params.n) +
                NuSTARModel.log_prior_sources(
                    params.sources_x, params.sources_y, params.sources_b, params.mask, params.n
                )
        )

    @partial(jit, static_argnums=(0,))
    def log_joint(self, params):
        emission_map = NuSTARModel.mean_emission_map(params.sources_x, params.sources_y, params.sources_b)
        return self.log_likelihood(emission_map) + NuSTARModel.log_prior(params)


if __name__ == "__main__":
    sources_x = np.array([-5 * NUSTAR_PIXEL_SIZE*2])
    sources_y = np.array([5 * NUSTAR_PIXEL_SIZE*2])
    sources_b = np.array([1])

    map = NuSTARModel.mean_emission_map(sources_x, sources_y, sources_b)
    map = onp.array(map)
    plt.matshow(map)
    plt.show()
1