from nustar_constants import *
from mcmc_configs import *

import jax.numpy as np
from jax import grad, jit, vmap, random, scipy
import numpy.random as nprand
from functools import partial
from collections import namedtuple


ParameterSample = namedtuple('ParameterSample', ['sources_x', 'sources_y', 'sources_b', 'mu'])


class NuSTARModel:

	def __init__(self, observed_image):
		self.observed_image = observed_image

	@staticmethod
	@jit
	def mean_emission_map(sources_x, sources_y, sources_b):
		# TODO: check jit on one source vs all

		# psf by pixel (i, j) in response to a single source
		def pixel_psf_powerlaw(i, j, source_x, source_y):
			psf_half_length = NUSTAR_IMAGE_LENGTH/2
			d = (
				((psf_half_length - i) - source_y/NUSTAR_PIXEL_SIZE)**2 +
				((j - psf_half_length) - source_x/NUSTAR_PIXEL_SIZE)**2
			)
			return 1/(1 + .1*d**2)

		# emission map psf in response to a single source
		image_psf = vmap(
			vmap(pixel_psf_powerlaw, in_axes=(None, 0, None, None)),
			in_axes=(0, None, None, None)
		)
		# image_psf = jit(image_psf)
		# emission map psf in response to a set of sources
		all_sources_psf = vmap(image_psf, in_axes=(None, None, 0, 0))

		psf_all = all_sources_psf(np.linspace(.5, 63.5, 64), np.linspace(0.5, 63.5, 64), sources_x, sources_y)
		psf_all = psf_all / np.sum(psf_all, axis=(1,2), keepdims=True)  # Normalize
		psf_all = psf_all * sources_b[:, np.newaxis, np.newaxis]  # Scale by brightness of sources
		emission_map = np.sum(psf_all, axis=0)
		return emission_map

	@staticmethod
	def sample_image(mean_emission_map):
		# NOTE: Cannot draw Poisson numbers using JAX
		return np.array(
			[[nprand.poisson(lam) for lam in row] for row in mean_emission_map]
		)

	@staticmethod
	@jit
	def log_poisson(mean, obs):
		mean = np.max([0, mean])
		return -mean + obs * np.log(mean) - scipy.special.gammaln(obs + 1)

	@staticmethod
	@jit
	def log_prior_mu(mu):
		return - np.log(mu * (np.log(N_MAX) - np.log(N_MIN)))

	@staticmethod
	@jit
	def log_prior_n(mu, n):
		return NuSTARModel.log_poisson(mu, n)

	@staticmethod
	@jit
	def log_prior_sources(sources_x, sources_y, sources_b):
		n = sources_x.shape[0]
		# Check if any sources lie outside prior support
		outside_x = np.any(np.greater(np.abs(sources_x), XY_MAX))
		outside_y = np.any(np.greater(np.abs(sources_y), XY_MAX))
		outside_b_min = np.any(np.less(sources_b, B_MIN))
		outside_b_max = np.any(np.greater(sources_b, B_MAX))
		outside_support = np.any([outside_x, outside_y, outside_b_min, outside_b_max])

		# Compute pdf
		source_inside_pdf = np.log((1.0/(XY_MAX - XY_MIN)**2))
		source_b_pdf = np.log(1.0/(B_MAX - B_MIN))
		return np.log(np.logical_not(outside_support)) + n * (source_inside_pdf + source_b_pdf)

	@partial(jit, static_argnums=(0,))
	def log_likelihood(self, sample_emission_map):
		# log poisson pdf over entire image
		map_log_poisson = vmap(vmap(self.log_poisson, in_axes=(0, 0)), in_axes=(0, 0))

		return np.sum(map_log_poisson(sample_emission_map, self.observed_image))

	@staticmethod
	def log_prior(params):
		n = params.sources_x.shape[0]
		return (
				NuSTARModel.log_prior_mu(params.mu) +
				NuSTARModel.log_prior_n(params.mu, n) +
				NuSTARModel.log_prior_sources(params.sources_x, params.sources_y, params.sources_b)
		)

	@partial(jit, static_argnums=(0,))
	def log_joint(self, params):
		emission_map = NuSTARModel.mean_emission_map(params.sources_x, params.sources_y, params.sources_b)
		return self.log_likelihood(emission_map) + NuSTARModel.log_prior(params)
