# nustar-probabilistic-cataloging
Inference of point sources observed by the NuSTAR X-Ray telescope using probabilistic cataloging.

NOTES:
- `model.py` contains the code for computing the likelihood of a set of parameters
- `sampler.py` contains the fully functional sampling code, powered by `jax`. The constraints of `jax` lead to some interesting ways of doing things such as indexing.
- Use file `mcmc_configs.py` to configure the sampling parameters and mock data, and `main.py` to run experiment.
- Use script `viz_results.py` to visualize the posterior.
- `psrf.py` computes and displays the $\hat{R}$ psrf statistic over all pixels of the 64x64 emission map
