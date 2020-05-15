# nustar-probabilistic-cataloging
Inference of point sources observed by the NuSTAR X-Ray telescope using probabilistic cataloging.

NOTES:
- `model.py` contains the code for computing the likelihood of a set of parameters
- `sampler.py` contains the fully functional sampling code, powered by `jax`. The constraints of `jax` lead to some interesting ways of doing things such as indexing.
- Use file `mcmc_configs.py` to configure the sampling parameters and mock data, and `main.py` to run experiment.
- Use script `viz_results.py` to visualize the posterior and read stats.
    - example: we set the `EXPERIMENT_DIR` to `experiments/testing` in `mcmc_configs.py`
        - $ `python pystar/viz_results.py experiments/testing`

To run experiment:
- Check configurations in `mcmc_configs.py`
- Perhaps adjust description in `main.py`
    - if single chain:
        - $ `python pystar/main.py`
    - if n chains:
        - $ `XLA_FLAGS="--xla_force_host_platform_device_count=n" python pystar/main.py`
        - without this will resort to `vmap`, which is preferred on machines without many cores
- Come back later!
