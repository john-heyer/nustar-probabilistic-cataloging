# nustar-probabilistic-cataloging
Inference of point sources observed by the NuSTAR X-Ray telescope using probabilistic cataloging.

Based on JAX, a flexible ML research library developed at google: https://github.com/google/jax

## Setup

Tested with Python 3.8 (recommended)

Install dependencies in a virtual environment: `pip install -r requirements.txt`


For GPU support, visit the <a href="https://github.com/google/jax#installation"> jax installation page</a>.

## Notes:
- `model.py` contains the code for computing the likelihood of a set of parameters
- `sampler.py` contains the fully functional sampling code, powered by `jax`. The functional constraints of `jax` lead to some interesting ways of doing things, such as indexing.
- Use file `mcmc_configs.py` to configure the sampling parameters and mock data, and `main.py` to run experiment.
- Use script `viz_results.py` to visualize the posterior and read stats.
    - example: we set the `EXPERIMENT_DIR` to `testing` in `mcmc_configs.py`
        - $ `python nucat/viz_results.py experiments/testing`

## Experiment Cookbook:
- Check configurations in `mcmc_configs.py`
- Perhaps adjust description in `main.py`
- Perhaps change `np_seed` and `jax_seed` in `main.py` for reproducibility
    - if single chain:
        - $ `python nucat/main.py`
    - if n chains:
        - $ `XLA_FLAGS="--xla_force_host_platform_device_count=n" python nucat/main.py`
        - without this will resort to `vmap`, which is preferred on machines without many cores
- Come back later!
