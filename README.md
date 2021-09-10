# Gaussian Process Dynamic Mode Decomposition
This repository is the official implementation of 'Gaussian Process Dynamic Mode Decomposition' an extension of the 
Master Thesis "Probabilistic Dynamic Mode Primitives" under the supervision of Hany Abdulsamad and Svenja Stark. 

## Getting Started
### Prerequisites
This project is compatible with Python 3.8 and
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/ "Setting up Anaconda"). Further [JAX](https://jax.readthedocs.io/en/latest/developer.html "Setting up JAX")
is required.

### Install
##### Install virtual environment:
You can also change the environment name if necessary.
```bash
$ conda env create -f base_env.yml
```
There can be some issues with installing jax. 

## Developers
- Kay Hansel

## Open Issues 
- All old DMD related methods are not tested yet if they still work. 
- At the moment, only ARD-Kernels are supported. 
- Implement ProMPs and other methods to get good benchmarks.
- The current yml file is not up to date. It contains also packages which are not used anymore.
