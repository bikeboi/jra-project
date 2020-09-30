# Mushroom Body Experiments

> **NOTE**: The organizational structure for this project is being revamped. Experiment workflow and details
>pertaining to such are likely to change in the near future, in an effort to simplify things.
> 
>-- Michael

## Setup
### Environment
The entire project lives inside an environment with all required dependencies installed
Create, activate, and navigate to a new python [virtual environment](https://docs.python.org/3/tutorial/venv.html):
```sh
python -m venv [dirname]
cd [dirname]
source bin/activate # Activate the venv
```

Install the required dependencies:
```sh
pip install -r requirements.text
```

### PyNN GeNN
Clone this repository, [GeNN](https://github.com/genn-team/genn), [pyNN_genn](https://github.com/genn-team/pynn_genn), and [omniglot](https://github.com/brendenlake/omniglot):
```sh
git clone git@github.com:bikeboi/jra-project.git
git clone git@github.com:genn-team/genn.git
git clone git@github.com:genn-team/pynn_genn.git
git clone git@github.com:brendenlake/omniglot.git
```

Navigate into the `genn` folder and build GeNN and pyGeNN with the following commands:
```sh
# Build GeNN for pyGeNN
make DYNAMIC=True LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper 

# Build pyGeNN
python setup.py develop
```

Finally, navigate into the `pynn_genn` folder and build:
```sh
python setup.py develop
```

If `develop` doesn't work, try `install`.

## Running Experiments
Experiments are run using the [Sacred](https://sacred.readthedocs.io/en/stable/experiment.html)
library.

The main experiment file (default configuration) is [experiment.py](./experiment.py). Running
```
$ python3 experiment.py
```
will execute the experiment with the default parameters specified within
the `base_config` function of experiment.py. Testing out variants of the experiment
is as easy as reassigning parameters within this function.

### Results
By default, all results are stored under: `results/{tag}_{model_type}_{n_class}`,
where `tag` is the experiment identifier (defaulting to `muted`), `model_type` should
be one of `supervised` or `unsupervised` (defaulting to the former), and `n_class` is
self-explanatory. 

Within this directory (henceforth referred to as `root`), is stored
`root/data_{pop}_{run_id}.pickle` files of spiking activity, where `pop` is the population
name, and `run_id` is the number indexing the run data was recorded from. Spike data
is storde in NeoIO format. See the [pyNN docs](http://neuralensemble.org/docs/PyNN/data_handling.html)
for more information.

Additionally, logs of output population weights are stored under `root/weight_log.npy` as
numpy arrays.
