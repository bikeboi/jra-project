# Mushroom Body Experiments

## Setup
### Environment
Create, activate, and navigate to a new [virtual environment](https://docs.python.org/3/tutorial/venv.html):
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