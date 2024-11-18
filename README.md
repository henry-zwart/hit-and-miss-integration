# Assignment 1 for Stochastic Simulation

## Quickstart - Reproducing Results

Ensure that you have a working version of `make` installed:

```zsh
# Should print 'GNU Make X.XX, etc.'
make --version
```

The [recommended approach](#with-uv-recommended) to running the experiments requires an installation of the [uv](https://github.com/astral-sh/uv) Python package manager.

If this is not installed, follow the [alternative instructions](#without-uv) for running experiments.

### With uv (recommended)

Ensure [uv](https://github.com/astral-sh/uv) is installed locally.

uv handles package and Python versioning, ensuring a consistent experiment environment across machines and operating systems.

To reproduce the results execute the following steps:

```zsh
cd path/to/repository

# If you have limited computational resources
make

# If you have at least ~32GB memory and several CPU cores
make -j 4
```

After this completes, the experimental results (metadata + figures) are located in `./results`, and intermediary data in `./data`.

### Without uv

Note that only the [uv approach to running experiments](#with-uv-recommended) has been thoroughly tested, and is therefore recommended over this method. 

The experiments rely on Python3.12, and the packages outlined in [requirements.txt](requirements.txt).

> **Note:** If your Python3.12 executable is called something other than `python3.12`, replace this accordingly in the following steps.

To reproduce the results, execute the following steps:

```zsh
cd path/to/repository

# == Create a virtual environment
python3.12 -m venv env

# == Activate the virtual environment
source env/bin/activate # On macOS or Linux
.\env\Scripts\activate # On windows

# == Install package and dependencies
pip install .

# == Run the experiments
# === If you have limited computational resources
ENTRYPOINT=python make

# === If you have at least ~32GB memory and several CPU cores
ENTRYPOINT=python make -j 4
```

After this completes, the experimental results (metadata + figures) are located in `./results`, and intermediary data in `./data`.


## Development - Getting started
See the [assignment specification](assignment_spec.pdf).

This repository uses _uv_ to manage Python and its dependencies, and _pre-commit_ to run
automatic code linting & formatting.

1. Install [uv](https://github.com/astral-sh/uv)

2. Navigate to this project directory

3. Install pre-commit:

```zsh
# We can use uv to install pre-commit!
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Check that pre-commit installed alright (should say 3.8.0 or similar)
pre-commit --version

# After installing pre-commit, you'll need to initialise it.
# This installs all pre-commit hooks, the scripts which run before a commit.
pre-commit install

# It's a good idea to run pre-commit now on all files.
pre-commit run --all-files
```

4. Run code:

```zsh
uv run <PATH-TO-PYTHON-FILE>
```

## Adding new packages with uv

It's simple to add a Python package to the project with uv using `uv add`.
For instance, to install numpy and scipy, and add them to the dependency graph, run:

```zsh
uv add numpy scipy
```

To remove a package dependency, e.g., scipy, run:

```zsh
uv remove scipy
```

## Running JupyterLab with uv

uv can also be used to run Jupyter, the same way as we run Python scripts:

```zsh
# If Jupyter Lab isn't already installed, add it as a dependency
uv add jupyterlab

# Start a JupyterLab server
uv run jupyter lab
```


## Pre-commit hooks

I've included a couple of pre-commit hooks 
(see [.pre-commit-config.yaml](.pre-commit-config.yaml)) which will be executed every 
time we commit code to git. Both pre-commit hooks come from the 
[ruff](https://github.com/astral-sh/ruff) Python linter:
- `ruff`: lints Python code to ensure it adheres to the PEP8 standards. Includes a bunch of nice things like automatic sorting of imports by name and type.
- `ruff format`: formats Python code in a consistent manner, including removing excess whitespace, swapping tabs for spaces, and a _tonne_ of other things.

These should help to keep code tidy and consistent, with no extra effort on our part. 
Both of the hooks should run automagically if you've followed the setup instructions for
[installing pre-commit with uv](#development---getting-started).
