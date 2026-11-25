# An introduction to substructuring: Craig–Bampton and Rubin-type approaches
This repository contains the code used to generate the numerical examples shown in the substructuring tutorials for the Chair of Structural Mechanics and Monitoring at ETH Zürich.

## Run the cantilever example

The example needs Python 3.12 or newer.

Clone the repository

```bash
git clone https://github.com/federicobetti-ethz/substructuring-tutorial.git
```

Install the needed packages

```bash
cd substructuring-tutorial
poetry install
```

Run the main script

```bash
poetry run python scripts/cantilever.py
```

## Build the documentation

The documentation is a Sphinx site with a theory page, the examples, and the
reference of the public functions.

```bash
poetry install --with docs
poetry run sphinx-build -b html docs docs/_build/html
```

## Authors

- Federico Betti
