# Tutorial on substructuring (SMM, ETH Zürich)
This repository contains the code used to generate the numerical examples shown in the substructuring tutorials for the Chair of Structural Mechanics and Monitoring at ETH Zürich.

## Run the cantilever example

The example needs Python 3.12 or newer and [Poetry](https://python-poetry.org/docs/#installation). 

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
