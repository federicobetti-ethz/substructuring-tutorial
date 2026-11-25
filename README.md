# An introduction to substructuring: Craig–Bampton and Rubin-type approaches
This repository contains the code used to generate the numerical examples shown in the substructuring tutorials for the Chair of Structural Mechanics and Monitoring at ETH Zürich.

## Repository

```text
├── docs/
│   ├── api.rst
│   ├── conf.py
│   ├── examples.rst
│   ├── index.rst
│   └── theory.rst
├── scripts/
│   ├── cantilever.py
│   └── modes.py
├── substructuring/
│   ├── __init__.py
│   ├── assembly.py
│   ├── craig_bampton.py
│   ├── fem.py
│   ├── rubin.py
│   ├── shapes.py
│   ├── static_condensation.py
│   ├── substructure.py
│   └── visualize.py
├── .gitignore
├── .pre-commit-config.yaml
├── poetry.lock
├── poetry.toml
├── pyproject.toml
└── README.md
```

## Clone the repository and install requirements

The repository requires Python 3.12 or newer.

Clone the repository with

```bash
git clone https://github.com/federicobetti-ethz/substructuring-tutorial.git
```

Install the needed packages with

```bash
cd substructuring-tutorial
poetry install --with dev --with docs
poetry run pre-commit install
```

The last command registers the commit hook. Black, Ruff, and the documentation
build then run before each commit.

## Example script

Run

```bash
poetry run python scripts/cantilever.py
```

## Build the documentation

To build the documentation, run

```bash
poetry install --with docs
poetry run sphinx-build -b html docs docs/_build/html
```

## Questions

For questions or comments, please reach out to [federico.betti@ibk.baug.ethz.ch](mailto:federico.betti@ibk.baug.ethz.ch). Otherwise, open a [GitHub issue](https://github.com/federicobetti-ethz/substructuring-tutorial/issues).

## Authors

- Federico Betti
- Charikleia Stoura
