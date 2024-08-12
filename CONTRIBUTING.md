# Contributing guidelines

## Development

*PROTEUS* targets Python 3.10 or newer.

Clone the repository into the `proteus` directory:

```console
git clone https://github.com/FormingWorlds/PROTEUS proteus
```

Install using `virtualenv`:

```console
cd proteus
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[develop]
```

Alternatively, install using Conda:

```console
cd proteus
conda create -n proteus python=3.10
conda activate proteus
pip install -e .[develop]
```

### Linting

Linting is a term for static code analysis to flag programming errors,
bugs, stylistic errors and [suspicious constructs](https://en.wikipedia.org/wiki/Lint_(software)).
PROTEUS uses [`ruff`](https://astral.sh/ruff) for linting.
The linting [rules](https://docs.astral.sh/ruff/rules/) are defined in [`pyproject.toml`](https://github.com/FormingWorlds/PROTEUS/blob/master/pyproject.toml).

This check are run automatically via a Github Action: [codestyle](https://github.com/FormingWorlds/PROTEUS/blob/master/.github/workflows/codestyle.yaml).

You can `ruff` on locally using one of these commands:

```console
ruff check start_proteus.py  # single file
ruff check src/proteus       # directory
ruff check .                 # everything
```

If you prepend `--fix`, it can also fix some issues for you:

```console
ruff check . --fix
```

You can also use [pre-commit](https://pre-commit.com/#usage) to automatically run `ruff` on every commit, e.g.:

```console
pre-commit install
```

### Running tests

PROTEUS uses [pytest](https://docs.pytest.org/en/latest/) to run the tests.

The tests are run automatically via a Github Action: [tests](https://github.com/FormingWorlds/PROTEUS/blob/master/.github/workflows/tests.yaml).

You can run the tests for yourself using:

```console
pytest
```

To check coverage:

```console
coverage run -m pytest
coverage report  # to output to terminal
coverage html    # to generate html report
```

### Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/), and uses [mkdocs](https://www.mkdocs.org/) to generate the pages.

To build the documentation for yourself:

```console
pip install -e .[docs]
mkdocs serve
```

You can find the documentation source in the [docs](https://github.com/FormingWorlds/PROTEUS/tree/master/docs) directory.
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/FormingWorlds/PROTEUS/blob/master/mkdocs.yml) under the `nav` entry.

The documentation is hosted on [readthedocs](https://readthedocs.io/projects/fwl-proteus).
