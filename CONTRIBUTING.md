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

### Running tests

PROTEUS uses [pytest](https://docs.pytest.org/en/latest/) to run the tests.
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

### Making a release

The versioning scheme we use is [CalVer](https://calver.org/).

0. Update requirements files:

```console
python tools/requirements_txt.py
pip-compile -o requirements_full.txt pyproject.toml
```

1. Bump the version (`release`/`patch`) as needed

```console
bump-my-version release
# 24.08.12
```

2. Commit and push your changes.

3. Make a new [release](https://github.com/FormingWorlds/PROTEUS/releases). Make sure to set the tag to the specified version, e.g. `24.08.12`.

4. The [upload to pypi](https://pypi.org/project/fwl-proteus) is triggered when a release is published and handled by [this workflow](https://github.com/FormingWorlds/PROTEUS/actions/workflows/publish.yaml).
