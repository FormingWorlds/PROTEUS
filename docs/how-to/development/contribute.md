# Contributing Code to PROTEUS

## Overview

Contributing to PROTEUS is straightforward. We use Git for version control and GitHub to host the code. This guide walks you through the contribution workflow.

## Basic Workflow

1. **Download the code** and ensure it runs on your machine
2. **Create a branch**: `git checkout -b MY_BRANCH`
3. **Make changes** to the code as desired
4. **Stage changes**: `git add .`
5. **Commit changes**: `git commit -m "MESSAGE"`
6. **Push to GitHub**: `git push -u origin MY_BRANCH`
7. **Open a Pull Request** on [GitHub](https://github.com/FormingWorlds/PROTEUS/pulls)
8. **Wait for review** from project maintainers
9. **Merge to main** when approved

## Pre-commit Hooks

Git hooks automatically validate your code when committing:

- Syntax checking
- Code validity verification
- Style verification
- Large file detection

These automatic checks prevent bugs and maintain code quality across the project.

## Development Requirements

### Python Version

PROTEUS targets **Python 3.12** exclusively. Earlier versions are not supported.

### Supported Platforms

- Linux
- macOS

Windows and BSD are not supported.

### Versioning

PROTEUS uses calendar versioning in the format `YY.MM.DD` (e.g., `24.08.12`), defined in the code and `pyproject.toml`.

## Code Style Guidelines

Follow these conventions for all code contributions:

**Variables and functions:**

- Use lowercase with underscores: `example_variable`, `example_function()`

**Constants:**

- Use block capitals: `CONSTANT_VALUE`

**Line length:**

- Keep lines under 92 characters when possible

**Docstrings:**

- Include docstrings for all functions describing purpose and parameters

**Indentation:**

- Avoid indentation deeper than 3 levels

## Code Quality Checks

### Linting with ruff

PROTEUS uses [`ruff`](https://astral.sh/ruff) for static code analysis. Rules are defined in `pyproject.toml`.

Check a single file:

```console
ruff check start_proteus.py
```

Check a directory:

```console
ruff check src/proteus
```

Check everything:

```console
ruff check .
```

Auto-fix some issues:

```console
ruff check . --fix
```

### Using pre-commit

Automatically run linting on every commit:

```console
pre-commit install
```

## Running Tests

PROTEUS uses [pytest](https://docs.pytest.org/en/latest/) for testing. Tests ensure code behaves correctly and catch bugs early.

Run all tests:

```console
pytest
```

Run tests matching a keyword:

```console
pytest -k keyword
```

Check test coverage:

```console
coverage run -m pytest
coverage report      # terminal output
coverage html        # generate HTML report
```

Tests are located in the `tests/` folder. Test files are automatically discovered by pytest.

## Data Management

### Large Files

**Do not commit large files** to the repository. These include model results, plots, and analysis outputs. Git is designed for text/code only.

Make files/folders invisible to Git by prepending `nogit_` to their names:

```text
nogit_analysis/    # contents ignored by Git
```

Model outputs are generated in `output/`, which is also ignored.

### Input Data

PROTEUS downloads large input data from [Zenodo](https://zenodo.org/communities/proteus_framework/) (with [OSF](https://osf.io/8dumn/) as backup). Data is stored locally in your `FWL_DATA` folder.

**Adding new data:**

1. **Updating existing record**: Add files to Zenodo in consistent format (e.g., adding new stellar spectra)
2. **Creating new record**: Create a new Zenodo record following the naming scheme `spectral_files/model_name/number_of_bands`
3. **Update download function**: Modify `src/proteus/utils/data.py` with the Zenodo record number
4. **Also upload to OSF**: Keep backups synchronized

After uploading, you may need to delete your local data cache to force a re-download.

## Building Documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/) and built with [mkdocs](https://www.mkdocs.org/).

Install documentation tools:

```console
pip install -e '.[docs]'
```

Start the development server:

```console
mkdocs serve
```

View the documentation at `http://127.0.0.1:8000`. Changes reload automatically.

### Updating Navigation

When adding new documentation pages, update [`mkdocs.yml`](https://github.com/FormingWorlds/PROTEUS/blob/main/mkdocs.yml) under the `nav` entry.

### Common Issue: Auto-reload Not Working

If the development server doesn't auto-reload when you save files, it's likely an incompatibility with the `click` library.

Check your `click` version:

```console
pip freeze | grep click
```

If you see version 8.1.4 or higher, downgrade:

```console
pip install "click<8.1"
```

Then restart `mkdocs serve`.

## Making a Release

PROTEUS uses [CalVer](https://calver.org/) versioning in format `YY.MM.DD` (e.g., `24.08.12`).

### Release Steps

1. Update requirements:

   ```console
   python tools/generate_requirements_txt.py
   pip-compile -o requirements_full.txt pyproject.toml
   ```

2. Bump the version:

   ```console
   bump-my-version bump release
   ```

3. Commit and push changes:

   ```console
   git commit -am "Bump version to X.X.X"
   git push origin main
   ```

4. Create a [GitHub Release](https://github.com/FormingWorlds/PROTEUS/releases):
   - Set the tag to the version (e.g., `24.08.12`)
   - Include release notes

5. PyPI upload is **automatic**: The [publish workflow](https://github.com/FormingWorlds/PROTEUS/actions/workflows/publish.yaml) triggers on release publication.

## Questions?

Current maintainers: Harrison Nicholls and Tim Lichtenberg

For more information on open-source contributions, see [opensource.guide](https://opensource.guide) and [fossa.com](https://fossa.com/learn/open-source-licenses/).
