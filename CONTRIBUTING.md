# Contributing guidelines

This page provides an overview on contributing to PROTEUS itself and the ecosystem.
Anyone who makes a Pull Request to the `main` branch should read this document *fully* first.

## Licensing and ownership

PROTEUS and its submodules are *free* software and also *open source* software. Roughly, this means that the users have the freedom to run, copy, distribute, study, change and improve the software. The term "free software" is a matter of liberty, not price or monetary value [[ref]](https://www.gnu.org/philosophy/free-sw.html).

Specifically, PROTEUS is made available under the *Apache 2.0 License* which is provided in the file `LICENSE.txt`, although some submodules are available under other (similar) licenses.

The Apache license permits [[ref]](https://choosealicense.com/licenses/apache-2.0/) commercial use, unlimited distribution, modification of the code elsewhere, and private use. However, there are caveats to these terms: contributors have no liability, the code has no warranty, and a trademark may not be applied [[ref]](https://opensource.com/article/18/2/apache-2-patent-license).

This does not specify who *owns* the material (i.e. PROTEUS and its submodules). It only specifies how the material may be used by both developers and non-developers alike.  By default, as a work is created its copyright belongs to the person who created it [[ref]](https://assets.publishing.service.gov.uk/media/5a7eaf0ae5274a2e87db13f3/c-notice-201402.pdf). Although in some cases your employer/university may be the copyright holder of work you create [[ref]](https://www.fsf.org/licensing/contributor-faq). PROTEUS is not 'owned' by a single entity; the individual parts of the framework are owned by the people who made them, and licensed for use and modification by others. More information can be found [here](https://opensource.guide) and [here](https://fossa.com/learn/open-source-licenses/).

The principle purpose of PROTEUS (and its submodules) is to generate data which are then used to make scientific conclusions and write papers. It is generally expected that the primary author of a paper is the person who contributed the most work to that specific project. We ask that:

1. authorship is offered to the Contributors based on the relevance of their work in making the paper,
2. appropriate credit is provided in the Acknowledgements section of the paper,
3. the Maintainers are made aware of when PROTEUS results are used in a scientific paper.

A suggested acknowledgement is:
> We thank the people who have contributed to PROTEUS and its broader ecosystem for their support and enabling the scientific outputs of this paper.

<b>
In summary:

* you generally own all of the code you write and material you create,
* you give irrevocable permission for the code to be used under the license when distributed,
* you are requested to give offer authorship and give credit in papers as appropriate.
</b>

PROTEUS would not be exist without the efforts of the wider community. Contributions from research scientists, software developers, students, and many others have made development of the current framework possible. Thank you for your interest in contributing to PROTEUS, and the immense task of simulating the lifetimes of entire planets and stars.

"Alone we can do do so little; together we can do so much." - Helen Keller

## How do I contribute?

Contributing to PROTEUS is relatively straightforward. We use Git to manage the source code, and GitHub to host it online. Here is a simple workflow:

1. Download the code and make sure that it runs on your machine
2. Create a new 'branch' called `MY_BRANCH` using Git: `git checkout -b MY_BRANCH`
3. Make changes to the code as you so desire
4. Add these changes to the repository: `git add .`
5. Commit these changes with a message: `git commit -m MESSAGE`
4. Push these changes to GitHub: `git push -u origin MY_BRANCH`
5. When you've got neat set of changes, make a 'pull request' on GitHub [here](https://github.com/FormingWorlds/PROTEUS/pulls). This makes you a Contributor to the project.
6. One of the Maintainers of the project will review the request.
7. When ready, the changes will be merged into the `main` branch and are made live!

A series of 'hooks' will check the syntax and validity of your code when committing. With a significant number of people contributing to the codebase, automatic checks are important for preventing programming errors, bugs, stylistic problems, and large files from being committed to the repositories [[ref]](https://en.wikipedia.org/wiki/Lint_(software)).

Currently, the Maintainers of the code are: Harrison Nicholls and Tim Lichtenberg.

## Development rules

### Versioning

PROTEUS targets Python version 3.12 and is not intended to work on earlier versions of Python.
We also target Linux and MacOS as the *only* supported operating systems. Windows and BSD are not supported.

The version of PROTEUS itself is defined using calendar versioning in the format `YY.MM.DD`. This is defined in the code and in the `pyproject.toml` metadata file.

### Code style

1. Variables should be written in lowercase with underscores: `example_variable`.
2. Functions should be written in lowercase with underscores: `example_function()`.
3. Constants should be written in block capitals: `CONSTANT_VALUE`.
4. Lines of code should avoid being longer than 92 characters where possible
5. Functions should include a docstring where possible, describing the function's purpose and parameters.
6. Indentation deeper than 3 levels should be avoided.

### Large files, outputs, and lookup data

Large files should **not** be committed to the repository. This means model results, plots, and files you create during analysis. Including these will make Git operations sluggish and make version control tricky, as Git is only meant for managing text (e.g. code) files.

You can make files/folders invisible to Git by prepending `nogit_` to their names. For example, anything in a folder called `nogit_analysis/` will be ignored by Git. Large files could then be safely placed in this folder. Model outputs are generated in the `output/` folder, which is also ignored by Git.

Large files can be useful in some cases, such as in the case of lookup-data tables used in the simulations. These files should be kept inside your FWL data folder defined by the `FWL_DATA` environment variable in your shell. Placing large files in this folder allows them to be kept on 'storage' file systems on clusters to avoid reaching your allocation limits.

PROTEUS will download large files from Zenodo when required. These are then kept in the FWL data folder. You can find our Zenodo community [here](https://zenodo.org/communities/proteus_framework/).

### Linting

Linting is a term for static code analysis to flag programming errors,
bugs, stylistic errors and suspicious constructs [[ref]](https://en.wikipedia.org/wiki/Lint_(software)).

PROTEUS uses [`ruff`](https://astral.sh/ruff) for linting. The linting [rules](https://docs.astral.sh/ruff/rules/) are defined in [`pyproject.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/pyproject.toml). This check are run automatically via a Github Action: [codestyle](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/codestyle.yaml).

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

PROTEUS uses [pytest](https://docs.pytest.org/en/latest/) to run the tests on the code. Tests are important for ensuring that the code behaves as expected, and for finding bugs/errors as soon as they arise. You can read more about software testing in general [here](https://www.geeksforgeeks.org/software-testing/software-testing-basics/).

Our tests are run automatically via a Github Action: [tests](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/tests.yaml). You can also run the tests for yourself using the command:

```console
pytest
```

To check the 'coverage' of the tests:

```console
coverage run -m pytest
coverage report  # to output to terminal
coverage html    # to generate html report
```

The 'coverage' of the tests describes the fraction of the code which is executed while tests are being run. However, care should also be taken to ensure that the *output* of the tests meets expectations.

### Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/), and uses [mkdocs](https://www.mkdocs.org/) to generate the pages.

To build the documentation for yourself:

```console
pip install -e '.[docs]'
mkdocs serve
```

This will generate the markdown files and serve them on a local server. You can view documentation while you edit by copy-pasting the displayed URL into your browser (e.g., `http://127.0.0.1:8000`).

You can find the documentation source in the [docs](https://github.com/FormingWorlds/PROTEUS/tree/main/docs) directory.
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/FormingWorlds/PROTEUS/blob/main/mkdocs.yml) under the `nav` entry.

The documentation is hosted on [readthedocs](https://readthedocs.io/projects/fwl-proteus).

### Making a release

The versioning scheme we use is [CalVer](https://calver.org/).

0. Update requirements files:

```console
python tools/generate_requirements_txt.py
pip-compile -o requirements_full.txt pyproject.toml
```

1. Bump the version (`release`/`patch`) as needed

```console
bump-my-version bump release
# 24.08.12
```

2. Commit and push your changes.

3. Make a new [release](https://github.com/FormingWorlds/PROTEUS/releases). Make sure to set the tag to the specified version, e.g. `24.08.12`.

4. The [upload to pypi](https://pypi.org/project/fwl-proteus) is triggered when a release is published and handled by [this workflow](https://github.com/FormingWorlds/PROTEUS/actions/workflows/publish.yaml).
