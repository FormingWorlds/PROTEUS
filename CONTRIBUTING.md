# Contributing guidelines

This page provides an overview on contributing to PROTEUS itself and the ecosystem.
Anyone who makes a Pull Request to the `main` branch should read this document *fully* first.

## Licensing and ownership

PROTEUS and its submodules are *free* software and also *open source* software. Roughly, this means that the users have the freedom to run, copy, distribute, study, change and improve the software. The term "free software" is a matter of liberty, not price or monetary value [[ref]](https://www.gnu.org/philosophy/free-sw.html).

Specifically, PROTEUS is made available under the *Apache 2.0 License* which is provided in the file `LICENSE.txt`, although some submodules are available under other (similar) licenses.

The Apache license permits [[ref]](https://choosealicense.com/licenses/apache-2.0/):

* Commercial use
* Unlimited distribution (e.g. online)
* Modification of the code elsewhere
* Patent use - meaning that no one is owed royalties [[ref]](https://opensource.com/article/18/2/apache-2-patent-license).
* Private use - the codes/tools may be used in private

However, there are caveats to these terms, including:

* That contributors have no liability and that the code has no warranty.
* A trademark may not be applied

This does not specify who *owns* the material (i.e. PROTEUS and its submodules). It only specifies how the material may be used by both developers and non-developers alike. The Apache License is based on the concept of Copyright.

Computer code is covered by copyright, much like books or movies or photographs. By default, the author of a  work is legally taken to be the first owner of its copyright; as a work is created, its copyright belongs to the person who created it [[ref]](https://assets.publishing.service.gov.uk/media/5a7eaf0ae5274a2e87db13f3/c-notice-201402.pdf). This means that, with a few exceptions or unless authorised, only a copyright holder is legally allowed to make copies or create derivative works of the code [[ref]](https://www.fsf.org/licensing/contributor-faq).

However, your employer can have very broad claims to any material that you develop (including code) and in some cases they may be the copyright holder (depending on the terms of your employment). Their claims can even extend to material you create in your free time, and may cover any patentable inventions, as well as the copyright on the code itself. If you are a student, universities can claim your work even if the work is not directly related to your studies [[ref]](https://www.fsf.org/licensing/contributor-faq).

To avoid these problems, **contributors to PROTEUS must sign a contribution license agreement** (CLA). This is a legal document in which you state you are entitled to contribute to PROTEUS (and its ecosystem) and are willing to have it used in distributions and derivative works (see license above). Signing a CLA avoids legal ambiguity as to the origins and ownership of any particular piece of code. The CLA also ensures that once you have provided a contribution, you cannot try to withdraw permission for its use at a later date [[ref]](https://www.djangoproject.com/foundation/cla/faq/).

PROTEUS and its submodules are an academic exercise. Their principle purpose is to generate data and other material which are then used to make scientific conclusions and write papers. Any such papers are **not** covered by the CLA or the licenses of the PROTEUS ecosystem. It is generally expected that the primary author of a paper arising from a particular project is the person who contributed the most work to that specific project. For a paper written using PROTEUS simulations, it is not **required** that authorship be provided to the contributors of PROTEUS. However, we ask that:

1. authorship is offered to the contributors of PROTEUS on based on the relevance of their contributions relative to the paper,
2. appropriate credit is provided in the Acknowledgements section of the paper,
3. the Maintainers are made aware of when PROTEUS results are used in a scientific paper.

<b>
In summary:

* you generally own all of the code you write and material you create,
* you give (via the CLA) irrevocable permission for it to be used under the License terms,
* you are requested to give credit and offer authorship on papers where appropriate.
</b>

More information can be found [here](https://en.wikipedia.org/wiki/Open_source), [here](https://opensource.guide), [here](https://oziellaw.ca/navigating-open-source-software-ownership-licensing-and-commercialization/), [here](https://contributoragreements.org), and [here](https://fossa.com/learn/open-source-licenses/).

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

### Linting

Linting is a term for static code analysis to flag programming errors,
bugs, stylistic errors and [suspicious constructs](https://en.wikipedia.org/wiki/Lint_(software)).
PROTEUS uses [`ruff`](https://astral.sh/ruff) for linting.
The linting [rules](https://docs.astral.sh/ruff/rules/) are defined in [`pyproject.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/pyproject.toml).

This check are run automatically via a Github Action: [codestyle](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/codestyle.yaml).

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

The tests are run automatically via a Github Action: [tests](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/tests.yaml).

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
