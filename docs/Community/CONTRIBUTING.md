# Contributing guidelines

This page provides an overview on contributing to PROTEUS itself and the ecosystem.
Anyone who makes a Pull Request to the `main` branch should read this document *fully* first.

## Licensing and credit

PROTEUS and its submodules are *free* software and also *open source* software. Roughly, this means that the users have the freedom to run, copy, distribute, study, change and improve the software. The term "free software" is a matter of liberty, not price or monetary value [[ref]](https://www.gnu.org/philosophy/free-sw.html).

PROTEUS is made available under the [Apache 2.0 License](https://choosealicense.com/licenses/apache-2.0/) which permits [[ref]](https://choosealicense.com/licenses/apache-2.0/) commercial use, unlimited distribution, modification of the code elsewhere, and private use. However, there are caveats to these terms: contributors have no liability, the code has no warranty, and a trademark may not be applied [[ref]](https://opensource.com/article/18/2/apache-2-patent-license).

The license only specifies how the material may be used by both developers and non-developers alike.  By default, as a work is created its copyright belongs to the person who created it [[ref]](https://assets.publishing.service.gov.uk/media/5a7eaf0ae5274a2e87db13f3/c-notice-201402.pdf). Although in some cases your employer/university may be the copyright holder of work you create [[ref]](https://www.fsf.org/licensing/contributor-faq). PROTEUS is not 'owned' by a single entity; the individual parts of the framework are owned by those who made them, and licensed for use and modification by others. More information can be found [here](https://opensource.guide) and [here](https://fossa.com/learn/open-source-licenses/).

The principle purpose of PROTEUS is to generate data to make scientific conclusions and write papers. It is generally expected that the primary author of a paper is the person who contributed the most work to that project. We ask that:

1. authorship is offered to the Contributors to PROTEUS based on the relevance of their work in a paper,
2. appropriate credit is provided in the Acknowledgements section of the paper,
3. the Maintainers are made aware of when PROTEUS results are used in a scientific paper.

A suggested acknowledgement is:
> We thank the people who have contributed to PROTEUS and its broader ecosystem for their support and enabling the scientific outputs of this paper. PROTEUS (version XX.XX.XX) may be found online at https://github.com/FormingWorlds and archived on Zenodo, where each release carries its own DOI (all versions: https://doi.org/10.5281/zenodo.21358380)

<b>
In summary:

* you generally own all of the code you write and material you create,
* you give irrevocable permission for the code to be used under the license when distributed,
* you are requested to offer authorship and give credit in papers as appropriate.
</b>

PROTEUS would not exist without the efforts of the wider community. Contributions from research scientists, software developers, students, and many others have made development of the current framework possible. Thank you for your interest in contributing to PROTEUS, and the immense task of simulating the lifetimes of entire planets and stars.

"Alone we can do so little; together we can do so much." - Helen Keller

## How do I contribute?

Contributing to PROTEUS is relatively straightforward. We use Git to manage the source code, and GitHub to host it online. Here is a simple workflow:

1. Download the code and make sure that it runs on your machine
2. Create a new 'branch' called `MY_BRANCH` using Git: `git checkout -b MY_BRANCH`
3. Make changes to the code as you so desire
4. Add these changes to the repository: `git add .`
5. Commit these changes with a message: `git commit -m MESSAGE`
6. Push these changes to GitHub: `git push -u origin MY_BRANCH`
7. When you've got a neat set of changes, make a 'pull request' on GitHub [here](https://github.com/FormingWorlds/PROTEUS/pulls). This makes you a Contributor to the project.
8. One of the Maintainers of the project will review the request.
9. When ready, the changes will be merged into the `main` branch and are made live!

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

### Large files, output, and input data

Large files should **not** be committed to the repository. This means that model results, plots, and files you create during analysis should not be be staged and committed to a PROTEUS online branch. Including these (even accidentally) in the repository will make Git operations sluggish and make version control tricky, as Git is only meant for managing text (e.g. code) files.

You can make files/folders invisible to Git by prepending `nogit_` to their names. For example, anything in a folder called `nogit_analysis/` will be ignored by Git. Large files could then be safely placed in this folder. Model outputs are generated in the `output/` folder, which is also ignored by Git.

PROTEUS relies on input data files that can be potentially large. These are stored outside the Git repository in [Zenodo](https://zenodo.org/communities/proteus_framework/), and mirrored on [DataverseNL](https://dataverse.nl/dataverse/Proteus_Fr). When running PROTEUS, it will automatically download the necessary input data and store it locally inside your FWL data folder defined by the `FWL_DATA` environment variable in your shell. Placing large files in this folder allows them to be kept on 'storage' file systems on clusters to avoid reaching your allocation limits.

A dataset is provisioned through [fwl-io](https://github.com/FormingWorlds/fwl-io), which pins it to a Zenodo version DOI and its DataverseNL mirror and verifies it against a committed checksum registry. [Reference data](../Reference/data.md) lists the datasets.

If you want to add new input data you can either update a Zenodo record or create a new one.

1. **Updating a record** For instance, if you want to add a new stellar spectra, you will update the `stellar_spectra/Named` record on Zenodo, making sure the file you add is in a consistent format with that of the other files contained in the record.
2. **Creating a record** If you want instead to add a new model for the absorption properties of the atmosphere, you will create a new Zenodo record named `spectral_files/model_name/number_of_bands` and upload the model files in a consistent format with that of the other spectral files used in PROTEUS. It is important for clarity and data management that the name of the records follow the organisation of the FWL data repository consistently. The next step is to declare the record in a manifest (see below) so that fwl-io downloads it; a dataset that a submodule reads (e.g. Mors, Aragog) is declared in that module's manifest or in the shared fwl-io manifest.

Declare a dataset that only PROTEUS reads as a table in `src/proteus/data/proteus_manifest.toml`, keyed by the location it should occupy below `FWL_DATA`, then run `fwl-io sync src/proteus/data/proteus_manifest.toml` and commit the registry file it writes alongside the manifest.

Once the record is on Zenodo, its DataverseNL mirror is created with `fwl-io mirror` and published with `fwl-io mirror-publish`, and its DOI goes in the `dataverse` field beside the `zenodo` one; `fwl-io check-mirrors` confirms that the mirror serves the registry.

### Linting

Linting is a term for static code analysis to flag programming errors,
bugs, stylistic errors and suspicious constructs [[ref]](https://en.wikipedia.org/wiki/Lint_(software)).

PROTEUS uses [`ruff`](https://astral.sh/ruff) for linting. The linting [rules](https://docs.astral.sh/ruff/rules/) are defined in [`pyproject.toml`](https://github.com/FormingWorlds/PROTEUS/blob/main/pyproject.toml). This check is run automatically via a Github Action: [code-style](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/code-style.yaml).

You can run `ruff` locally using one of these commands:

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

Our tests are run automatically via a Github Action: [CI - Fast PR Checks](https://github.com/FormingWorlds/PROTEUS/blob/main/.github/workflows/ci-pr-checks.yml). You can also run the tests for yourself using the command:

```console
pytest
```

Or to run a specific test containing a keyword:

```console
pytest -k keyword
```

To check the 'coverage' of the tests:

```console
coverage run -m pytest
coverage report  # to output to terminal
coverage html    # to generate html report
```

The 'coverage' of the tests describes the fraction of the code which is executed while tests are being run. However, care should also be taken to ensure that the *output* of the tests meets expectations.

The test files are located in the `tests/` folder. The pytest library will automatically find functions contained within Python files in this folder, and run them as tests. A test fails if a function raises an errors or fails an assertion.

### Building the documentation

The documentation is written in [Markdown](https://www.markdownguide.org/basic-syntax/) and generated using [Zensical](https://zensical.org/docs/get-started/). Zensical is fully compatible with a `mkdocs.yml` configuration file, which we use for the build process.

To build the documentation for yourself:

```console
pip install -e '.[docs]'
zensical serve
```

This will generate the markdown files and serve them on a local server. You can view documentation while you edit by copy-pasting the displayed URL into your browser (e.g., `http://127.0.0.1:8000`).

You can find the documentation source in the [docs](https://github.com/FormingWorlds/PROTEUS/tree/main/docs) directory.
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/FormingWorlds/PROTEUS/blob/main/mkdocs.yml) under the `nav` entry.

The documentation is hosted on the [PROTEUS framework website](https://proteus-framework.org/PROTEUS/).

### Making a release

The versioning scheme we use is [CalVer](https://calver.org/), in the format `YY.MM.DD`, without a leading 'v'. Versions are derived from git tags by [`setuptools-scm`](https://setuptools-scm.readthedocs.io/); there is no hand-edited version string anywhere in the repository.

1. Tag the release on `main` and push the tag:

```console
git tag 26.05.14
git push origin 26.05.14
```

2. Create a GitHub release pointing at the tag:

```console
gh release create 26.05.14 --generate-notes
```

3. The [upload to PyPI](https://pypi.org/project/fwl-proteus) is triggered when the release is published and handled by [this workflow](https://github.com/FormingWorlds/PROTEUS/actions/workflows/publish.yaml).

## Writing Validation and Explanation pages

The Validation pages and the Explanation pages are read by new MSc and PhD students. They know physics and some Python, but they do not know this code, its history or its test framework. Write every Validation page and every new Explanation section for that reader, in simple English. Explain why each validation is there, what it tests and why it is important. Keep every technical item: the reference, the tolerance, the test id, the command and the plot. Add the parts for the reader around them; do not delete content to make a page short.

The page [Validation: metal reservoir](https://proteus-framework.org/CALLIOPE/Validation/metal.html) of CALLIOPE uses this layout and is a good model.

### The top of a Validation page

Start each Validation page with a section `## What this page checks` of 3 to 6 sentences. Say:

- which part of the model the page covers, in plain words (for example "how much of each volatile element goes into the metal core");
- what validation means here: we run the code on a case where another source gives the right answer, such as an analytical result, a published table or measurement, or a second independent code, and the test checks that the code gives that answer;
- which output of a PROTEUS run would be wrong if this part of the code were wrong;
- how each case below is laid out.

### The layout of each validation case

Use this order for every case. The bold lead-ins are always the same, so that a reader can scan the page.

```markdown
## <Title that says what is checked, in plain words>

**Why this check exists.** 1 to 3 sentences: what could go wrong in the code, and what wrong result a user would then get.

**What it tests.** The setup in plain words first, then the expected result. Define every symbol at first use ("D, the partition coefficient: the ratio of the concentration in the metal to that in the melt").

**How we know the right answer.** The reference in plain words: "this follows from mass balance alone", "Tagawa et al. (2021) measured these 6 values in the laboratory".

**Why it matters.** 1 to 2 sentences: which model result or which other module depends on this.

<plot pair, light and dark>

**How to read the plot.** 1 to 3 sentences: the axes, what the lines or points are, and what a pass looks like ("the points lie on the line").

??? info "Test details"
    - Reference: ...
    - Tolerance: ...
    - Test id: ...
    - Command: ...
```

- The box "Test details" holds the Reference, Tolerance, Test id and Command lines.
- `??? info` makes a collapsed box. It needs `pymdownx.details` in the `markdown_extensions` of `mkdocs.yml`. A repository without it adds the extension directly after `admonition`, in the same pull request.
- A case without a plot leaves out the plot and **How to read the plot.**
- A case that fails, or that has a known limit, says so in plain words in its own line **Known limit.**, before the box.

### New Explanation sections

- Start with the question that the section answers, in 1 sentence ("How much hydrogen can a liquid iron core hold, and how does it change the core density?").
- Then give the short answer in 2 to 3 sentences, before any equation.
- Define every symbol and every term at first use, with its unit.
- Write one idea per paragraph. Put an equation after the sentence that says what it means.
- Give 1 worked example with real numbers where it helps ("for an Earth-mass planet at 4000 K, ...").
- End with a short line **Where this is checked.** that links to the Validation case.
- Put bounds and limits in a list. For each one, say what happens when a value is outside it: an error, a warning or a clamp.

### Simple English

- Keep sentences to about 25 words or fewer, with one statement per sentence.
- Use the active voice where a person or the code acts ("the solver rejects the root").
- Use one term for one thing, in the same way on every page.
- Define each abbreviation at first use on each page (IW, EOS, CMB).
- Give numbers with units. In prose, use a word where it is clearer than an exponent ("100 times").
- Describe the code as it is now, without internal labels or the history of how the code changed.

### Formatting that guides the reader

- Headings say what is checked or explained, not code names. Put the code name in the text.
- A change of a heading changes its anchor. Where a link points at a heading (the Validation index table, another page, a pull-request text), keep the heading text, or change every link and search the docs for the old anchor.
- Text that a test or a tool reads stays intact: number tags such as `<!--k:...-->`, test ids in backticks, and the rows of the table in `docs/Validation/index.md`. Run every docs test of the repository after a change.
- Use lists for 3 or more parallel items, and tables only when the reader compares values.
- Give every plot alt text that names what it shows, and a sentence that explains it.
