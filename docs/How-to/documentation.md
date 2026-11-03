# Documentation development

This page explains how to build, preview, and extend the PROTEUS documentation locally.

## Overview

The PROTEUS documentation is written in [Markdown](https://www.markdownguide.org/basic-syntax/) and generated using [Zensical](https://zensical.org/docs/get-started/).

Zensical is fully compatible with the project's `mkdocs.yml` configuration, which is used for the documentation build process and site structure.

The documentation source files live in the `docs/` directory of the repository.

## Local development

To build and preview the documentation locally, install the documentation dependencies and start the development server:

```console
pip install -e '.[docs]'
zensical serve
```

This command will generate the documentation and start a local server for previewing changes as you edit.

Once the server is running, open the URL shown in your terminal in a browser. In most cases, it will look similar to:

```console
http://127.0.0.1:8000
```
!!! note "Using a different port"
    If port 8000 is already busy, you can build the documentation on a different port:
    ```console
    zensical serve -a localhost:8001
    ```

!!! note "SSH connection"
    If you are connected to a remote server via SSH, for example on the Habrok, Snellius, or Kapteyn cluster, you may need to create an additional SSH tunnel. On your **local machine**, run:

    ```console
    ssh -L 4000:localhost:8000 account@server
    ```

    Then open `http://localhost:4000` in your browser.

## Documentation structure

Documentation source files are located in the [`docs`](https://github.com/FormingWorlds/PROTEUS/tree/main/docs) directory, and mostly follow the [Diátaxis](https://diataxis.fr/) approach of documentation. This means they are categorised into:

- **Tutorials**: learning-oriented guides for new users, such as a first workflow tutorial.
- **How-to guides**: task-oriented instructions, such as installation, configuration, or test development.
- **Explanations**: conceptual material, such as model overviews or code architecture.
- **Reference**: lookup material, such as reference data, API-like documentation, or bibliographies.

Using this structure helps keep pages focused on a single purpose. When adding a new page, try to decide first which of these four categories it belongs to.

## Layout of the `docs/` directory

Within the `docs/` directory, documentation pages are generally organised according to the categories above. In addition, several extra directories contain assets and project-level documentation material.

The current `docs/` folder structure is shown below:

```text
.
├── assets
│   └── logos
├── Community
├── Explanations
├── funding.md
├── getting_started.md
├── How-to
├── index.md
├── javascripts
├── overrides
├── paper
├── Reference
├── stylesheets
├── submodules.md
├── Tutorials
└── Validation
```

The main directories and files are used as follows:

- `Tutorials/`: contains tutorials.
- `How-to/`: contains how-to guides.
- `Explanations/`: contains explanation pages.
- `Reference/`: contains reference information.
- `Community/`: contains community pages such as the code of conduct, contributing guide, and contact information.
- `assets/`: contains static assets such as images, diagrams, and logos.
- `stylesheets/`: contains custom CSS used by the documentation site.
- `javascripts/`: contains JavaScript files used for additional frontend behaviour.
- `overrides/`: contains theme or template overrides used by the site.
- `paper/`: contains material related to the PROTEUS JOSS paper.
- `Validation/`: per-source-file scientific validation pages, recording the
  published benchmarks, analytical limits, and cross-implementation checks
  that pin each physics module's behavior. They form the top-level `Validation` tab, after Explanations and before Reference; the plots of their cases are in `assets/validation/` (see [Scientific validation](../Validation/index.md)).
- `index.md`: the **landing page** of the documentation (Home).
- `getting_started.md`: a page to get the user started, just below Home.
- `funding.md`: contains our sponsors.
- `submodules.md`: introduces the documentation of all PROTEUS' submodules.

## Adding a new page

When adding a new documentation page:

1. Create the Markdown file in the appropriate location under `docs/`.
2. Add the page to the navigation in `mkdocs.yml` under the `nav` section.
3. Start the local preview with `zensical serve`. If the preview is already open, saving `mkdocs.yml` should refresh the page (check this in the terminal).
4. Check that the page renders correctly and appears in the expected place in the navigation.
5. Verify that internal links, images, and code blocks display correctly.

For example, a page stored at `docs/Tutorials/model_earth.md` might be added to `mkdocs.yml` like this:

```yaml
nav:
  - Tutorials:
    - Simulate Earth: Tutorials/model_earth.md

```

!!! info "Documentation files are relative to `docs/` directory"
    Zensical looks for documentation files relative to the `docs/` directory. In `mkdocs.yml`, add files like this: `Tutorials/model_earth.md` and not like this: `docs/Tutorials/model_earth.md`.

## Update the tutorial figures

The figures of the pages under `docs/Tutorials` are drawn by `tools/make_docs_figures.py` from the output folder of a run, as a pair: `<name>.avif` for the light colour scheme of the site and `<name>_dark.avif` for the dark one. A page that the manifest lists under `exempt_pages` is the exception: its figures do not come from the tool. The list is empty. Both files of a pair follow the PROTEUS visual language through the [`proteus-mpl`](https://pypi.org/project/proteus-mpl/) matplotlib style, which the `docs` and `develop` extras install. The manifest `tools/docs_figures.toml` lists the figures; the comment above each entry names the config that makes its run.

To redraw the figures, give the folder that holds the run folders:

```console
python tools/make_docs_figures.py --runs output/
python tools/make_docs_figures.py --runs output/ --only earth_analogue.md
```

The second form draws the figures of one page; `--only` also takes the `id` of one figure. Each name in the `runs` of an entry is the folder of that name under `--runs`. `--run <name>=<folder>` gives the folder of one run directly and can be repeated. It does not reach an entry whose arguments name a run folder below `--runs`, and the tool then stops with an error: for the entries of the CHILI page, put or link the run folders under `--runs` with the names of the manifest. The quick start run needs it, because PROTEUS names that folder after the start time of the run. The figures in the repository come from the runs of the release, where the quick start run and the Earth and Moon run have other folder names:

```console
python tools/make_docs_figures.py --runs <runs> \
    --run quick_start=<runs>/run_20261012_175211_ae08 \
    --run tutorial_earth_moon=<runs>/tutorial_earth_moon_08myr
```

`--out <root>` writes the files below another root than the repository, for a trial draw. `--manifest <file>` reads another manifest.

The tool reads the run folders and never writes into them. For each figure and variant it prints a table of the text geometry: texts that the saved file cuts off (an error), texts outside the figure box, tick labels that intersect or crowd, and legends or labels that lie on data. Only a text that is cut off fails the command, and the files of that figure are then not written; the other rows tell you where to look. Look at both variants of a figure before you commit it. The size of a figure in pixels depends on the matplotlib version, so redraw all figures of a page with the same environment; the tool prints the versions it runs with.

Next to the entries, the manifest holds 4 values:

- `proteus_mpl`: the lowest version of the style that the tool accepts.
- `avif_quality`: the quality of the AVIF encoder, from 1 to 100.
- `pages`: the folder of the pages that `--check` reads.
- `exempt_pages`: the pages in that folder whose figures do not come from the tool; `--check` names and skips them.

To add a figure:

1. Add a `[[figure]]` table to `tools/docs_figures.toml` with an `id`, the `page`, the `asset` path stem under `docs/assets`, the `runs` (the names of every run the figure reads) and the `maker`. An entry draws both variants; `variants = ["light"]` or `["dark"]` limits it to one.
2. For one of the 4 plot functions of `proteus.plot` that the tool can call, the maker is `proteus:<name>`: `global` (`plot_global`, with the arguments `logt` and `tmin`), `orbit` (`plot_orbit`), `fluxes_global` (`plot_fluxes_global`) or `lovenumber` (`plot_lovenumber` on every tides snapshot of the run, as the run draws it at its end). For any other plot, write a function in a script under `tools/` and name it as `tools/<script>.py:<function>`. The tool calls it as `maker(run_dirs, out_path, **args)`, where `args` is the `args` table of the entry; a string in it can hold `{runs}` (the folder given with `--runs`) and `{chili_repo}` (the folder given with `--chili-repo`). The function draws under the active matplotlib style: it sets no font, no surface colour and no neutral colour of its own (`proteus.utils.plot.style_neutral` gives the neutral colours of the active style) and takes its series colours from the colour cycle of the style. The exception is a figure that follows a fixed colour table of its own: the figures of the CHILI page follow the colours of the CHILI analysis code. It saves the figure with `Figure.savefig`, without a `dpi` argument, and the file is at `out_path` when it returns; the tool measures the text geometry in that `savefig` call.
3. Embed the pair in the page, the light file first:

    ```markdown
    ![Alt text](../assets/tutorials/my_figure.avif#only-light){ width="100%" }
    ![Alt text](../assets/tutorials/my_figure_dark.avif#only-dark){ width="100%" }
    ```

4. Run `python tools/make_docs_figures.py --check`. It needs no run folder and fails when a page embeds an image without a manifest entry, when an entry has no page or lacks a file, or when a page does not embed the two files as a pair. A unit test runs the same check.

## Before opening a pull request

The `Documentation links and anchors` check on every pull request runs `zensical build --clean --strict`, which aborts the build on a link to a page or an anchor that does not exist. Run the same command locally to see the failure and its file and line before you push. Image paths are not checked, so confirm those in the local preview.

Before submitting documentation changes, it is a good idea to check that:

- the site builds locally without errors,
- the page appears in the correct navigation section,
- internal links resolve correctly,
- headings are ordered sensibly,
- code examples are formatted correctly,
- equations are displayed correctly,
- and any new assets are included and displayed properly.

If the change accompanies a code update, make sure the documentation reflects the current behaviour of the code.
