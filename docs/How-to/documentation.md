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

!!! note "SSH connection"
    If you are connected to a remote server via SSH, for example on the Habrok, Snellius, or Kapteyn cluster, you may need to create an additional SSH tunnel. On your **local machine**, run:

    ```console
    ssh -L 4000:localhost:8000 account@server
    ```

    Then open `http://localhost:4000` in your browser.

## Documentation structure

Documentation source files are located in the [`docs`](https://github.com/FormingWorlds/PROTEUS/tree/main/docs) directory, and mostly follow the [Diátaxis](https://diataxis.fr/) approach of documentation. This means they are categorised into:

- Tutorials (e.g. a first workflow tutorial)
- How-to guides (e.g. installation, configuration, test development)
- Explanation (e.g. model overview, code architecture overview)
- Reference (e.g. reference data, bibliography)

In the `docs` directory, documentation files are also listed under the folders described above. Additionally, the `assets` folder contains images and logos, `paper` contains material for the PROTEUS JOSS paper, `stylesheets` contains .css files for documentation formatting and javascripts contains .js files for extra javascript functions. Finally, the `Community` folder contains files such as the code of conduct, contributing guide, and a contact page.

```
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
└── Tutorials
```
