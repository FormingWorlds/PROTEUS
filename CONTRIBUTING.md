# Contributing guidelines

## Development

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
