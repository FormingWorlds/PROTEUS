# Writing docs

This page explains what to write on a page of the PROTEUS documentation or of a module's documentation, and how to cite a paper.
How to build the site, add a page and draw the tutorial figures is in [Developing documentation](../How-to/documentation.md).

The reader of every page is a new MSc or PhD student.
This reader knows physics and some Python, but not this code, its history or its test framework.

## Steps

1. Choose the page type from the [table of page types](#page-types).
2. Put the content on the page whose job it is.
3. Write for the reader, in [plain words](#plain-words-and-numbers), with numbers and their units.
4. Cite each paper in the [citation form](#citing-a-paper): the author and the year, followed at once by a footnote.
5. Keep an Explanation page within the [size budget](#size-and-where-detail-goes), and move detail to the page whose job it is.
6. Add each plot as a [light and dark pair](#plots) with alt text.
7. Add a new page to the `nav` entry of `mkdocs.yml`.
8. Link to the page from the pages where a reader meets its topic.
9. Preview the page with `zensical serve` and read it in light and dark mode.
10. Run `zensical build --clean --strict`, as the pull-request check does. It stops on a link or an anchor that does not exist.

## Checklist before a pull request

- Define every symbol, term and abbreviation at first use on the page, with its unit.
- Cite every paper as `Name (Year)[^cite-key]`. On a page built with Documenter, cite as a [link to the bibliography](#modules-with-julia-documentation).
- Define each footnote once at the end of the page, and use each definition.
- Keep DOIs out of the text. A link to a Zenodo release is the exception.
- Keep an Explanation page and each of its sections within the size budget, or say in the pull request why it needs the length.
- Give every plot a light and a dark file, and alt text that names what it shows.
- Add all 4 parts for a [change to physics or chemistry](#a-change-to-physics-or-chemistry).
- Run `zensical build --clean --strict`, and read the page on the preview.

## Page types

Each piece of detail goes to the page whose job it is. Link to it from the other pages.

| Page type | What it holds |
|---|---|
| Tutorial | a worked run with its commands and its output |
| How-to | the steps for one task |
| Explanation | how a mechanism works, why it matters, and its limits |
| Reference | parameter tables, config keys, file formats and the API (application programming interface) |
| Validation | each check, its source, its tolerance, its test id and its plot |

- In a Tutorial, every code block runs on the current code, and every number comes from that run.
- An Explanation section has 1 worked example where it helps, and links to the Reference and Validation pages for the rest. Its layout is in [New Explanation sections](CONTRIBUTING.md#new-explanation-sections).
- A Validation page follows the layout in [Writing Validation and Explanation pages](CONTRIBUTING.md#writing-validation-and-explanation-pages).
- A Reference page states the units and the ranges of each value. The API pages come from the NumPy docstrings, so write the units and ranges there.

## A change to physics or chemistry

A pull request that adds or changes physics or chemistry carries 4 parts.
Examples are a new solver part, law, reservoir, closure or boundary treatment, or a change to how a result is computed, numerics included.

- An Explanation page or section on how it works. Link to it from every Explanation and How-to page where a reader meets it.
- A runnable Tutorial that shows how to use it. For a part that users extend, such as a law, a table or a module, it also shows how to add a new one.
- The Validation cases that check it.
- The Reference: the API pages of the new or changed functions, and the reference for every new config key, file field or output.

A refactor that changes no result needs no Tutorial.
Nor does a pure change to CI (continuous integration, the checks that run on a pull request).

## Size and where detail goes

- An Explanation page has at most 2500 words, and a section (the text under one heading) at most 300.
- Count the words of the prose, without code, math, footnote definitions, HTML comments and table rules.
- When an Explanation page is over the budget, move the detail to the page where it belongs and link to it. If the page needs the length, say why in the pull request.
- Move detail, never drop it. A reference, tolerance, test id, command or plot that leaves a page goes to a named page in the same pull request.

## Plain words and numbers

- Write sentences of at most about 25 words, with one statement per sentence.
- Give numbers with units.
- Describe the code as it is now, without its history or internal labels.
- Do not use em or en dashes. A page range uses a hyphen.

The full rules on sentences, terms, abbreviations and headings are in [Simple English](CONTRIBUTING.md#simple-english) and [Formatting that guides the reader](CONTRIBUTING.md#formatting-that-guides-the-reader).

## Citing a paper

In the text, write the author and the year, followed at once by a footnote:

- 1 author: `Abe (1995)[^cite-abe1995]`
- 2 authors: `Rumpf & Gupte (1971)[^cite-rumpfgupte1971]`
- 3 or more authors: `Bower et al. (2018)[^cite-bower2018]`

Citation rules:

- Each paper of a list gets its own footnote: `Abe (1995)[^cite-abe1995] and Fei et al. (2021)[^cite-fei2021]`.
- A later mention of the same paper on the same page needs no new footnote: `Bower et al. (2018) Eq. 12`.
- The key is `cite-` plus the surname of the first author and the year, in lowercase letters a to z and digits: `cite-elkinstanton2008`, `cite-vanderwal2010`.
- The surname of the second author can follow the first: `cite-rumpfgupte1971`.
- For 2 papers of one author in one year, add `a` and `b` to the key and the text: `Abe (1995a)[^cite-abe1995a]`.
- Put the definitions at the end of the page, 1 per line.
- Define each footnote exactly once, and cite each definition at least once.
- Keep the DOI (digital object identifier) out of the text, as a `doi.org` link or in the form `doi:10.xxxx`.
- A software or data release on Zenodo is the exception: link it in the text.
- Do not add a reference list in another form. The footnote definitions are the list.
- Compare every field of a footnote with the paper itself, on the DOI page or on [SciX](https://scixplorer.org), not with your memory.

A footnote definition has these parts, in this order:

1. The initials and surnames of all authors, separated by commas.
2. The title in italics, linked to the DOI.
3. The journal, the volume, and the pages with a hyphen.
4. The year.
5. The SciX link, when the paper has one.

A book or chapter without a DOI keeps its title in italics, without a link.
Some older pages use another form. New and edited definitions use this one.

Example, for a paper that the PROTEUS docs cite:

```markdown
The interior model follows Bower et al. (2018)[^cite-bower2018].

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
```

### Modules with Julia documentation

A module whose `docs/` folder holds `make.jl` builds its pages with Documenter, the documentation tool of Julia.
Documenter shows a `[^cite-...]` footnote as plain text, so do not use footnotes there.
Cite with a link to a heading of the bibliography page of the site, and put the full entry under that heading:

```markdown
[McKenzie (1984)](../reference/bibliography.md#Silicate-Melting-and-Magma-Rheology)
```

- The link text starts with the surname of the first author and holds the year.
- The section under the linked heading holds that surname and that year.
- Read that section, and confirm that its entry is the paper you cite.

## Plots

Add each plot as a pair, one file for light mode and one for dark mode:

```markdown
![What the plot shows](../assets/<name>.svg#only-light)
![What the plot shows](../assets/<name>_dark.svg#only-dark)
```

- Give each plot alt text that names what it shows, and a caption sentence.
- A Validation plot also has a short note on how to read it. The rules are in [The layout of each validation case](CONTRIBUTING.md#the-layout-of-each-validation-case).
- The tutorial figures are drawn by a tool. The steps are in [Update the tutorial figures](../How-to/documentation.md#update-the-tutorial-figures).
