# Writing docs

This page says how to write or change a page of the PROTEUS documentation or of the documentation of a module.
The reader of every page is a new MSc or PhD student who knows physics and some Python, but not this code, its history or its test framework.
Write each page so that this reader can follow it without help.

## Steps

1. Choose the page type from the [table of page types](#page-types) and find the page where the content belongs.
2. Write for the reader, in [plain words](#plain-words-and-numbers), with a unit on every number.
3. Cite each paper in the [reference form](#citing-a-paper): the author and the year, followed at once by a footnote.
4. Keep the page within the [size budget](#size-and-where-detail-goes), and move detail to the page whose job it is.
5. Add each plot as a [light and dark pair](#plots) with alt text.
6. Add a new page to the `nav` entry of `mkdocs.yml`, and link to it from the pages where a reader meets its topic.
7. Build the docs with `zensical serve` (see [Building the documentation](CONTRIBUTING.md#building-the-documentation)) and read the page on the local preview, in light and dark mode.

## Checklist before a pull request

- The page has one type, and its content fits that type.
- Every symbol, term and abbreviation is defined at first use on the page, with its unit.
- Every paper is cited as `Name (Year)[^cite-key]`, and each footnote is defined once at the end of the page.
- No DOI appears in the text.
- The page and each section are within the size budget, or the pull request says why the page needs the length.
- Every plot has a light and a dark file and alt text that names what it shows.
- A change to physics or chemistry comes with all 4 parts: Explanation, Tutorial, Validation and Reference.
- The page builds, its links work, and you have read it on the preview.

## Page types

Each piece of content has one home. Put it on the page whose job it is, and link to it from the others.

| Page type | Its job | What it holds |
|---|---|---|
| Tutorial | learning by doing | a worked run with its commands and its output |
| How-to | one task | the steps for that task |
| Explanation | understanding | how a mechanism works, why it matters, and its limits |
| Reference | looking things up | parameter tables, config keys, file formats and the API |
| Validation | trust in the results | each check, its source, its tolerance, its test id and its plot |

- In a Tutorial, every code block runs on the current code, and every number comes from that run.
- An Explanation section keeps 1 worked example where it helps, and links to the Reference and Validation pages for the rest. Its layout is in [New Explanation sections](CONTRIBUTING.md#new-explanation-sections).
- A Validation page follows the layout in [Writing Validation and Explanation pages](CONTRIBUTING.md#writing-validation-and-explanation-pages).
- A Reference page for a module states the units and the ranges of each value. The API pages are built from the NumPy docstrings, so write the units and ranges there.

## A change to physics or chemistry

A pull request that adds or changes physics or chemistry, for example a new law, reservoir, closure or boundary treatment, or a change to how a result is computed, carries all 4 parts in the same pull request:

- An Explanation page or section on how it works, linked from every Explanation and How-to page where a reader meets it.
- A runnable Tutorial that shows how to use it. For a part that users extend, such as a law, a table or a module, it also shows how to add a new one.
- The Validation cases that check it.
- The Reference: the API pages of the new or changed functions, and the reference for every new config key, file field or output.

A change to CI or a refactor that changes no result needs no tutorial.

## Size and where detail goes

- An Explanation page has at most 2500 words, and a section (the text under one heading) at most 300. Count the words of the prose, without code, math, footnote definitions and table rules.
- When a page is over the budget, move the detail to the page where it belongs and link to it, or say in the pull request why the page needs the length.
- Move detail, never drop it. A reference, a tolerance, a test id, a command or a plot that leaves a page goes to a named page in the same pull request.

## Plain words and numbers

- Keep sentences to about 25 words, with one statement per sentence.
- Use the active voice where a person or the code acts ("the solver rejects the root").
- Use one term for one thing, in the same way on every page.
- Give every number its unit. In prose, use a word where it is clearer than an exponent ("100 times").
- Describe the code as it is now. Leave out its history and internal labels.

More rules on sentences, terms and headings are in [Simple English](CONTRIBUTING.md#simple-english) and [Formatting that guides the reader](CONTRIBUTING.md#formatting-that-guides-the-reader).

## Citing a paper

In the text, write the author and the year, followed at once by a footnote:

- 1 author: `Abe (1995)[^cite-abe1995]`
- 2 authors: `Rumpf & Gupte (1971)[^cite-rumpfgupte1971]`
- 3 or more authors: `Bower et al. (2018)[^cite-bower2018]`

Rules:

- Each paper in a list gets its own footnote.
- A later mention of the same paper on the same page needs no new footnote: `Bower et al. (2018) Eq. 12`.
- The key is `cite-` plus the surname of the first author and the year, in lowercase ASCII letters and digits. For 2 papers of one author in one year, add `a` and `b` to the key and the text: `Abe (1995a)[^cite-abe1995a]`.
- Put the definitions at the end of the page, 1 per line.
- Do not put a DOI in the text, as a `doi.org` link or in the form `doi:10.xxxx`. Software and data releases on Zenodo are the exception: link them in the text.
- Define every footnote that the page uses once, and use every definition.
- Compare every field of a footnote with the paper itself, on the DOI page or on SciX, not with your memory.

A definition holds, in this order: the initials and surnames of all authors, separated by commas; the title in italics, linked to the DOI; the journal, the volume and the pages with a hyphen; the year; and the SciX link when the paper has one. A book or chapter without a DOI keeps its title in italics, without a link.

Example, for a paper that the PROTEUS docs cite:

```markdown
The interior model follows Bower et al. (2018)[^cite-bower2018].

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
```

### Modules with Julia documentation

A module whose `docs/` folder holds `make.jl` builds its pages with Documenter, which does not render `[^cite-...]` footnotes.
On such a page, cite with a link to a heading of the bibliography page of the site, and put the full reference under that heading:

```markdown
[McKenzie (1984)](../reference/bibliography.md#Silicate-Melting-and-Magma-Rheology)
```

The link text starts with the surname of the first author and holds the year. The section under the linked heading holds that surname and that year.

## Plots

- Add each plot as a pair: a file for light mode and a file for dark mode, shown with `#only-light` and `#only-dark`.
- Give each plot alt text that names what it shows, and a sentence that says how to read it.
- A Validation plot follows the rules in [The layout of each validation case](CONTRIBUTING.md#the-layout-of-each-validation-case).
