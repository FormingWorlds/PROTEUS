# Documentation figure sources

## Code-architecture diagram

`docs/assets/proteus_architecture.svg` and its dark counterpart are generated
from `arch_final.json`, which holds every shape, edge, label and link of the
diagram in one coordinate model. `gen_tikz.py` renders that model to a
standalone TikZ document, once per colour mode; `add_svg_links.py` re-attaches
the clickable regions, which the PDF-to-SVG conversion does not carry over.

Rebuild both variants with:

```bash
bash tools/figures/build_architecture.sh
```

The model is the single source for both modes. Neutral colours (text, hairlines,
surfaces) are mapped per mode in `gen_tikz.py`; the domain hues are identical in
both, and each label takes the ink that contrasts with the surface it sits on.
Label positions are stored as measured baselines, so the line breaking is fixed
in the model rather than left to the typesetter.

### Updating from draw.io

Edit the diagram in draw.io, export it as SVG (File > Export as > SVG, "Embed
images" on, "Convert labels to SVG" off), then regenerate the model and rebuild:

```bash
python3 tools/figures/extract_drawio.py path/to/export.svg tools/figures/arch_final.json \
    --logo-dark path/to/dark_logo.png
bash tools/figures/build_architecture.sh
```

`extract_drawio.py` measures the labels in headless Chromium (Playwright), so
line breaks match draw.io. A shape in a new colour needs an entry in its
`PALETTE` (and, for a neutral, in `DARK_MAP` in `gen_tikz.py`). It overwrites
the logos in `img/` with the ones in the export. Do not commit the export to
`docs/assets/`.

### Editing the model by hand

To change the diagram without draw.io, edit `arch_final.json`:

- shapes are `rect` (rounded rectangles, `rx` is the corner radius) and `path`
  (hexagons, the decision rhombus, the archive parallelogram, and every edge),
  in the SVG coordinate system with y increasing downwards;
- `text` items carry one entry per line in `mlines`, with the line box the label
  occupies and the ink colour of its light-mode form;
- `href` on any item makes it clickable, both in the PDF and in the SVG;
- `cell` groups the primitives that belong to one diagram element.

A label's baseline is `top + BASELINE_K[size] * size`, so to place a new one,
set `top` to `anchor - 7.3` for a single 13 px line centred on `anchor`, or to
`anchor - 5.5` for a 10 px line; a two-line 13 px label puts its lines at
`anchor - 14.5` and `anchor + 1.09`. `bottom` is `top + 1.2 * size`. For a
centred label only the midpoint of `left` and `right` is used for placement, so
those two can span the shape the label sits in. Alignment comes from `align`
(`center`, `left` or `right`), which decides whether the line is anchored by its
midpoint or by an edge.

Colours are written in their light-mode form. `gen_tikz.py` maps the neutrals
for the dark variant and leaves the domain hues alone. A label's ink follows the
surface underneath it: on a chip whose colour is the same in both modes it keeps
the ink given here, and only a label whose surface changes between modes has its
ink recomputed by contrast.

### Code links

A link into the code names its target instead of a line, so it survives edits:

```
https://github.com/FormingWorlds/PROTEUS/blob/main/src/proteus/proteus.py#interior
```

The part after `#` is a function, class or method (`Proteus.start`) or the slug
of a comment (`############### ORBIT AND TIDES` is `#orbit-and-tides`, `# Run
offline chemistry` is `#run-offline-chemistry`), and must match exactly once.

- The docs deploy runs `code_links.py --deploy`, which turns each target into a
  `#L<n>` link on `main`. A target that no longer resolves loses its link, with
  a warning.
- PR CI runs `code_links.py --check`, which fails on a target that no longer
  resolves or a hard-coded `#L` anchor. Renaming a linked function or banner
  means updating the link in draw.io and `arch_final.json`.
- `code_links.py` without options refreshes the lines in the committed SVGs from
  `origin/main`.

## Module schematic

`docs/assets/proteus_modules_schematic.svg` and its dark counterpart are drawn
in draw.io, with the editable diagram embedded in each file. Their labels are
native SVG text, so every renderer shows them, not only browsers. After
exporting both variants (File > Export as > SVG, "Include a copy of my diagram"
on, "Convert labels to SVG" off, as it drops italics and subscripts), convert
the labels:

```bash
python3 tools/figures/svg_labels_to_text.py docs/assets/proteus_modules_schematic.svg
python3 tools/figures/svg_labels_to_text.py docs/assets/proteus_modules_schematic_darkmode.svg
```

The script needs Playwright and Arial or a font with Arial's widths (Liberation
Sans, Arimo, Nimbus Sans); otherwise labels wrap differently than in draw.io.
