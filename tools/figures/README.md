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

The model can be regenerated from a draw.io SVG export of the diagram, so the
usual way to change the figure is to edit it in draw.io:

1. In draw.io, export the light-mode diagram as SVG (File > Export as > SVG,
   with "Embed images" on and "Convert labels to SVG" off) and, if the logo
   changed, the dark-mode export as well.
2. Regenerate the model and rebuild:

```bash
python3 tools/figures/extract_drawio.py path/to/export.svg tools/figures/arch_final.json \
    --logo-dark path/to/dark_logo.png
bash tools/figures/build_architecture.sh
```

`extract_drawio.py` renders the export in headless Chromium (Playwright) and
measures every label line there, so line breaks match draw.io. It maps the
draw.io colours to the model palette through its `PALETTE` table; a shape in a
new colour passes through unchanged and needs an entry there (and, if it is a
neutral, one in `DARK_MAP` in `gen_tikz.py`). Labels keep their italics and
subscripts from the draw.io HTML. The embedded logo is written to
`img/arch_light_img00.png`; `--logo-dark` takes the PNG for the dark variant.
Never commit the draw.io export itself to `docs/assets/`: the build overwrites
those two files.

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

The loop-stage boxes link to the call site in `src/proteus/proteus.py` that runs
them, and module boxes link to the file that implements them. Both are line
anchors on `main` and are worth re-checking when the loop is restructured.

## Module schematic

`docs/assets/proteus_modules_schematic.svg` and its dark counterpart are drawn
in draw.io; the editable diagram is embedded in each file's `content`
attribute. Their labels are native SVG text rather than HTML in a `foreignObject`
with a raster fallback, so every renderer, not just a browser, shows the current
wording. Re-exporting from draw.io restores the HTML-plus-raster form, and the
labels then have to be converted back.
