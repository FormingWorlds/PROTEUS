"""Replace the HTML labels of a draw.io SVG export with native SVG text.

draw.io writes each label as a ``<switch>`` holding the HTML in a
``foreignObject`` and a raster ``<image>`` fallback, so renderers without HTML
support show a picture of the label. This script renders the export in
headless Chromium, measures every character of every label there, and writes
each label as one ``<text>`` per line, with a ``<tspan>`` for each run of
equal style (italics, bold, subscripts and superscripts). Centred lines are
anchored at their centre, so a renderer whose fonts differ from Chromium's
keeps them centred. Line breaks, positions and colours are the ones Chromium
lays out, including colours given with ``light-dark()``, which resolve by the
``color-scheme`` of the export.

Labels are laid out in Arial or, where Arial is not installed, a font with
Arial's advance widths (Liberation Sans, Arimo, Nimbus Sans). Chromium would
otherwise fall back to a wider font and break lines that fit in draw.io, and
the breaks are fixed in the output, so the script stops if none of these fonts
is installed.

Everything outside the ``<switch>`` elements is copied byte for byte; a label
without visible text is dropped together with its raster.

Usage:
    python3 tools/figures/svg_labels_to_text.py EXPORT.svg [OUT.svg]

Without OUT the export is rewritten in place. Requires playwright with a
Chromium build.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from playwright.sync_api import sync_playwright

FONT_FAMILY = 'Arial, Helvetica, sans-serif'
HEADER = '<!-- Labels are native SVG text, so every renderer shows the same wording. -->\n'

# Arial and the fonts that share its advance widths, as local() font names
ARIAL_FACES = {
    ('normal', 'normal'): ['Arial', 'ArialMT', 'Liberation Sans', 'Arimo', 'Nimbus Sans'],
    ('bold', 'normal'): [
        'Arial Bold',
        'Arial-BoldMT',
        'Liberation Sans Bold',
        'Arimo Bold',
        'Nimbus Sans Bold',
        'NimbusSans-Bold',
    ],
    ('normal', 'italic'): [
        'Arial Italic',
        'Arial-ItalicMT',
        'Liberation Sans Italic',
        'Arimo Italic',
        'Nimbus Sans Italic',
        'NimbusSans-Italic',
    ],
    ('bold', 'italic'): [
        'Arial Bold Italic',
        'Arial-BoldItalicMT',
        'Liberation Sans Bold Italic',
        'Arimo Bold Italic',
        'Nimbus Sans Bold Italic',
        'NimbusSans-BoldItalic',
    ],
}

FONTS_JS = r"""
async (faces) => {
  // map Arial and Helvetica to the first installed font with Arial's metrics
  const css = [];
  for (const [weight, style, names] of faces) {
    const src = names.map(n => `local("${n}")`).join(', ');
    for (const family of ['Arial', 'Helvetica']) {
      css.push(`@font-face { font-family: "${family}"; font-weight: ${weight}; ` +
               `font-style: ${style}; src: ${src}; }`);
    }
  }
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  el.textContent = css.join('\n');
  document.documentElement.prepend(el);
  const missing = [];
  for (const [weight, style] of faces) {
    const got = await document.fonts.load(`${style} ${weight} 12px Arial`).catch(() => []);
    if (!got.length) missing.push(`${weight} ${style}`);
  }
  await document.fonts.ready;
  return missing;
}
"""

JS = r"""
() => {
  const ctx = new OffscreenCanvas(1, 1).getContext('2d');
  const out = [];
  for (const sw of document.querySelectorAll('switch')) {
    const fo = sw.querySelector('foreignObject');
    if (!fo) { out.push(null); continue; }
    // client pixels -> user units of the element that holds the switch
    const inv = sw.parentNode.getScreenCTM().inverse();
    const toUser = (x, y) => {
      const p = new DOMPoint(x, y).matrixTransform(inv);
      return [p.x, p.y];
    };
    const chars = [];
    const tw = document.createTreeWalker(fo, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const p = t.parentElement;
      const cs = getComputedStyle(p);
      let sub = false, sup = false;
      for (let n = p; n && n !== fo; n = n.parentElement) {
        const tag = n.tagName.toLowerCase(), va = getComputedStyle(n).verticalAlign;
        if (tag === 'sub' || va === 'sub') sub = true;
        if (tag === 'sup' || va === 'super') sup = true;
      }
      ctx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      const ascent = ctx.measureText('x').fontBoundingBoxAscent;
      const r = document.createRange();
      for (let i = 0; i < t.length; i++) {
        r.setStart(t, i); r.setEnd(t, i + 1);
        const rects = r.getClientRects();
        if (!rects.length || rects[0].width === 0) continue;  // collapsed whitespace
        const b = rects[0];
        const [x, y] = toUser(b.left, b.top + ascent);
        const [x1] = toUser(b.right, b.top);
        chars.push({ch: t.data[i], x, x1, base: y, size: parseFloat(cs.fontSize),
          weight: parseInt(cs.fontWeight), italic: cs.fontStyle === 'italic',
          sub, sup, color: cs.color});
      }
    }
    // the innermost block sets the alignment of the label
    const blocks = fo.querySelectorAll('div');
    const align = blocks.length ? getComputedStyle(blocks[blocks.length - 1]).textAlign : 'center';
    out.push({chars, align});
  }
  return out;
}
"""


def lines(chars):
    """Split measured characters into visual lines and each line into runs of
    one style; a new line starts where a glyph sits left of the previous one."""
    out = []
    for c in chars:
        if not out or c['x'] < out[-1][-1]['x1'] - 0.5:
            out.append([])
        out[-1].append(c)
    result = []
    for ln in out:
        while ln and ln[0]['ch'].isspace():
            ln = ln[1:]
        while ln and ln[-1]['ch'].isspace():
            ln = ln[:-1]
        if not ln:
            continue
        runs = []
        for c in ln:
            key = (round(c['base'], 2), c['size'], c['weight'], c['italic'], c['color'])
            if runs and runs[-1]['key'] == key:
                runs[-1]['text'] += c['ch']
            else:
                runs.append({**c, 'key': key, 'text': c['ch']})
        main = [c for c in ln if not (c['sub'] or c['sup'])] or ln
        result.append(
            {'left': ln[0]['x'], 'right': ln[-1]['x1'], 'base': main[0]['base'], 'runs': runs}
        )
    return result


def style_attrs(r) -> list[str]:
    attrs = [f'font-size="{round(r["size"], 3):g}"']
    if r['weight'] >= 600:
        attrs.append('font-weight="bold"')
    if r['italic']:
        attrs.append('font-style="italic"')
    attrs.append(f'fill={quoteattr(r["color"])}')
    return attrs


def text_element(ln, align: str) -> str:
    """One <text> per line, anchored where the label is aligned, so a renderer
    with other font metrics keeps a centred label centred."""
    if align == 'left':
        x, anchor = ln['left'], 'start'
    elif align == 'right':
        x, anchor = ln['right'], 'end'
    else:
        x, anchor = (ln['left'] + ln['right']) / 2, 'middle'
    head = [f'x="{x:.3f}"', f'y="{ln["base"]:.3f}"', f'font-family="{FONT_FAMILY}"']
    if anchor != 'start':
        head.append(f'text-anchor="{anchor}"')
    runs = ln['runs']
    if len(runs) == 1 and abs(runs[0]['base'] - ln['base']) < 1e-3:
        attrs = head + style_attrs(runs[0]) + ['xml:space="preserve"']
        return f'<text {" ".join(attrs)}>{escape(runs[0]["text"])}</text>'
    body, y = [], ln['base']
    for r in runs:
        attrs = style_attrs(r)
        if abs(r['base'] - y) >= 1e-3:
            attrs.append(f'dy="{r["base"] - y:.3f}"')
            y = r['base']
        body.append(f'<tspan {" ".join(attrs)}>{escape(r["text"])}</tspan>')
    return f'<text {" ".join(head)} xml:space="preserve">{"".join(body)}</text>'


def convert(src: str, labels) -> tuple[str, int, int]:
    switches = list(re.finditer(r'<switch>.*?</switch>', src, re.S))
    if len(switches) != len(labels):
        raise SystemExit(f'{len(switches)} <switch> in the text, {len(labels)} in the DOM')
    parts, pos, n_text, n_empty = [], 0, 0, 0
    for m, label in zip(switches, labels):
        if label is None:
            raise SystemExit('a <switch> without a foreignObject; not a draw.io label')
        parts.append(src[pos : m.start()])
        lns = lines(label['chars'])
        if not lns:
            n_empty += 1
        n_text += len(lns)
        parts.append(''.join(text_element(ln, label['align']) for ln in lns))
        pos = m.end()
    parts.append(src[pos:])
    out = ''.join(parts)
    if '<!-- Labels are native SVG text' not in out:
        i = out.index('<svg')
        out = out[:i] + HEADER + out[i:]
    return out, n_text, n_empty


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('svg')
    ap.add_argument('out', nargs='?')
    args = ap.parse_args()
    path = Path(args.svg).resolve()
    src = path.read_text(encoding='utf-8')

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(path.as_uri())
        page.wait_for_timeout(300)
        faces = [[w, st, names] for (w, st), names in ARIAL_FACES.items()]
        missing = page.evaluate(FONTS_JS, faces)
        if missing:
            browser.close()
            raise SystemExit(
                f'no font with Arial metrics for {", ".join(missing)}; install Arial '
                'or Liberation Sans, or line breaks would differ from draw.io'
            )
        labels = page.evaluate(JS)
        browser.close()

    out, n_text, n_empty = convert(src, labels)
    if not re.search(r'<svg[^>]*\scontent="', out):
        print('warning: no content attribute; the export does not embed the diagram')
    Path(args.out or path).write_text(out, encoding='utf-8')
    print(
        f'{args.out or path}: {len(labels)} labels -> {n_text} lines of <text>, {n_empty} empty dropped'
    )


if __name__ == '__main__':
    main()
