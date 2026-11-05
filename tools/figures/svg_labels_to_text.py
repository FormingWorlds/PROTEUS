"""Replace the HTML labels of a draw.io SVG export with native SVG text.

draw.io writes each label as HTML in a ``foreignObject`` with a raster
fallback. This script lays the labels out in headless Chromium and writes each
line as a ``<text>``, with a ``<tspan>`` per run of italics, bold or subscript.
The rest of the file is kept as it is.

The line breaks are fixed in the output, so the layout uses Arial or a font
with Arial's metrics; with a wider fallback font, lines that fit in draw.io
would wrap.

Usage:
    python3 tools/figures/svg_labels_to_text.py EXPORT.svg [OUT.svg]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from playwright.sync_api import sync_playwright

FONT_FAMILY = 'Arial, Helvetica, sans-serif'
HEADER = '<!-- Labels are native SVG text, so every renderer shows the same wording. -->\n'

# Fonts with Arial's character widths
ARIAL_LIKE = ['Arial', 'Liberation Sans', 'Arimo', 'Nimbus Sans']
FACES = {
    ('normal', 'normal'): '',
    ('bold', 'normal'): ' Bold',
    ('normal', 'italic'): ' Italic',
    ('bold', 'italic'): ' Bold Italic',
}
# Arial's vertical metrics (hhea 1854, 434, 67 of 2048), which Nimbus Sans lacks
ARIAL_VERTICAL = 'ascent-override: 90.53%; descent-override: 21.19%; line-gap-override: 3.27%;'
FONT_CSS = ''
for (weight, style), suffix in FACES.items():
    src = ', '.join(f'local("{f}{suffix}")' for f in ARIAL_LIKE)
    for family in ('Arial', 'Helvetica'):
        FONT_CSS += (
            f'@font-face {{ font-family: "{family}"; font-weight: {weight}; '
            f'font-style: {style}; src: {src}; {ARIAL_VERTICAL} }}\n'
        )

FONTS_JS = r"""
async ([css, faces]) => {
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  el.textContent = css;
  document.documentElement.prepend(el);
  const missing = [];
  for (const [weight, style] of faces) {
    const got = await document.fonts.load(`${style} ${weight} 12px Arial`).catch(() => []);
    if (!got.length) missing.push(`${weight} ${style}`);
  }
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
    const inv = sw.parentNode.getScreenCTM().inverse();
    const toUser = (x, y) => new DOMPoint(x, y).matrixTransform(inv);
    const chars = [];
    const tw = document.createTreeWalker(fo, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const cs = getComputedStyle(t.parentElement);
      let sub = false, sup = false;
      for (let n = t.parentElement; n !== fo; n = n.parentElement) {
        const tag = n.tagName.toLowerCase(), va = getComputedStyle(n).verticalAlign;
        sub ||= tag === 'sub' || va === 'sub';
        sup ||= tag === 'sup' || va === 'super';
      }
      ctx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      const ascent = ctx.measureText('x').fontBoundingBoxAscent;
      const r = document.createRange();
      for (let i = 0; i < t.length; i++) {
        r.setStart(t, i); r.setEnd(t, i + 1);
        const b = r.getClientRects()[0];
        if (!b || b.width === 0) continue;  // collapsed whitespace
        const p = toUser(b.left, b.top + ascent);
        chars.push({ch: t.data[i], x: p.x, x1: toUser(b.right, b.top).x, base: p.y,
          size: parseFloat(cs.fontSize), weight: parseInt(cs.fontWeight),
          italic: cs.fontStyle === 'italic', sub, sup, color: cs.color});
      }
    }
    const blocks = fo.querySelectorAll('div');
    const align = blocks.length ? getComputedStyle(blocks[blocks.length - 1]).textAlign : 'center';
    out.push({chars, align});
  }
  return out;
}
"""


def lines(chars):
    """Group characters into lines (a glyph left of the previous one starts a new
    line) and each line into runs of one style."""
    groups = []
    for c in chars:
        if not groups or c['x'] < groups[-1][-1]['x1'] - 0.5:
            groups.append([])
        groups[-1].append(c)
    out = []
    for ln in groups:
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
        out.append(
            {'left': ln[0]['x'], 'right': ln[-1]['x1'], 'base': main[0]['base'], 'runs': runs}
        )
    return out


def style_attrs(r) -> list[str]:
    attrs = [f'font-size="{round(r["size"], 3):g}"']
    if r['weight'] >= 600:
        attrs.append('font-weight="bold"')
    if r['italic']:
        attrs.append('font-style="italic"')
    return attrs + [f'fill={quoteattr(r["color"])}']


def text_element(ln, align: str) -> str:
    """One line as ``<text>``, anchored at its centre (or edge), so other fonts
    keep it aligned."""
    x, anchor = {
        'left': (ln['left'], 'start'),
        'right': (ln['right'], 'end'),
    }.get(align, ((ln['left'] + ln['right']) / 2, 'middle'))
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


def convert(src: str, labels) -> tuple[str, int]:
    """Replace each ``<switch>`` label in ``src``; returns the SVG and its line count."""
    switches = list(re.finditer(r'<switch>.*?</switch>', src, re.S))
    if len(switches) != len(labels) or None in labels:
        raise SystemExit('the <switch> elements are not all draw.io labels')
    parts, pos, n = [], 0, 0
    for m, label in zip(switches, labels):
        lns = lines(label['chars'])
        n += len(lns)
        parts += [src[pos : m.start()], ''.join(text_element(ln, label['align']) for ln in lns)]
        pos = m.end()
    out = ''.join(parts) + src[pos:]
    if HEADER not in out:
        i = out.index('<svg')
        out = out[:i] + HEADER + out[i:]
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('svg')
    ap.add_argument('out', nargs='?')
    args = ap.parse_args()
    path = Path(args.svg).resolve()
    out_path = Path(args.out or path)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(path.as_uri())
        missing = page.evaluate(FONTS_JS, [FONT_CSS, list(FACES)])
        if missing:
            raise SystemExit(
                f'no font with Arial metrics for {", ".join(missing)}; install Liberation Sans'
            )
        labels = page.evaluate(JS)
        browser.close()

    out, n = convert(path.read_text(encoding='utf-8'), labels)
    if not re.search(r'<svg[^>]*\scontent="', out):
        print('warning: the export does not embed the diagram')
    out_path.write_text(out, encoding='utf-8')
    print(f'{out_path}: {len(labels)} labels -> {n} lines')


if __name__ == '__main__':
    main()
