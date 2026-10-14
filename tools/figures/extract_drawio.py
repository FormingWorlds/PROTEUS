"""Extract the diagram model (arch_final.json) from a draw.io SVG export.

The export is rendered in headless Chromium, so labels keep draw.io's line
breaks. draw.io colours are mapped to the model palette through ``PALETTE``,
which ``gen_tikz.py`` maps to dark mode.

Usage:
    python3 tools/figures/extract_drawio.py EXPORT.svg arch_final.json [--logo-dark PNG]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent

# draw.io colour -> model colour.  Unlisted colours pass through unchanged.
PALETTE = {
    '#000000': '#10151B',  # ink, outlines
    '#FFFFFF': '#FDFDFE',  # chip text, paper
    '#3A414A': '#2A343D',  # connectors
    '#3B414A': '#2A343D',
    '#D04B35': '#E23D28',  # interior
    '#553F71': '#593E74',  # orbit
    '#D7A548': '#E0A32E',  # star
    '#366DA3': '#1B6FA8',  # escape, chemistry
    '#94392A': '#A03123',  # outgassing
    '#65A1D4': '#4FA3D9',  # atmosphere climate
    '#A0907D': '#A38F7A',  # accretion
    '#6F7581': '#5A6B7A',  # loop frame, section tags
    '#6F7681': '#5A6B7A',
    '#E4E9ED': '#E3E9EE',  # housekeeping
    '#DFEAEF': 'rgba(198, 225, 238, 0.5)',  # archive
    '#68CC2A': '#2E8B57',  # "Yes" branch
    '#FB6F6A': '#C2362B',  # "No" branch
}

# Label inks that differ from the draw.io export: dark ink on the light star
# chips, and the secondary ink for edge annotations and section titles.
INK_ON_SURFACE = {'#E0A32E': '#10151B'}
SECONDARY_SIZES = (10, 13.3, 15)
SECONDARY_INK = '#3E4A55'

JS = r"""
() => {
  const out = [];
  const num = v => (v === null || v === undefined || v === '') ? null : parseFloat(v);
  function translate(el) {
    let tx = 0, ty = 0;
    for (let n = el; n && n.tagName !== 'svg'; n = n.parentElement) {
      const t = n.getAttribute && n.getAttribute('transform');
      if (t) {
        const m = t.match(/^translate\(\s*([-\d.]+)[ ,]+([-\d.]+)\s*\)$/);
        if (!m) throw new Error('unsupported transform ' + t);
        tx += parseFloat(m[1]); ty += parseFloat(m[2]);
      }
    }
    return [tx, ty];
  }
  function cellOf(el) {
    const g = el.closest('[data-cell-id]');
    return g ? g.getAttribute('data-cell-id') : '';
  }
  function hrefOf(el) {
    const a = el.closest('a');
    return a ? (a.getAttribute('xlink:href') || a.getAttribute('href') || '') : '';
  }
  function paint(el) {
    const cs = getComputedStyle(el);
    return {
      fill: cs.fill === 'none' ? 'none' : cs.fill,
      fill_opacity: parseFloat(el.getAttribute('fill-opacity') ?? cs.fillOpacity ?? 1),
      stroke: cs.stroke === 'none' ? 'none' : cs.stroke,
      stroke_width: parseFloat(el.getAttribute('stroke-width') ?? 1),
      dash: el.getAttribute('stroke-dasharray'),
    };
  }
  const svg = document.querySelector('svg');
  const walker = document.createTreeWalker(svg, NodeFilter.SHOW_ELEMENT);
  for (let el = walker.currentNode; el; el = walker.nextNode()) {
    const tag = el.tagName;
    if (el.parentElement && el.parentElement.closest('foreignObject')) continue;
    if (tag === 'rect' || tag === 'path' || tag === 'ellipse') {
      const [tx, ty] = translate(el);
      const base = {cell: cellOf(el), href: hrefOf(el), tx, ty, ...paint(el)};
      if (tag === 'rect') {
        out.push({kind: 'rect', ...base,
          x: num(el.getAttribute('x')) || 0, y: num(el.getAttribute('y')) || 0,
          w: num(el.getAttribute('width')), h: num(el.getAttribute('height')),
          rx: num(el.getAttribute('rx')) || 0});
      } else if (tag === 'path') {
        out.push({kind: 'path', ...base, d: el.getAttribute('d')});
      } else {
        throw new Error('ellipse not supported');
      }
    } else if (tag === 'image' && !el.closest('switch')) {
      const [tx, ty] = translate(el);
      out.push({kind: 'image', cell: cellOf(el), href: hrefOf(el), tx, ty,
        x: num(el.getAttribute('x')) || 0, y: num(el.getAttribute('y')) || 0,
        w: num(el.getAttribute('width')), h: num(el.getAttribute('height')),
        par: el.getAttribute('preserveAspectRatio') || 'xMidYMid meet',
        data: el.getAttribute('xlink:href')});
    } else if (tag === 'foreignObject') {
      // one label: measure every character, then group into visual lines
      const chars = [];
      const tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      let align = null;
      for (let t = tw.nextNode(); t; t = tw.nextNode()) {
        const p = t.parentElement;
        const cs = getComputedStyle(p);
        align = align || cs.textAlign;
        let italic = false, sub = false, sup = false;
        for (let n = p; n && n !== el; n = n.parentElement) {
          const ncs = getComputedStyle(n);
          if (ncs.fontStyle === 'italic') italic = true;
          if (n.tagName.toLowerCase() === 'sub' || ncs.verticalAlign === 'sub') sub = true;
          if (n.tagName.toLowerCase() === 'sup' || ncs.verticalAlign === 'super') sup = true;
        }
        const r = document.createRange();
        for (let i = 0; i < t.length; i++) {
          r.setStart(t, i); r.setEnd(t, i + 1);
          const rects = r.getClientRects();
          if (!rects.length) continue;  // collapsed whitespace
          const b = rects[0];
          chars.push({ch: t.data[i], left: b.left, right: b.right, top: b.top,
            bottom: b.bottom, size: parseFloat(cs.fontSize), weight: cs.fontWeight,
            style: italic ? 'italic' : 'normal', sub, sup, color: cs.color});
        }
      }
      // innermost block that sets the alignment of the text
      const blocks = el.querySelectorAll('div');
      const inner = blocks[blocks.length - 1];
      out.push({kind: 'text', cell: cellOf(el), href: hrefOf(el), chars,
        align: getComputedStyle(inner).textAlign});
    }
  }
  return {width: num(svg.getAttribute('width')), height: num(svg.getAttribute('height')),
          viewBox: svg.getAttribute('viewBox'), items: out};
}
"""


def hexify(css: str | None) -> str | None:
    if css in (None, 'none', ''):
        return None
    m = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)', css)
    if not m:
        raise ValueError(f'unparsed colour {css!r}')
    r, g, b = (int(m.group(i)) for i in (1, 2, 3))
    h = f'#{r:02X}{g:02X}{b:02X}'
    h = PALETTE.get(h, h)
    if h.startswith('rgba'):
        return h
    if m.group(4) is not None and float(m.group(4)) < 1:
        rr, gg, bb = (int(h[i : i + 2], 16) for i in (1, 3, 5))
        return f'rgba({rr}, {gg}, {bb}, {m.group(4)})'
    return h


def shift_path(d: str, tx: float, ty: float) -> str:
    toks = re.findall(r'[A-Za-z]|[-+]?[\d.]+(?:[eE][-+]?\d+)?', d)
    out, xy = [], 0
    for t in toks:
        if t.isalpha():
            if t not in 'MLCQZ':
                raise ValueError(f'unsupported path command {t} in {d!r}')
            out.append(t)
            xy = 0
        else:
            v = float(t) + (tx if xy % 2 == 0 else ty)
            out.append(f'{v:.3f}'.rstrip('0').rstrip('.'))
            xy += 1
    return ' '.join(out)


def group_lines(chars):
    """Split measured characters into visual lines (a new line starts when a
    glyph begins left of the previous one and lower down)."""
    lines, cur = [], []
    for c in chars:
        if cur and c['left'] < cur[-1]['left'] - 0.5 and c['top'] > cur[-1]['top'] + 1:
            lines.append(cur)
            cur = []
        cur.append(c)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        while ln and ln[0]['ch'].isspace():
            ln = ln[1:]
        while ln and ln[-1]['ch'].isspace():
            ln = ln[:-1]
        if not ln:
            continue
        main = [c for c in ln if not (c['sub'] or c['sup'])]
        size = max(c['size'] for c in main)
        body = [c for c in main if c['size'] == size]
        runs = []
        for c in ln:
            key = (c['style'], c['sub'], c['sup'])
            ch = ' ' if c['ch'] == ' ' else c['ch']
            if runs and runs[-1]['key'] == key:
                runs[-1]['text'] += ch
            else:
                runs.append({'key': key, 'text': ch})
        runs = [
            {
                'text': r['text'],
                'italic': r['key'][0] == 'italic',
                'sub': r['key'][1],
                'sup': r['key'][2],
            }
            for r in runs
        ]
        out.append(
            {
                'top': round(min(c['top'] for c in body), 4),
                'bottom': round(max(c['bottom'] for c in body), 4),
                'left': round(ln[0]['left'], 4),
                'right': round(ln[-1]['right'], 4),
                'text': ''.join(r['text'] for r in runs),
                'runs': runs,
                'size': size,
                'weight': body[0]['weight'],
                'color': hexify(body[0]['color']),
            }
        )
    return out


def surface_fill(items, x, y):
    """Fill of the smallest opaque shape whose bounding box contains (x, y)."""
    best, area = None, float('inf')
    for it in items:
        if (
            it['kind'] not in ('rect', 'path')
            or it['fill'] == 'none'
            or it['fill_opacity'] == 0
        ):
            continue
        if it['kind'] == 'rect':
            x0, y0, x1, y1 = it['x'], it['y'], it['x'] + it['w'], it['y'] + it['h']
        else:
            v = [float(t) for t in re.findall(r'[-\d.]+', it['d'])]
            x0, x1, y0, y1 = min(v[0::2]), max(v[0::2]), min(v[1::2]), max(v[1::2])
        a = (x1 - x0) * (y1 - y0)
        if x0 <= x <= x1 and y0 <= y <= y1 and a < area:
            best, area = it['fill'], a
    return best


def apply_ink_rules(items):
    for it in items:
        if it['kind'] != 'text':
            continue
        for ln in it['mlines']:
            x, y = (ln['left'] + ln['right']) / 2, (ln['top'] + ln['bottom']) / 2
            bg = surface_fill(items, x, y)
            if bg in INK_ON_SURFACE:
                ln['color'] = INK_ON_SURFACE[bg]
            elif ln['size'] in SECONDARY_SIZES and ln['color'] == '#10151B':
                ln['color'] = SECONDARY_INK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('out')
    ap.add_argument('--logo-dark')
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1600, 'height': 1600})
        page.goto(Path(args.svg).resolve().as_uri())
        page.wait_for_timeout(300)
        raw = page.evaluate(JS)
        browser.close()

    items = []
    img_n = 0
    for it in raw['items']:
        k = it['kind']
        if k == 'text':
            mlines = group_lines(it['chars'])
            if not mlines:
                continue
            items.append(
                {
                    'kind': 'text',
                    'cell': it['cell'],
                    'href': it['href'],
                    'align': it['align'] if it['align'] in ('left', 'right') else 'center',
                    'mlines': mlines,
                }
            )
            continue
        tx, ty = it['tx'], it['ty']
        if k == 'image':
            m = re.match(r'data:image/png;base64,(.+)', it['data'] or '')
            if not m:
                raise SystemExit(f'image in {it["cell"]} is not an embedded PNG')
            path = f'img/arch_light_img{img_n:02d}.png'
            (HERE / path).write_bytes(base64.b64decode(m.group(1)))
            img_n += 1
            items.append(
                {
                    'kind': 'image',
                    'cell': it['cell'],
                    'href': it['href'],
                    'x': it['x'] + tx,
                    'y': it['y'] + ty,
                    'w': it['w'],
                    'h': it['h'],
                    'stretch': it['par'] == 'none',
                    'file': path,
                }
            )
            continue
        fill, stroke = hexify(it['fill']), hexify(it['stroke'])
        base = {
            'kind': k,
            'cell': it['cell'],
            'href': it['href'],
            'fill': fill or 'none',
            'fill_opacity': it['fill_opacity'],
            'stroke': stroke or 'none',
            'stroke_width': it['stroke_width'],
            'dash': it['dash'],
        }
        if k == 'rect':
            base.update(x=it['x'] + tx, y=it['y'] + ty, w=it['w'], h=it['h'], rx=it['rx'])
        else:
            base['d'] = shift_path(it['d'], tx, ty)
        items.append(base)

    apply_ink_rules(items)
    if args.logo_dark:
        Path(HERE / 'img/arch_dark_img00.png').write_bytes(Path(args.logo_dark).read_bytes())

    meta = {
        'width': raw['width'],
        'height': raw['height'],
        'viewBox': raw['viewBox'],
        'source': Path(args.svg).name,
    }
    Path(args.out).write_text(
        json.dumps({'meta': meta, 'items': items}, indent=1, ensure_ascii=False) + '\n'
    )
    n = {k: sum(1 for i in items if i['kind'] == k) for k in ('rect', 'path', 'text', 'image')}
    print(f'wrote {args.out}: {n}')


if __name__ == '__main__':
    main()
