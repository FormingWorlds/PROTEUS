"""Add clickable regions to the SVG exported from the TikZ figure.

The PDF-to-SVG converter drops link annotations, so the clickable areas are
re-attached here as an overlay of transparent rectangles, one per linked shape,
in the figure's own coordinate system.  The overlay is appended last so it sits
above the artwork and receives the clicks.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path


def main(svg_path, links_path, out_path, fig_w=1412.0, fig_h=1415.0):
    svg = Path(svg_path).read_text()
    links = json.loads(Path(links_path).read_text())

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not m:
        raise SystemExit('no viewBox in the exported SVG')
    vw, vh = float(m.group(1)), float(m.group(2))
    sx, sy = vw / fig_w, vh / fig_h

    seen = set()
    rows = []
    for lb in links:
        key = (round(lb['x'], 2), round(lb['y'], 2), lb['href'])
        if key in seen:
            continue
        seen.add(key)
        title = lb['href'].split('/blob/main/')[-1].split('/tree/main/')[-1]
        rows.append(
            f'<a xlink:href="{escape(lb["href"], quote=True)}" target="_blank">'
            f'<rect x="{lb["x"] * sx:.3f}" y="{lb["y"] * sy:.3f}" '
            f'width="{lb["w"] * sx:.3f}" height="{lb["h"] * sy:.3f}" '
            f'fill="none" pointer-events="all">'
            f'<title>{escape(title)}</title></rect></a>'
        )

    overlay = '<g id="links">' + ''.join(rows) + '</g>\n'
    if '</svg>' not in svg:
        raise SystemExit('malformed SVG')
    svg = svg.replace('</svg>', overlay + '</svg>')

    # An SVG loaded as its own document is painted on the user agent's default
    # canvas, which would box the figure in white on a dark page.  Declaring the
    # background transparent and naming the scheme the colours were built for
    # lets the page show through in both schemes.
    scheme = 'dark' if 'dark' in Path(out_path).stem else 'light'
    style = f'background: transparent; background-color: transparent; color-scheme: {scheme};'
    if 'background: transparent' not in svg:
        svg = svg.replace('<svg ', f'<svg style="{style}" ', 1)
    Path(out_path).write_text(svg)
    print(f'{out_path}: {len(rows)} clickable regions')


if __name__ == '__main__':
    main(*sys.argv[1:])
