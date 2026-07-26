"""Generate a standalone TikZ figure from extracted draw.io SVG primitives.

The generator works in the SVG coordinate system (origin top-left, y down,
1 unit = 1 bp) so every coordinate can be transcribed verbatim.  Colours are
resolved per output mode, which makes the TikZ source the single authority for
the light and dark variants instead of two separately exported files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Colour handling
# --------------------------------------------------------------------------

# Neutral colours flip between modes; domain hues are identical in both, having
# been picked to hold contrast on either background.
DARK_MAP = {
    '#10151B': '#E9EEF2',  # ink -> dark-mode text
    '#3E4A55': '#9FB0BE',  # secondary ink -> secondary dark-mode text
    '#2A343D': '#9FB0BE',  # connector stroke
    '#FDFDFE': '#0E131B',  # paper -> basalt (surfaces only, see CHIP_INK)
    '#E3E9EE': '#12202E',  # sunken paper -> raised basalt
    '#F2F5F7': '#0E131B',
    '#5A6B7A': '#E9EEF2',  # section tag chip
    '#C6E1EE': '#3A120C',
}

# The two label inks.  Which one a label uses is decided by the surface it sits
# on, not by the mode: a chip whose colour is the same in both modes keeps the
# same ink in both.
INK_LIGHT = '#FDFDFE'
INK_DARK = '#10151B'
CHIP_INK = INK_LIGHT
PAGE_BG = {'light': '#FDFDFE', 'dark': '#05070B'}


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hexc: str) -> float:
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def best_ink(bg: str) -> str:
    return INK_LIGHT if contrast(INK_LIGHT, bg) > contrast(INK_DARK, bg) else INK_DARK


def parse_colour(val: str, mode: str) -> tuple[str, float] | None:
    """Return (hex, alpha) or None for 'none'."""
    if val is None:
        return None
    val = val.strip()
    if val in ('none', ''):
        return None
    if val.startswith('light-dark(') and val.endswith(')'):
        inner = val[len('light-dark(') : -1]
        depth, split = 0, None
        for i, ch in enumerate(inner):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                split = i
                break
        if split is None:
            raise ValueError(f'malformed light-dark: {val!r}')
        val = (inner[:split] if mode == 'light' else inner[split + 1 :]).strip()
    m = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$', val)
    if m:
        r, g, b = (int(m.group(i)) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) else 1.0
        return f'#{r:02X}{g:02X}{b:02X}', a
    if val.startswith('#'):
        return val.upper(), 1.0
    raise ValueError(f'unparsed colour {val!r}')


def resolve(val, mode: str):
    """Resolve a source colour into the hex used for this output mode."""
    c = parse_colour(val, 'light')  # always start from the light value
    if c is None:
        return None
    hexc, alpha = c
    if mode == 'dark':
        hexc = DARK_MAP.get(hexc, hexc)
    return hexc, alpha


COLOUR_NAMES: dict[str, str] = {}


def cname(hexc: str) -> str:
    key = hexc.lstrip('#').upper()
    COLOUR_NAMES[key] = key
    return f'c{key}'


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def f(v: float) -> str:
    s = f'{v:.3f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


def path_to_tikz(d: str) -> str:
    """Convert an absolute-command SVG path into a TikZ path body.

    Quadratic segments are promoted to the equivalent cubic, which TikZ draws
    natively and which is an exact transformation, not an approximation.
    """
    toks = re.findall(r'[MLCQZ]|[-+]?[\d.]+(?:[eE][-+]?\d+)?', d)
    out = []
    i = 0
    cmd = None
    cur = None
    first = True
    while i < len(toks):
        t = toks[i]
        if re.match(r'^[A-Z]$', t):
            cmd = t
            i += 1
            if cmd == 'Z':
                out.append('-- cycle')
            continue
        if cmd == 'M':
            x, y = float(toks[i]), float(toks[i + 1])
            out.append(('' if first else ' ') + f'({f(x)},{f(y)})')
            cur = (x, y)
            first = False
            cmd = 'L'  # implicit lineto for subsequent pairs
            i += 2
        elif cmd == 'L':
            x, y = float(toks[i]), float(toks[i + 1])
            out.append(f'-- ({f(x)},{f(y)})')
            cur = (x, y)
            i += 2
        elif cmd == 'Q':
            qx, qy = float(toks[i]), float(toks[i + 1])
            x, y = float(toks[i + 2]), float(toks[i + 3])
            c1 = (cur[0] + 2 / 3 * (qx - cur[0]), cur[1] + 2 / 3 * (qy - cur[1]))
            c2 = (x + 2 / 3 * (qx - x), y + 2 / 3 * (qy - y))
            out.append(
                f'.. controls ({f(c1[0])},{f(c1[1])}) and ({f(c2[0])},{f(c2[1])}) .. ({f(x)},{f(y)})'
            )
            cur = (x, y)
            i += 4
        elif cmd == 'C':
            c1 = (float(toks[i]), float(toks[i + 1]))
            c2 = (float(toks[i + 2]), float(toks[i + 3]))
            x, y = float(toks[i + 4]), float(toks[i + 5])
            out.append(
                f'.. controls ({f(c1[0])},{f(c1[1])}) and ({f(c2[0])},{f(c2[1])}) .. ({f(x)},{f(y)})'
            )
            cur = (x, y)
            i += 6
        else:
            raise ValueError(f'unsupported command {cmd}')
    return ' '.join(out)


def dash_opt(dash: str | None) -> str:
    if not dash:
        return ''
    parts = [p for p in re.split(r'[ ,]+', dash.strip()) if p]
    if len(parts) == 2:
        return f', dash pattern=on {parts[0]}bp off {parts[1]}bp'
    return ''


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------

TEX_ESCAPE = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
}

# Quantities written with a subscript.  The figure labels name hf_row keys, so
# the stem case follows the code (F_xuv, not F_XUV).
SUBSCRIPTED = ['F_xuv', 'F_bol', 'F_tide', 'F_atm', 'T_surf', 'R_planet', 'R_int', 'M_planet']


def tex_escape(s: str) -> str:
    return ''.join(TEX_ESCAPE.get(ch, ch) for ch in s)


def tex_text(s: str, subscripts: bool) -> str:
    """Escape a label, optionally promoting ``X_sub`` to a real subscript."""
    if subscripts:
        pattern = '|'.join(re.escape(q) for q in SUBSCRIPTED)
        parts = re.split(f'({pattern})', s)
        out = []
        for p in parts:
            if p in SUBSCRIPTED:
                stem, sub = p.split('_')
                out.append(rf'\textit{{{stem}}}\textsubscript{{{sub}}}')
            else:
                out.append(tex_escape(p))
        s = ''.join(out)
    else:
        s = tex_escape(s)
    # the only non-ASCII glyph in the figure; the text font carries no Greek, so
    # it is set from the sans math alphabet
    s = s.replace('Φ', r'\ensuremath{\mathsf{\Phi}}')
    return s


def font_cmd(size: float) -> str:
    return rf'\fontsize{{{f(size)}bp}}{{{f(size * 1.2)}bp}}\selectfont'


# Baseline offset below the measured line-box top, as a fraction of the font
# size.  Fitted against the reference render (fit_baseline.py); the browser's
# half-leading is not a fixed fraction of the size, hence the per-size table.
BASELINE_K_DEFAULT = 0.94
BASELINE_K = {10: 1.015, 13: 0.94, 13.3: 0.93, 14: 0.895, 15: 0.945}


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


def path_bbox(d: str):
    pts = re.findall(r'([-\d.]+)[ ,]([-\d.]+)', d)
    if not pts:
        return None
    xs = [float(a) for a, _ in pts]
    ys = [float(b) for _, b in pts]
    return min(xs), min(ys), max(xs), max(ys)


def blend(hexc: str, alpha: float, over: str) -> str:
    a = [int(hexc[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(over[i : i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(alpha * x + (1 - alpha) * y):02X}' for x, y in zip(a, b))


def surfaces(items, mode: str):
    """Filled shapes, largest first, so the last hit is the closest surface."""
    out = []
    for it in items:
        if it.get('fill_opacity', 1) == 0 or it.get('fill') in (None, 'none'):
            continue
        c = resolve(it['fill'], mode)
        if not c:
            continue
        col = c[0] if c[1] >= 1 else blend(c[0], c[1], PAGE_BG[mode])
        if it['kind'] == 'rect':
            out.append((it['x'], it['y'], it['x'] + it['w'], it['y'] + it['h'], col))
        elif it['kind'] == 'path':
            bb = path_bbox(it['d'])
            if bb:
                out.append((*bb, col))
    out.sort(key=lambda s: (s[2] - s[0]) * (s[3] - s[1]), reverse=True)
    return out


def background_at(surfs, x: float, y: float, mode: str) -> str:
    bg = PAGE_BG[mode]
    for x0, y0, x1, y1, col in surfs:
        if x0 <= x <= x1 and y0 <= y <= y1:
            bg = col
    return bg


def label_ink(item, line, surfs, surfs_light, mode: str) -> str:
    """Pick a label's ink from the surface it sits on.

    The light mode uses the model's colour as it stands.  In dark mode the ink
    follows the surface: a chip whose colour is the same in both modes keeps the
    ink it has in light mode, and only a label whose surface actually changes
    has its ink recomputed, by contrast against the new surface.  Choosing by
    contrast alone would flip the ink on a saturated chip where the two inks are
    nearly tied, leaving dark text on an unchanged red or blue chip.
    """
    src = resolve(line['color'], 'light')[0]
    if mode == 'light':
        return src
    if src not in (INK_LIGHT, INK_DARK):
        return resolve(line['color'], 'dark')[0]  # secondary text follows the map
    x = (line['left'] + line['right']) / 2
    y = (line['top'] + line['bottom']) / 2
    bg_light = background_at(surfs_light, x, y, 'light')
    bg_dark = background_at(surfs, x, y, mode)
    if bg_dark == bg_light:
        return src
    return best_ink(bg_dark)


def emit(items, meta, mode: str, *, subscripts: bool, links: bool) -> str:
    body: list[str] = []
    linkboxes: list[dict] = []
    surfs = surfaces(items, mode)
    surfs_light = surfaces(items, 'light')

    for it in items:
        kind = it['kind']
        href = it.get('href') or ''

        if kind == 'rect':
            fill = resolve(it['fill'], mode)
            stroke = resolve(it['stroke'], mode)
            fo = it['fill_opacity']
            if fo == 0:
                fill = None
            if fill is None and stroke is None:
                continue
            opts = []
            if fill:
                opts.append(f'fill={cname(fill[0])}')
                if fill[1] < 1:
                    opts.append(f'fill opacity={f(fill[1])}')
            if stroke:
                opts.append(f'draw={cname(stroke[0])}')
                opts.append(f'line width={f(it["stroke_width"])}bp')
            else:
                opts.append('draw=none')
            r = it['rx']
            if r:
                opts.append(f'rounded corners={f(r)}bp')
            body.append(
                f'\\path[{", ".join(opts)}{dash_opt(it["dash"])}] '
                f'({f(it["x"])},{f(it["y"])}) rectangle '
                f'({f(it["x"] + it["w"])},{f(it["y"] + it["h"])});'
            )
            if href and links:
                linkboxes.append(
                    {
                        'x': it['x'],
                        'y': it['y'],
                        'w': it['w'],
                        'h': it['h'],
                        'href': href,
                        'title': it.get('title'),
                    }
                )

        elif kind == 'path':
            fill = resolve(it['fill'], mode)
            stroke = resolve(it['stroke'], mode)
            if it.get('fill_opacity', 1) == 0:
                fill = None
            opts = []
            if fill:
                opts.append(f'fill={cname(fill[0])}')
                if fill[1] < 1:
                    opts.append(f'fill opacity={f(fill[1])}')
            if stroke:
                opts.append(f'draw={cname(stroke[0])}')
                opts.append(f'line width={f(it["stroke_width"])}bp')
            if not opts:
                continue
            opts.append('line join=miter')
            opts.append('line cap=butt')
            body.append(
                f'\\path[{", ".join(opts)}{dash_opt(it["dash"])}] {path_to_tikz(it["d"])};'
            )
            if href and links:
                xs, ys = zip(
                    *[
                        (float(a), float(b))
                        for a, b in re.findall(
                            r'\(([-\d.]+),([-\d.]+)\)', path_to_tikz(it['d'])
                        )
                    ]
                )
                linkboxes.append(
                    {
                        'x': min(xs),
                        'y': min(ys),
                        'w': max(xs) - min(xs),
                        'h': max(ys) - min(ys),
                        'href': href,
                        'title': it.get('title'),
                    }
                )

        elif kind == 'text':
            # Each line is placed on its own measured baseline, so the figure
            # reproduces the reference line breaking rather than re-deriving it
            # from TeX's paragraph builder.
            for ln in it['mlines']:
                size = ln['size']
                col = (label_ink(it, ln, surfs, surfs_light, mode), 1.0)
                anchor, xpos = {
                    'left': ('base west', ln['left']),
                    'right': ('base east', ln['right']),
                    'center': ('base', (ln['left'] + ln['right']) / 2),
                }[it['align']]
                base = ln['top'] + BASELINE_K.get(size, BASELINE_K_DEFAULT) * size
                opts = [
                    'inner sep=0',
                    'outer sep=0',
                    f'text={cname(col[0])}',
                    f'anchor={anchor}',
                    f'font={font_cmd(size)}',
                ]
                body.append(
                    f'\\node[{", ".join(opts)}] at ({f(xpos)},{f(base)}) '
                    f'{{{tex_text(ln["text"], subscripts)}}};'
                )

        elif kind == 'image' and it.get('file'):
            # the raster logo has a light-on-dark counterpart
            path = it['file']
            alt = path.replace('arch_light_img', 'arch_dark_img')
            if mode == 'dark' and (HERE / alt).exists():
                path = alt
            # SVG images default to preserveAspectRatio="xMidYMid meet": scale to
            # fit inside the box and centre the remainder
            iw, ih = Image.open(HERE / path).size
            scale = min(it['w'] / iw, it['h'] / ih)
            dw, dh = iw * scale, ih * scale
            body.append(
                f'\\node[inner sep=0, outer sep=0, anchor=north west] at '
                f'({f(it["x"] + (it["w"] - dw) / 2)},{f(it["y"] + (it["h"] - dh) / 2)}) '
                f'{{\\includegraphics[width={f(dw)}bp,height={f(dh)}bp]{{{path}}}}};'
            )

    return body, linkboxes


PREAMBLE = r"""\documentclass[tightpage]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage{sansmath}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage[hidelinks]{hyperref}
\usetikzlibrary{calc}
\sansmath
"""


def build(src: Path, out: Path, mode: str, *, subscripts: bool, links: bool):
    data = json.loads(src.read_text())
    items = data['items']
    meta = data['meta']
    missing = [it['cell'] for it in items if it['kind'] == 'text' and 'mlines' not in it]
    if missing:
        raise SystemExit(f'labels without line geometry: {missing}')
    body, linkboxes = emit(items, meta, mode, subscripts=subscripts, links=links)

    colours = '\n'.join(rf'\definecolor{{c{k}}}{{HTML}}{{{k}}}' for k in sorted(COLOUR_NAMES))
    W, H = meta['width'], meta['height']
    bg = '#FDFDFE' if mode == 'light' else '#05070B'
    lines = [
        PREAMBLE,
        colours,
        rf'\definecolor{{pagebg}}{{HTML}}{{{bg.lstrip("#")}}}',
        r'\begin{document}',
        r'\begin{tikzpicture}[x=1bp, y=-1bp, every node/.style={inner sep=0, outer sep=0}]',
        rf'\useasboundingbox (0,0) rectangle ({f(W)},{f(H)});',
        *body,
    ]
    if links:
        for lb in linkboxes:
            lines.append(
                f'\\node[anchor=north west] at ({f(lb["x"])},{f(lb["y"])}) '
                f'{{\\href{{{lb["href"]}}}{{\\phantom{{\\rule{{{f(lb["w"])}bp}}'
                f'{{{f(lb["h"])}bp}}}}}}}};'
            )
    lines += [r'\end{tikzpicture}', r'\end{document}']
    out.write_text('\n'.join(lines) + '\n')
    Path(out.with_suffix('.links.json')).write_text(json.dumps(linkboxes, indent=1))
    print(f'wrote {out} ({len(body)} primitives, {len(linkboxes)} links, mode={mode})')


if __name__ == '__main__':
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else 'light'
    subs = '--no-subscripts' not in sys.argv
    links = '--no-links' not in sys.argv
    build(src, out, mode, subscripts=subs, links=links)
