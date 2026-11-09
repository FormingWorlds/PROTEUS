"""Draw the lid coupling schematic of docs/Explanations/lid_coupling.md.

Writes ``docs/assets/lid_coupling_schematic.svg`` (light) and
``docs/assets/lid_coupling_schematic_dark.svg`` (dark) with the proteus-mpl
style. All text is stored as paths, so the files need no web font.

Usage::

    python tools/figures/lid_coupling_schematic.py           # write both files
    python tools/figures/lid_coupling_schematic.py --check   # compare with a fresh render

Coordinates are in SVG pixels of a 920 x 472 canvas with y pointing down; one
pixel is one point. The temperature profile of the lid uses the erf family of
the Aragog lid with s = 1, for illustration only.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from math import erf
from pathlib import Path

import matplotlib
import matplotlib.colors

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import proteus_mpl  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs' / 'assets'
W, H = 920, 472

# Depth geometry: surface, skin base, lid base, bottom of the drawn melt.
S, SK, LB, MB = 130, 140, 220, 420
# Temperature axis positions of T_surf, T_b and T_node (pixels, not kelvin).
TS, TB, TN = 690, 820, 860

MELT, SKIN, LID_TOP, LID_BASE = '#E23D28', '#8E1F12', '#3A120C', '#8E1F12'
ON_ROCK, ON_MELT = '#F2F5F7', '#10151B'
THEMES = {
    'light': dict(
        bg='#F2F5F7',
        ink='#10151B',
        ink2='#3E4A55',
        mut='#7A8894',
        eye='#1B6FA8',
        atm='#4FA3D9',
        atm_a=0.22,
        grid='#D2DAE1',
    ),
    'dark': dict(
        bg='#05070B',
        ink='#E9EEF2',
        ink2='#9FB0BE',
        mut='#5A6B7A',
        eye='#4FA3D9',
        atm='#1B6FA8',
        atm_a=0.38,
        grid='#1A2230',
    ),
}
LAB, SM, EYE = 13.0, 11.5, 10.5
MONO = 'Spline Sans Mono'


def _hex_mix(a: str, b: str, t: float) -> str:
    """Return the colour at fraction ``t`` between hex colours ``a`` and ``b``."""
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x + (y - x) * t):02X}' for x, y in zip(ca, cb))


class Canvas:
    """Axes in SVG pixel coordinates with helpers for text runs and arrows."""

    def __init__(self, theme: dict):
        self.t = theme
        self.fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
        self.fig.patch.set_facecolor(theme['bg'])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_axis_off()
        self.renderer = self.fig.canvas.get_renderer()

    def text(self, x, y, s, size=LAB, color=None, family=None, weight='normal', ha='left'):
        """Draw one text at baseline ``y``; return the x of its right edge."""
        art = self.ax.text(
            x,
            y,
            s,
            fontsize=size,
            color=color or self.t['ink'],
            family=family or 'sans-serif',
            weight=weight,
            ha=ha,
            va='baseline',
        )
        bb = art.get_window_extent(self.renderer)
        return x + bb.width if ha == 'left' else x

    def run(self, x, y, parts, size=LAB, color=None):
        """Draw text pieces one after another; a piece is (text,) or (text, family)."""
        for p in parts:
            x = self.text(
                x, y, p[0], size=size, color=color, family=p[1] if len(p) > 1 else None
            )
        return x

    def tracked(self, x, y, s, color, size=EYE, spacing=0.16, rotate=False, ha='left'):
        """Draw upper-case monospaced text with letter spacing, one glyph at a time."""
        prop = FontProperties(family=MONO, size=size)
        step = TextPath((0, 0), 'M' * 10, prop=prop).get_extents().width / 10 + spacing * size
        start = -(step * len(s) - spacing * size) / 2 if ha == 'center' else 0.0
        for i, ch in enumerate(s):
            if ch == ' ':
                continue
            d = start + i * step
            gx, gy = (x, y + d) if rotate else (x + d, y)
            self.ax.text(
                gx,
                gy,
                ch,
                fontproperties=prop,
                color=color,
                rotation=-90 if rotate else 0,
                rotation_mode='anchor',
                va='baseline',
            )

    def up(self, x, y1, y2, color):
        """Draw an upward arrow from ``y1`` with its tip at ``y2``."""
        self.ax.plot([x, x], [y1, y2 + 7], color=color, lw=2, solid_capstyle='butt')
        self.ax.add_patch(
            Polygon([(x - 5, y2 + 9), (x + 5, y2 + 9), (x, y2)], fc=color, ec='none')
        )

    def rect(self, x, y, w, h, color, alpha=1.0):
        """Draw a filled rectangle with its top-left corner at (x, y)."""
        self.ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='none', alpha=alpha, lw=0))

    def line(self, xs, ys, color, lw=1.0, dash=None, alpha=1.0, cap='butt'):
        """Draw a polyline, dashed when ``dash`` is an (on, off) pair."""
        (ln,) = self.ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle=cap)
        if dash:
            ln.set_dashes(dash)
            ln.set_dash_capstyle(cap)


def draw(theme_name: str):
    """Build the figure of one theme and return it."""
    t = THEMES[theme_name]
    c = Canvas(t)
    ink, ink2 = t['ink'], t['ink2']

    # Atmosphere.
    c.rect(40, 40, 520, S - 40, t['atm'], t['atm_a'])
    c.tracked(52, 60, 'ATMOSPHERE · AGNI OR DUMMY', t['eye'])
    c.up(330, 128, 48, ink)
    c.text(342, 72, r'$F_\mathrm{atm}(T_\mathrm{surf})$')
    c.text(342, 90, 'flux of the atmosphere call,', size=SM, color=ink2)
    c.text(342, 105, 'one balance for both parts', size=SM, color=ink2)
    c.tracked(52, 123, 'LID PART', ink2)
    c.tracked(342, 123, 'OPEN PART', ink2)

    # Melt, lid (vertical gradient in 2 px bands) and skin.
    c.rect(40, LB, 290, MB - LB, MELT)
    c.rect(330, SK, 230, MB - SK, MELT)
    rgb = [[matplotlib.colors.to_rgb(_hex_mix(LID_TOP, LID_BASE, i / 63))] for i in range(64)]
    c.ax.imshow(rgb, extent=(40, 330, LB, S), aspect='auto', interpolation='none')
    c.ax.set_xlim(0, W)
    c.ax.set_ylim(H, 0)
    c.rect(330, S, 230, SK - S, SKIN)
    c.line([40, 330], [LB, LB], ON_ROCK, lw=1.2, dash=(5, 4))
    c.line([330, 330], [S, MB], ON_MELT, lw=1.0, dash=(3, 4), alpha=0.35)
    c.line([40, 560], [S, S], ink, lw=1.6)
    c.text(552, 123, r'$T_\mathrm{surf}$', ha='right')

    # Lid part labels.
    c.text(54, 154, 'solid lid', color=ON_ROCK, weight='semibold')
    c.text(54, 172, r'thickness $\delta$, centimetres to decimetres', size=SM, color=ON_ROCK)
    x = c.text(
        54, 190, r"$G_\mathrm{lid} = k_\mathrm{lid}\,f'(0)/\delta$", size=SM, color=ON_ROCK
    )
    c.text(x + 10, 190, 'lid_conductance', size=SM - 1, color=ON_ROCK, family=MONO)
    c.text(
        54,
        208,
        r'$\delta$ and $E_\mathrm{lid}$ are Aragog solver states',
        size=SM,
        color=ON_ROCK,
    )
    c.up(306, 216, 134, ON_ROCK)
    c.text(298, 160, '$C$', color=ON_ROCK, ha='right')
    x = c.text(54, 238, r'$T_\mathrm{b}$ at the solidus', color=ON_MELT)
    c.text(x + 10, 238, 'lid_T_base', size=EYE, color=ON_MELT, family=MONO)
    c.up(306, 300, 224, ON_ROCK)
    c.text(298, 270, r'$F_\mathrm{melt}$', color=ON_MELT, ha='right')
    c.text(
        54,
        276,
        r'Rule 2: $F_\mathrm{melt} = F_\mathrm{surf} + \mathrm{d}E_\mathrm{lid}/\mathrm{d}t$',
        size=SM,
        color=ON_MELT,
    )

    # Open part labels; the dot is the top basic node directly under the skin.
    c.ax.plot([346], [SK], 'o', ms=8, mfc=ON_MELT, mec=ON_ROCK, mew=1.2)
    x = c.text(358, 160, r'$T_\mathrm{node}$', color=ON_MELT)
    c.text(x + 10, 160, 'lid_T_node', size=EYE, color=ON_MELT, family=MONO)
    c.text(358, 176, 'top basic node of the mesh', size=SM, color=ON_MELT)
    c.text(358, 200, 'skin above it: $d$ = 0.01 m', size=SM, color=ON_MELT)
    c.text(
        358,
        216,
        r'$G_\mathrm{skin} = k/d$ = 200 W m$^{-2}$ K$^{-1}$',
        size=SM,
        color=ON_MELT,
    )
    c.up(540, 330, 144, ON_ROCK)
    c.text(530, 300, r'$F_\mathrm{open}$', color=ON_MELT, ha='right')
    c.text(530, 316, 'lid_F_open', size=EYE, color=ON_MELT, family=MONO, ha='right')

    # Melt eyebrow and the break line of the cut column.
    c.tracked(54, MB - 40, 'MELT · ARAGOG TOP CELL', ON_MELT)
    xs = [40 + 8 * i for i in range(66)]
    ys = [MB - 24 - (5 if i % 2 else 0) for i in range(66)]
    c.line(xs, ys, ON_ROCK, lw=1.4)

    # Top cell bracket.
    mut = t['mut']
    c.line([580, 580], [S, MB], mut)
    c.line([575, 585], [S, S], mut)
    c.line([575, 585], [MB, MB], mut)
    c.tracked(
        590,
        (S + MB) / 2,
        'TOP CELL · 65.8 KM · NOT TO SCALE',
        ink2,
        spacing=0.12,
        rotate=True,
        ha='center',
    )

    # Weight brackets.
    c.line([42, 42, 328, 328], [MB + 12, MB + 20, MB + 20, MB + 12], ink2)
    c.text(185, MB + 38, '$w$', ha='center')
    c.line([332, 332, 558, 558], [MB + 12, MB + 20, MB + 20, MB + 12], ink2)
    c.text(445, MB + 38, '$1 - w$', ha='center')

    # Temperature with depth, on the depth axis of the column.
    c.tracked(660, 60, 'TEMPERATURE WITH DEPTH', t['eye'])
    c.text(660, 78, 'same depth axis as the column', size=SM, color=ink2)
    c.line([660, 884], [S, S], ink, lw=1.2)
    c.ax.add_patch(Polygon([(884, S - 4), (884, S + 4), (892, S)], fc=ink, ec='none'))
    c.text(898, S + 4, '$T$')
    c.line([660, 660], [S, MB - 6], ink, lw=1.2)
    c.ax.add_patch(Polygon([(656, MB - 6), (664, MB - 6), (660, MB + 2)], fc=ink, ec='none'))
    c.text(660, MB + 18, '$z$', ha='center')
    for xt, lab in (
        (TS, r'$T_\mathrm{surf}$'),
        (TB, r'$T_\mathrm{b}$'),
        (TN, r'$T_\mathrm{node}$'),
    ):
        c.line([xt, xt], [S - 4, S + 4], ink)
        c.text(xt, S - 10, lab, size=SM, ha='center')
    c.line([660, 890], [SK, SK], t['grid'], dash=(3, 3))
    c.line([660, 890], [LB, LB], t['grid'], dash=(3, 3))
    c.text(652, SK + 4, '$d$', ha='right')
    c.text(652, LB + 4, r'$\delta$', ha='right')
    zs = [i / 40 for i in range(41)]
    c.line(
        [TS + (TB - TS) * erf(z) / erf(1.0) for z in zs],
        [S + (LB - S) * z for z in zs],
        ink,
        lw=2,
    )
    c.line([TB, TB + 6], [LB, MB - 10], ink, lw=1.4, dash=(1.5, 4), cap='round')
    c.line([TS, TN], [S, SK], ink, lw=2, dash=(6, 4))
    c.line([TN, TN + 6], [SK, MB - 10], ink, lw=1.4, dash=(1.5, 4), cap='round')
    c.text(770, 160, 'open part: skin', size=SM)
    c.text(700, 206, 'lid part', size=SM)
    c.text(672, 290, 'dotted: melt below', size=SM)
    c.text(672, 306, 'each part conducts', size=SM)
    c.text(672, 322, 'from its own warm side,', size=SM)
    c.text(672, 338, r'$T_\mathrm{b}$ or $T_\mathrm{node} \geq T_\mathrm{b}$', size=SM)
    return c.fig


def render(theme_name: str) -> bytes:
    """Return the SVG bytes of one theme; equal inputs give equal bytes."""
    proteus_mpl.use(theme_name)
    with plt.rc_context(
        {
            'svg.hashsalt': 'lid_coupling_schematic',
            'svg.fonttype': 'path',
            'savefig.bbox': 'standard',
        }
    ):
        fig = draw(theme_name)
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', metadata={'Date': None})
        plt.close(fig)
    return _compact(buf.getvalue().decode())


def _compact(svg: str) -> bytes:
    """Round path coordinates to 0.1 px and join path data onto one line."""

    def path_data(m):
        d = re.sub(
            r'-?\d+\.\d+',
            lambda n: f'{float(n.group()):.1f}'.rstrip('0').rstrip('.'),
            m.group(1),
        )
        return 'd="' + ' '.join(d.split()) + '"'

    svg = re.sub(r'd="([^"]*)"', path_data, svg)
    svg = re.sub(
        r'translate\(([-\d.]+) ([-\d.]+)\)',
        lambda m: f'translate({float(m.group(1)):.2f} {float(m.group(2)):.2f})',
        svg,
    )
    # Short glyph ids in order of first use; no comments, no indentation.
    ids = {}
    for name in re.findall(r'<path id="([^"]+)" d=', svg):
        ids.setdefault(name, f'g{len(ids)}')
    svg = re.sub(
        r'(id="|href="#)([^"]+)"',
        lambda m: m.group(1) + ids.get(m.group(2), m.group(2)) + '"',
        svg,
    )
    svg = re.sub(r'\s*<!--.*?-->', '', svg, flags=re.S)
    # Let the viewer smooth the 1 x 64 lid gradient image.
    svg = svg.replace(' style="image-rendering:crisp-edges;image-rendering:pixelated"', '')
    svg = re.sub(r'<g id="(?:text|patch|line2d|axes|figure)_\d+">', '<g>', svg)

    # Glyph outlines at 1/16 of their resolution (under 0.02 px at these sizes).
    def glyph(m):
        d = re.sub(r'-?\d+', lambda n: str(round(int(n.group()) / 16)), m.group(1))
        return f'd="{d}" transform="scale(0.25)"'

    svg = re.sub(r'd="([^"]*)" transform="scale\(0\.015625\)"', glyph, svg)
    # Drop uses of glyphs with no outline (spaces have no definition).
    defined = set(re.findall(r'<path id="([^"]+)"', svg))
    svg = re.sub(
        r'\s*<use xlink:href="#([^"]+)"[^>]*/>',
        lambda m: m.group() if m.group(1) in defined else '',
        svg,
    )

    def offset(m):
        x, y = (f'{float(v):.1f}'.rstrip('0').rstrip('.') for v in m.groups())
        return f' x="{x}"' + ('' if y in ('0', '-0') else f' y="{y}"')

    svg = re.sub(r' transform="translate\(([-\d.]+) ([-\d.]+)\)"(?=/>)', offset, svg)
    svg = svg.replace('xlink:href="#', 'href="#')
    svg = re.sub(r'd="[^"]*"', lambda m: re.sub(r' ?([A-Za-z]) ', r'\1', m.group()), svg)
    svg = re.sub(r'\.00\b', '', svg)
    # Fill as an attribute; no clip paths (the axes cover the whole canvas).
    svg = re.sub(r'style="fill: (#[0-9a-f]{6})"', r'fill="\1"', svg)
    svg = re.sub(r' clip-path="url\(#[^)]*\)"', '', svg)
    svg = re.sub(r'\s*<clipPath id="[^"]*">.*?</clipPath>', '', svg, flags=re.S)
    # A text of one glyph: move its group fill and transform onto the use.
    svg = re.sub(
        r'<g>\s*<g fill="([^"]+)" transform="([^"]+)">\s*(<use href="#[^"]+"[^>]*?)/>\s*</g>\s*</g>',
        r'\3 fill="\1" transform="\2"/>',
        svg,
    )
    return '\n'.join(line.strip() for line in svg.splitlines()).encode()


def main(argv=None) -> int:
    """Write both SVG files, or with ``--check`` compare them with a fresh render."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='compare the files in docs/assets')
    args = ap.parse_args(argv)
    bad = 0
    for name, suffix in (('light', ''), ('dark', '_dark')):
        out = ASSETS / f'lid_coupling_schematic{suffix}.svg'
        data = render(name)
        if args.check:
            same = out.is_file() and out.read_bytes() == data
            print(f'{out.relative_to(ROOT)}: {"ok" if same else "differs"}')
            bad += not same
        else:
            out.write_bytes(data)
            print(f'wrote {out.relative_to(ROOT)} ({len(data)} bytes)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
