"""Draw the two schematics of docs/Explanations/redox.md.

Writes, in light and dark (``_dark``) variants, ``docs/assets/redox/redox_domain.svg``
(the mantle as the melt-redox tracker sees it) and ``docs/assets/redox/redox_flow.svg``
(one call of ``update_melt_redox`` in a coupling step), with the proteus-mpl style. All
text is stored as paths, so the files need no web font.

Usage::

    python tools/figures/redox_schematics.py           # write the four files
    python tools/figures/redox_schematics.py --check   # compare with a fresh render

Coordinates are SVG pixels with y pointing down; one pixel is one point. Cell count,
layer depths and the base of the melt are illustrative.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import proteus_mpl  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs' / 'assets' / 'redox'

MELT, MUSH, ROCK, CORE, METAL = '#E23D28', '#B4553F', '#5B3A2E', '#8A96A0', '#2B2F33'
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
        step='#DCE9F3',
        test='#F6E2CC',
        edge='#3E4A55',
    ),
    'dark': dict(
        bg='#05070B',
        ink='#E9EEF2',
        ink2='#9FB0BE',
        mut='#5A6B7A',
        eye='#4FA3D9',
        atm='#1B6FA8',
        atm_a=0.38,
        step='#16283A',
        test='#3A2614',
        edge='#9FB0BE',
    ),
}
LAB, SM, EYE = 13.0, 11.5, 10.5
# Fonts by file from proteus-mpl, so a copy installed elsewhere is never taken.
FONT_DIR = Path(proteus_mpl.__file__).parent / 'fonts'
FONTS = {
    'sans': FONT_DIR / 'InstrumentSans-Regular.ttf',
    'semibold': FONT_DIR / 'InstrumentSans-SemiBold.ttf',
    'mono': FONT_DIR / 'SplineSansMono-Regular.ttf',
}


class Canvas:
    """Axes in SVG pixel coordinates with helpers for text, boxes and arrows."""

    def __init__(self, theme: dict, width: int, height: int):
        self.t = theme
        self.fig = plt.figure(figsize=(width / 72, height / 72), dpi=72)
        self.fig.patch.set_facecolor(theme['bg'])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_axis_off()

    def text(self, x, y, s, size=LAB, color=None, font='sans', ha='left'):
        """Draw one line of text with its baseline at ``y``."""
        self.ax.text(
            x,
            y,
            s,
            fontproperties=FontProperties(fname=FONTS[font], size=size),
            color=color or self.t['ink'],
            ha=ha,
            va='baseline',
        )

    def lines(self, x, y, rows, size=SM, color=None, ha='left', lead=15):
        """Draw text rows from baseline ``y`` down, ``lead`` px apart."""
        for k, row in enumerate(rows):
            self.text(x, y + lead * k, row, size=size, color=color, ha=ha)

    def eyebrow(self, x, y, s, color=None):
        """Upper-case monospaced label."""
        self.text(x, y, s, size=EYE, color=color or self.t['eye'], font='mono')

    def rect(self, x, y, w, h, color, alpha=1.0):
        """Filled rectangle with its top-left corner at (x, y)."""
        self.ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='none', alpha=alpha, lw=0))

    def box(self, x, y, w, h, fill, dashed=False):
        """Rounded box with its top-left corner at (x, y)."""
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle='round,pad=0,rounding_size=7',
                fc=fill,
                ec=self.t['edge'],
                lw=1.0,
                ls=(0, (4, 3)) if dashed else '-',
            )
        )

    def line(self, xs, ys, color=None, lw=1.2, dash=None):
        """Polyline, dashed when ``dash`` is an (on, off) pair."""
        (ln,) = self.ax.plot(
            xs, ys, color=color or self.t['ink2'], lw=lw, solid_capstyle='butt'
        )
        if dash:
            ln.set_dashes(dash)

    def arrow(self, xs, ys, color=None, dash=None):
        """Polyline ending in an arrow head at its last point."""
        color = color or self.t['ink2']
        (x0, y0), (x1, y1) = (xs[-2], ys[-2]), (xs[-1], ys[-1])
        L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        self.line(
            list(xs[:-1]) + [x1 - 7 * ux], list(ys[:-1]) + [y1 - 7 * uy], color, dash=dash
        )
        bx, by = x1 - 9 * ux, y1 - 9 * uy
        self.ax.add_patch(
            Polygon(
                [(bx - 4.5 * uy, by + 4.5 * ux), (bx + 4.5 * uy, by - 4.5 * ux), (x1, y1)],
                fc=color,
                ec='none',
            )
        )


# ── Model domain ─────────────────────────────────────────────────────────────
DOM_W, DOM_H = 920, 540
COL_X, COL_W = 40, 320  # mantle column
SURF, CELL, N_MELT, N_MUSH, N_SOLID = 96, 30, 6, 1, 6
BASE_MELT = SURF + N_MELT * CELL  # bottom of the deepest melt cell
P22 = SURF + 4 * CELL  # illustrative depth of 22 GPa
FE2, FE3 = r'Fe$^{2{+}}$', r'Fe$^{3{+}}$'


def draw_domain(theme_name: str):
    """Build the model-domain figure of one theme and return it."""
    t = THEMES[theme_name]
    c = Canvas(t, DOM_W, DOM_H)
    ink2 = t['ink2']
    right = COL_X + COL_W

    # Atmosphere and surface.
    c.rect(COL_X, 30, COL_W, SURF - 30, t['atm'], t['atm_a'])
    c.eyebrow(COL_X + 12, 50, 'ATMOSPHERE')
    c.text(COL_X + 12, 70, 'CALLIOPE or atmodeller', size=SM, color=ink2)
    c.text(COL_X + 12, 86, r'fO$_2$ = IW + $\Delta$IW at $T_{\rm out}$', size=SM, color=ink2)

    # Cells: melt, one mush cell, solid, core.
    for k in range(N_MELT + N_MUSH + N_SOLID):
        y = SURF + k * CELL
        color = MELT if k < N_MELT else MUSH if k < N_MELT + N_MUSH else ROCK
        c.rect(COL_X, y, COL_W, CELL, color)
        c.line([COL_X, right], [y, y], ON_ROCK, lw=0.5)
    bottom = SURF + (N_MELT + N_MUSH + N_SOLID) * CELL
    c.rect(COL_X, bottom, COL_W, DOM_H - 20 - bottom, CORE)
    c.text(COL_X + 12, bottom + 22, 'core', size=SM, color=ON_MELT)
    c.line([COL_X, right], [SURF, SURF], t['ink'], lw=1.6)

    # Labels inside the column.
    c.text(COL_X + 12, SURF + 20, r'melt cells, $\phi_i > 0$', size=SM, color=ON_MELT)
    c.text(
        COL_X + 12,
        SURF + 20 + CELL,
        f'one {FE3}/{FE2} ratio for all melt',
        size=SM,
        color=ON_MELT,
    )
    c.text(
        COL_X + 12,
        BASE_MELT + 20,
        r'$\phi_i < 0.15$: solid when the metal step is on',
        size=SM - 0.5,
        color=ON_ROCK,
    )
    c.text(
        COL_X + 12,
        BASE_MELT + CELL + 20,
        'solid cells keep the iron they took',
        size=SM,
        color=ON_ROCK,
    )

    # Layer boundaries of the partition coefficients.
    c.line([COL_X, right], [P22, P22], ON_ROCK, lw=1.2, dash=(5, 4))
    c.text(COL_X + 12, P22 - 7, '22 GPa: Cpx + Opx above', size=SM - 1, color=ON_MELT)
    c.text(COL_X + 12, P22 + 18, 'bridgmanite below', size=SM - 1, color=ON_MELT)
    p180 = bottom - CELL
    c.line([COL_X, right], [p180, p180], ON_ROCK, lw=1.2, dash=(5, 4))
    c.text(
        COL_X + 12,
        p180 + 19,
        '180 GPa: post-perovskite below (mgsio3_kd)',
        size=SM - 1,
        color=ON_ROCK,
    )

    # Metal-saturation cell: the base of the melt by default.
    yb = BASE_MELT - CELL / 2
    xd = COL_X + COL_W - 60
    c.ax.plot([xd], [yb], 'o', ms=8, mfc=METAL, mec=ON_ROCK, mew=1.2)
    for dx in (12, 21, 30):
        c.rect(xd + dx, yb - 2, 5, 5, METAL)

    # Right panel.
    x0 = 420
    c.eyebrow(x0, 50, 'GLOBAL MELT RESERVOIRS')
    c.text(x0, 70, rf'$n^{{2{{+}}}}$, $n^{{3{{+}}}}$ [mol]: all {FE2} and {FE3} of the melt')
    c.lines(
        x0,
        88,
        [
            'shared by the melt cells in proportion to their melt mass,',
            rf'so {FE3}/Fe$_{{\rm T}}$ is the same in every melt cell',
        ],
        color=ink2,
    )

    c.eyebrow(x0, 140, 'SURFACE OXYGEN FUGACITY')
    c.text(x0, 160, r'$T_{\rm out} = \max(T_{\rm magma}, T_{\rm floor})$ and 1 bar')
    c.lines(
        x0,
        178,
        [
            r'Hirschmann (2022) Eq. 21 gives fO$_2$; its offset $\Delta$IW from',
            'the Hirschmann (2021) buffer goes to the outgassing; CALLIOPE',
            r'turns it back into fO$_2$ with the same buffer',
        ],
        color=ink2,
    )
    c.arrow([x0 - 10, right + 22, right + 22, right + 6], [155, 155, 66, 66])

    c.eyebrow(x0, 252, 'CRYSTALLISATION AND REMELTING')
    c.lines(
        x0,
        272,
        [
            rf'new solid in cell $i$ takes {FE2} and {FE3} with the $D_i$',
            'of its depth: mgsio3, bulk_BSE or mgsio3_kd;',
            'a cell that remelts returns the iron of that solid',
        ],
        color=ink2,
    )

    c.eyebrow(x0, 340, 'METAL STEP (metal_saturation = true)')
    c.lines(
        x0,
        360,
        [
            r'3FeO = 2FeO$_{1.5}$ + Fe in one melt cell, the base (dot) or the',
            r'cell of largest $a_{\rm Fe}$: a supersaturated melt goes to $a_{\rm Fe}$ = 1,',
            'and the metal (squares) stays in that cell',
        ],
        color=ink2,
    )
    c.arrow([x0 - 10, xd + 44], [356, yb + 2])

    c.eyebrow(x0, 428, 'RADIAL fO2 PROFILE (DIAGNOSTIC)')
    c.lines(
        x0,
        448,
        [
            'Eq. 21 at the T and P of each melt cell; with the metal step',
            r'or fO2_pressure_term it adds $\int\Delta V\,dP$ of the oxidation',
            'of FeO (redox_dV_eos). Stored in the interior snapshot only',
        ],
        color=ink2,
    )
    return c.fig


# ── One call of update_melt_redox ────────────────────────────────────────────
FLOW_W, FLOW_H = 920, 700
BX, BW, BH = 250, 370, 40  # main column of boxes
SX, SW = 670, 220  # right boxes
LX, LW = 20, 190  # left box


def draw_flow(theme_name: str):
    """Build the flow figure of one theme and return it."""
    t = THEMES[theme_name]
    c = Canvas(t, FLOW_W, FLOW_H)
    ink2 = t['ink2']
    cx = BX + BW / 2

    def box(x, y, w, rows, kind='step', h=BH, dashed=False):
        c.box(x, y, w, h, t[kind], dashed)
        first = y + (h - 15 * (len(rows) - 1)) / 2 + 4
        c.lines(x + w / 2, first, rows, size=SM, ha='center', lead=15)

    def label(x, y, s):
        c.text(x, y, s, size=SM - 1, color=ink2, ha='center')

    c.eyebrow(BX, 26, 'ONE COUPLING STEP')
    rows = [
        (
            40,
            [
                'interior solve, Aragog or SPIDER:',
                r'$\phi_i$, $m_i$, $P_i$, $T_i$ of each cell',
            ],
            'step',
        ),
        (110, ['first call?'], 'test'),
        (170, ['global melt fraction below 0.05?'], 'test'),
        (
            230,
            [
                'remelted cells return their iron;',
                rf'new solid takes {FE2} and {FE3} with $D_i$',
            ],
            'step',
        ),
        (290, [r'melt FeO + FeO$_{1.5}$ above the melt mass?'], 'test'),
        (350, [rf'update $n^{{2{{+}}}}$, $n^{{3{{+}}}}$ and {FE3}/Fe$_{{\rm T}}$'], 'step'),
        (410, ['metal_saturation?'], 'test'),
        (480, [r'radial fO$_2$ profile at each melt cell'], 'step'),
        (
            540,
            [
                r'surface fO$_2$ and $\Delta$IW at $T_{\rm out}$ and 1 bar',
                'helpfile column fO2_shift_IW_mantle',
            ],
            'step',
        ),
        (
            600,
            [
                'outgassing: CALLIOPE or atmodeller,',
                r'fO$_2$ = IW + $\Delta$IW (CALLIOPE: Hirschmann 2021)',
            ],
            'step',
        ),
    ]
    for y, text, kind in rows:
        box(BX, y, BW, text, kind, dashed=(y == 480))
    for (y0, _, _), (y1, _, _) in zip(rows[:-1], rows[1:]):
        c.arrow([cx, cx], [y0 + BH, y1])
    for y in (110, 170, 290, 410):
        c.text(cx + 8, y + BH + 15, 'no', size=SM - 1, color=ink2)

    # First call: initialise, then straight to the profile (no metal step).
    box(
        SX,
        104,
        SW,
        [r'initialise $n^{2{+}}$, $n^{3{+}}$ from $f_0$', r'and $D_i$ of each cell'],
        h=52,
    )
    c.arrow([BX + BW, SX], [130, 130])
    label((BX + BW + SX) / 2, 124, 'yes')
    c.arrow([SX + SW, 905, 905, BX + BW], [130, 130, 492, 492])

    # Hold: low melt fraction or too much iron; also straight to the profile.
    box(LX, 164, LW, ['hold the state until', 'the melt returns'], kind='test', h=52)
    c.arrow([BX, LX + LW], [190, 190])
    label((BX + LX + LW) / 2, 184, 'yes')
    c.line([BX, LX + LW / 2], [310, 310])
    label(185, 304, 'yes: hold')
    c.arrow([LX + LW / 2, LX + LW / 2, BX], [216, 500, 500])

    # Partition prescriptions.
    c.lines(SX, 246, ['$D_i$: mgsio3, bulk_BSE', 'or mgsio3_kd'], color=ink2)

    # Metal step.
    box(
        SX,
        398,
        SW,
        [
            'metal step in one melt cell:',
            r'$a_{\rm Fe}$ to 1 at once, or over',
            'the convective transit time',
        ],
        h=64,
    )
    c.arrow([BX + BW, SX], [430, 430])
    label((BX + BW + SX) / 2, 424, 'yes')
    c.arrow([SX + SW / 2, SX + SW / 2, BX + BW], [462, 484, 484])

    # Time step of the next coupling step.
    box(SX, 600, SW, ['next dt capped on the change', r'of $a_{\rm Fe}$ (metal step on)'])
    c.arrow([BX + BW, SX], [620, 620])

    c.lines(
        LX,
        672,
        [
            'warm boxes: tests of the tracker state; cool boxes: steps; dashed: diagnostic, '
            'not used by the outgassing',
        ],
        color=ink2,
    )
    return c.fig


FIGURES = {'redox_domain': draw_domain, 'redox_flow': draw_flow}


def render(name: str, theme_name: str) -> bytes:
    """Return the SVG bytes of one figure and theme; equal inputs give equal bytes."""
    proteus_mpl.use(theme_name)
    with plt.rc_context(
        {
            'svg.hashsalt': name,
            'svg.fonttype': 'path',
            'savefig.bbox': 'standard',
            'lines.scale_dashes': False,
        }
    ):
        fig = FIGURES[name](theme_name)
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', metadata={'Date': None, 'Creator': None})
        plt.close(fig)
    return _compact(buf.getvalue().decode())


def _compact(svg: str) -> bytes:
    """Round coordinates to 0.1 px, shorten glyph ids and drop comments and clip paths."""

    def number(n):
        return f'{float(n.group()):.1f}'.rstrip('0').rstrip('.')

    svg = re.sub(
        r'd="([^"]*)"',
        lambda m: 'd="' + ' '.join(re.sub(r'-?\d+\.\d+', number, m.group(1)).split()) + '"',
        svg,
    )
    svg = re.sub(
        r'(translate\()([-\d.]+) ([-\d.]+)\)',
        lambda m: f'{m.group(1)}{float(m.group(2)):.1f} {float(m.group(3)):.1f})',
        svg,
    )
    ids = {}
    for name in re.findall(r'<path id="([^"]+)" d=', svg):
        ids.setdefault(name, f'g{len(ids)}')
    svg = re.sub(
        r'(id="|href="#)([^"]+)"',
        lambda m: m.group(1) + ids.get(m.group(2), m.group(2)) + '"',
        svg,
    )
    svg = re.sub(r'\s*<!--.*?-->', '', svg, flags=re.S)
    svg = re.sub(r'<g id="(?:text|patch|line2d|axes|figure)_\d+">', '<g>', svg)
    svg = re.sub(
        r'd="([^"]*)" transform="scale\(0\.015625\)"',
        lambda m: (
            'd="'
            + re.sub(r'-?\d+', lambda n: str(round(int(n.group()) / 16)), m.group(1))
            + '" transform="scale(0.25)"'
        ),
        svg,
    )
    svg = re.sub(r' clip-path="url\(#[^)]*\)"', '', svg)
    svg = re.sub(r'\s*<clipPath id="[^"]*">.*?</clipPath>', '', svg, flags=re.S)
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
    svg = re.sub(r'style="fill: (#[0-9a-f]{6})"', r'fill="\1"', svg)
    svg = re.sub(
        r'<g>\s*<g fill="([^"]+)" transform="([^"]+)">\s*(<use href="#[^"]+"[^>]*?)/>\s*</g>\s*</g>',
        r'\3 fill="\1" transform="\2"/>',
        svg,
    )
    return '\n'.join(line.strip() for line in svg.splitlines()).encode()


def main(argv=None) -> int:
    """Write the SVG files, or with ``--check`` compare them with a fresh render."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='compare the files in docs/assets')
    ap.add_argument('--png', type=Path, help='also write PNG previews to this folder')
    args = ap.parse_args(argv)
    for path in FONTS.values():
        if not path.is_file():
            print(f'font file {path} not found: install proteus-mpl, which ships it')
            return 2
    bad = 0
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        for theme, suffix in (('light', ''), ('dark', '_dark')):
            out = ASSETS / f'{name}{suffix}.svg'
            data = render(name, theme)
            if args.check:
                same = out.is_file() and out.read_bytes() == data
                print(f'{out.relative_to(ROOT)}: {"ok" if same else "differs"}')
                bad += not same
            else:
                out.write_bytes(data)
                print(f'wrote {out.relative_to(ROOT)} ({len(data)} bytes)')
            if args.png:
                proteus_mpl.use(theme)
                fig = FIGURES[name](theme)
                fig.savefig(args.png / f'{name}{suffix}.png', dpi=110)
                plt.close(fig)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
