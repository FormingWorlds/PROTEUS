#!/usr/bin/env python
"""Generate an interactive flowchart of the PROTEUS framework architecture.

Produces an SVG (with clickable links to GitHub source) and a self-contained
HTML wrapper.  Requires the ``graphviz`` Python package **and** the Graphviz
system binary (``dot``).

Usage
-----
    python tools/generate_flowchart.py                        # defaults
    python tools/generate_flowchart.py --output-dir docs/assets
    python tools/generate_flowchart.py --format pdf           # pdf/png/svg

Installation (if not already available)
---------------------------------------
    conda install graphviz python-graphviz   # or
    brew install graphviz && pip install graphviz
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import graphviz  # noqa: F401
except ImportError:
    sys.exit(
        'ERROR: The "graphviz" Python package is required.\n'
        'Install it with:\n'
        '  conda install graphviz python-graphviz\n'
        'or:\n'
        '  brew install graphviz && pip install graphviz'
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GITHUB_BASE = 'https://github.com/FormingWorlds/PROTEUS/blob/main/'
SRC = 'src/proteus/'

# Colours (accessible palette)
C_INIT = '#4E79A7'  # blue-grey  – initialisation
C_INTERIOR = '#E15759'  # red        – interior
C_ORBIT = '#F28E2B'  # orange     – orbit / tides
C_STAR = '#EDC948'  # yellow     – stellar
C_ESCAPE = '#76B7B2'  # teal       – escape
C_OUTGAS = '#59A14F'  # green      – outgassing
C_ATMOS = '#AF7AA1'  # purple     – atmosphere climate
C_CHEM = '#FF9DA7'  # pink       – atmosphere chemistry
C_OBS = '#9C755F'  # brown      – observations
C_HOUSE = '#BAB0AC'  # grey       – housekeeping
C_DECISION = '#FFFFFF'  # white      – decision nodes
C_BACKEND = '#F0F0F0'  # light grey – backend boxes

FONT = 'Helvetica'


def _url(rel_path: str, github_base: str = GITHUB_BASE) -> str:
    """Return a full GitHub URL for *rel_path* (relative to repo root)."""
    return github_base + rel_path


def _src_url(mod_path: str, github_base: str = GITHUB_BASE) -> str:
    """Shorthand for source files under ``src/proteus/``."""
    return _url(SRC + mod_path, github_base)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


DEFAULT_LOGO = Path('docs/assets/PROTEUS_white.png')


def build_graph(
    github_base: str = GITHUB_BASE,
    logo_path: Path | None = DEFAULT_LOGO,
) -> graphviz.Digraph:
    """Build the full PROTEUS architecture flowchart."""

    def src_url(mod_path: str) -> str:
        return _src_url(mod_path, github_base)

    g = graphviz.Digraph(
        'PROTEUS',
        format='svg',
        engine='dot',
        graph_attr={
            'rankdir': 'TB',
            'fontname': FONT,
            'fontsize': '14',
            'label': 'Coupling loop and module architecture',
            'labelloc': 't',
            'labeljust': 'c',
            'pad': '0.5',
            'nodesep': '0.6',
            'ranksep': '0.7',
            'bgcolor': 'white',
            'style': 'filled',
            'fillcolor': 'white',
        },
        node_attr={
            'fontname': FONT,
            'fontsize': '11',
            'style': 'filled,rounded',
            'shape': 'box',
            'penwidth': '1.5',
        },
        edge_attr={
            'fontname': FONT,
            'fontsize': '9',
            'color': '#666666',
            'arrowsize': '0.8',
        },
    )

    # ── Logo ──────────────────────────────────────────────────────────
    if logo_path and logo_path.exists():
        g.node(
            'logo',
            '',
            shape='none',
            image=str(logo_path.resolve()),
            imagescale='true',
            width='3.5',
            height='1.0',
            fixedsize='true',
            URL='https://proteus-framework.org',
            target='_blank',
            tooltip='PROTEUS framework',
        )
        g.edge('logo', 'init', style='invis')

    # ── Initialisation ────────────────────────────────────────────────
    g.node(
        'init',
        'Initialisation\n(config, dirs, data)',
        fillcolor=C_INIT,
        fontcolor='white',
        URL=src_url('proteus.py'),
        target='_blank',
        tooltip='Proteus.__init__ + start(): config parsing, directory setup, data download',
    )
    g.node(
        'init_star',
        'Init Star',
        fillcolor=C_STAR,
        URL=src_url('star/wrapper.py'),
        target='_blank',
        tooltip='init_star(): prepare stellar model & modern spectrum',
    )
    g.node(
        'init_orbit',
        'Init Orbit',
        fillcolor=C_ORBIT,
        URL=src_url('orbit/wrapper.py'),
        target='_blank',
        tooltip='init_orbit(): prepare orbital parameters',
    )

    g.edge('init', 'init_star', label='config, dirs')
    g.edge('init_star', 'init_orbit')
    g.edge('init_orbit', 'loop_start')

    # ── Main loop entry ───────────────────────────────────────────────
    g.node(
        'loop_start',
        'COUPLING LOOP\n(while not converged)',
        shape='diamond',
        fillcolor=C_DECISION,
        penwidth='2',
        width='2.8',
        height='0.9',
        URL=src_url('proteus.py'),
        target='_blank',
        tooltip='Proteus.start(): main while-loop (line ~347)',
    )

    # ── Interior ──────────────────────────────────────────────────────
    with g.subgraph(name='cluster_interior') as s:
        s.attr(
            label='Interior Evolution',
            style='dashed,rounded',
            color=C_INTERIOR,
            fontcolor=C_INTERIOR,
            penwidth='1.5',
        )
        s.node(
            'run_interior',
            'run_interior()',
            fillcolor=C_INTERIOR,
            fontcolor='white',
            URL=src_url('interior/wrapper.py'),
            target='_blank',
            tooltip='Run interior mantle evolution model',
        )
        # Backends
        for backend, path, tip in [
            ('SPIDER\n(C)', 'interior/spider.py', 'Thermal evolution – entropy formalism'),
            (
                'Aragog\n(Python)',
                'interior/aragog.py',
                'Thermal evolution – temperature formalism',
            ),
            ('dummy', 'interior/dummy.py', 'Dummy interior for testing'),
        ]:
            nid = 'int_' + backend.split('\n')[0].lower()
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('run_interior', nid, style='dashed', arrowsize='0.6')

    # ── Structure ─────────────────────────────────────────────────────
    with g.subgraph(name='cluster_struct') as s:
        s.attr(
            label='Planetary Structure',
            style='dashed,rounded',
            color=C_INTERIOR,
            fontcolor=C_INTERIOR,
            penwidth='1.5',
        )
        s.node(
            'solve_structure',
            'solve_structure()',
            fillcolor=C_INTERIOR,
            fontcolor='white',
            URL=src_url('interior/wrapper.py'),
            target='_blank',
            tooltip='Determine mass/radius relationship',
        )
        for backend, path, tip in [
            ('self\n(built-in)', 'interior/wrapper.py', 'Simple M-R scaling'),
            ('Zalmoxis\n(Python)', 'interior/zalmoxis.py', 'Detailed interior structure'),
        ]:
            nid = 'struct_' + backend.split('\n')[0].lower()
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('solve_structure', nid, style='dashed', arrowsize='0.6')

    # ── Orbit & Tides ─────────────────────────────────────────────────
    with g.subgraph(name='cluster_orbit') as s:
        s.attr(
            label='Orbit & Tides',
            style='dashed,rounded',
            color=C_ORBIT,
            fontcolor=C_ORBIT,
            penwidth='1.5',
        )
        s.node(
            'run_orbit',
            'run_orbit()',
            fillcolor=C_ORBIT,
            fontcolor='white',
            URL=src_url('orbit/wrapper.py'),
            target='_blank',
            tooltip='Orbital evolution, tidal heating, Hill radius',
        )
        for backend, path, tip in [
            ('LovePy\n(Julia)', 'orbit/lovepy.py', 'Multi-phase tidal heating'),
            ('dummy', 'orbit/dummy.py', 'Dummy orbit for testing'),
        ]:
            nid = 'orb_' + backend.split('\n')[0].lower()
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('run_orbit', nid, style='dashed', arrowsize='0.6')

    # ── Stellar Flux ──────────────────────────────────────────────────
    with g.subgraph(name='cluster_star') as s:
        s.attr(
            label='Stellar Evolution & Spectrum',
            style='dashed,rounded',
            color='#B8860B',
            fontcolor='#B8860B',
            penwidth='1.5',
        )
        s.node(
            'stellar_flux',
            'Stellar Flux\nManagement',
            fillcolor=C_STAR,
            URL=src_url('star/wrapper.py'),
            target='_blank',
            tooltip='update_stellar_quantities, get_new_spectrum, scale_spectrum_to_toa',
        )
        for backend, path, tip in [
            ('MORS – Spada\n(Python)', 'star/wrapper.py', 'Spada rotation-evolution tracks'),
            ('MORS – Baraffe\n(Python)', 'star/wrapper.py', 'Baraffe pre-main-sequence tracks'),
            ('dummy', 'star/wrapper.py', 'Fixed blackbody spectrum'),
        ]:
            nid = 'star_' + backend.split('\n')[0].lower().replace(' ', '_').replace('–', '')
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('stellar_flux', nid, style='dashed', arrowsize='0.6')

    # ── Escape ────────────────────────────────────────────────────────
    with g.subgraph(name='cluster_escape') as s:
        s.attr(
            label='Atmospheric Escape',
            style='dashed,rounded',
            color=C_ESCAPE,
            fontcolor=C_ESCAPE,
            penwidth='1.5',
        )
        s.node(
            'run_escape',
            'run_escape()',
            fillcolor=C_ESCAPE,
            URL=src_url('escape/wrapper.py'),
            target='_blank',
            tooltip='Energy-limited or detailed atmospheric escape',
        )
        for backend, path, tip in [
            ('ZEPHYRUS\n(Python)', 'escape/wrapper.py', 'Energy-limited escape'),
            ('Boreas\n(Python)', 'escape/boreas.py', 'Detailed escape model'),
            ('dummy', 'escape/wrapper.py', 'Fixed mass-loss rate'),
        ]:
            nid = 'esc_' + backend.split('\n')[0].lower()
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('run_escape', nid, style='dashed', arrowsize='0.6')

    # ── Outgassing ────────────────────────────────────────────────────
    with g.subgraph(name='cluster_outgas') as s:
        s.attr(
            label='Volatile Outgassing',
            style='dashed,rounded',
            color=C_OUTGAS,
            fontcolor=C_OUTGAS,
            penwidth='1.5',
        )
        s.node(
            'run_outgas',
            'run_outgassing()',
            fillcolor=C_OUTGAS,
            fontcolor='white',
            URL=src_url('outgas/wrapper.py'),
            target='_blank',
            tooltip='Solve volatile in-/outgassing and surface pressures',
        )
        s.node(
            'outgas_calliope',
            'CALLIOPE\n(Python)',
            fillcolor=C_BACKEND,
            shape='box',
            style='filled',
            URL=src_url('outgas/wrapper.py'),
            target='_blank',
            tooltip='Volatile thermodynamics and speciation',
        )
        s.edge('run_outgas', 'outgas_calliope', style='dashed', arrowsize='0.6')

    # ── Atmosphere Climate ────────────────────────────────────────────
    with g.subgraph(name='cluster_atmos') as s:
        s.attr(
            label='Atmosphere Climate',
            style='dashed,rounded',
            color=C_ATMOS,
            fontcolor=C_ATMOS,
            penwidth='1.5',
        )
        s.node(
            'run_atmos',
            'run_atmosphere()',
            fillcolor=C_ATMOS,
            fontcolor='white',
            URL=src_url('atmos_clim/wrapper.py'),
            target='_blank',
            tooltip='Radiative-convective atmosphere solver',
        )
        for backend, path, tip in [
            ('JANUS\n(Python)', 'atmos_clim/janus.py', '1-D convective atmosphere'),
            ('AGNI\n(Julia)', 'atmos_clim/agni.py', 'Radiative-convective energy balance'),
            ('dummy', 'atmos_clim/dummy.py', 'Parameterised atmosphere'),
        ]:
            nid = 'atm_' + backend.split('\n')[0].lower()
            s.node(
                nid,
                backend,
                fillcolor=C_BACKEND,
                shape='box',
                style='filled',
                URL=src_url(path),
                target='_blank',
                tooltip=tip,
            )
            s.edge('run_atmos', nid, style='dashed', arrowsize='0.6')

    # ── Housekeeping & Convergence ────────────────────────────────────
    g.node(
        'housekeeping',
        'Housekeeping\n& Convergence Check',
        fillcolor=C_HOUSE,
        URL=src_url('utils/terminate.py'),
        target='_blank',
        tooltip='check_termination(): solidification, energy balance, escape, time, iterations',
    )
    g.node(
        'converged',
        'Converged?',
        shape='diamond',
        fillcolor=C_DECISION,
        penwidth='2',
        width='1.5',
        height='0.7',
        URL=src_url('utils/terminate.py'),
        target='_blank',
        tooltip='Two sequential iterations must satisfy criteria (strict mode)',
    )

    # ── Post-loop ─────────────────────────────────────────────────────
    with g.subgraph(name='cluster_postloop') as s:
        s.attr(
            label='Post-processing (after loop)',
            style='dashed,rounded',
            color='#555555',
            fontcolor='#555555',
            penwidth='1.5',
        )
        s.node(
            'run_chem',
            'Offline Chemistry\n(VULCAN)',
            fillcolor=C_CHEM,
            URL=src_url('atmos_chem/wrapper.py'),
            target='_blank',
            tooltip='run_chemistry(): offline atmospheric kinetics',
        )
        s.node(
            'run_obs',
            'Synthetic Observations\n(PLATON)',
            fillcolor=C_OBS,
            fontcolor='white',
            URL=src_url('observe/wrapper.py'),
            target='_blank',
            tooltip='run_observe(): transit/eclipse depth spectra',
        )
        s.node(
            'finalise',
            'Final Plots\n& Archive',
            fillcolor=C_HOUSE,
            URL=src_url('proteus.py'),
            target='_blank',
            tooltip='UpdatePlots, archive, print_citation',
        )

    g.node(
        'end',
        'END',
        shape='oval',
        fillcolor='#333333',
        fontcolor='white',
        penwidth='2',
    )

    # ── Main-loop edges ───────────────────────────────────────────────
    g.edge('loop_start', 'run_interior', label='hf_row')
    g.edge(
        'run_interior',
        'solve_structure',
        label='M_int, R_int\nT_magma, Phi_global',
        style='dotted',
    )
    g.edge('run_interior', 'run_orbit', label='dt, T profiles')
    g.edge('run_orbit', 'stellar_flux', label='separation\nF_tidal')
    g.edge('stellar_flux', 'run_escape', label='F_ins, F_xuv\nstellar spectrum')
    g.edge('run_escape', 'run_outgas', label='element inventories\nesc_rate')
    g.edge('run_outgas', 'run_atmos', label='partial pressures\nP_surf, VMRs')
    g.edge('run_atmos', 'housekeeping', label='F_atm, T_surf\nR_obs, albedo')
    g.edge('housekeeping', 'converged')
    g.edge('converged', 'loop_start', label='No', color='#E15759', fontcolor='#E15759')
    g.edge('converged', 'run_chem', label='Yes', color='#59A14F', fontcolor='#59A14F')

    # Post-loop flow
    g.edge('run_chem', 'run_obs')
    g.edge('run_obs', 'finalise')
    g.edge('finalise', 'end')

    # ── Config node (floating) ────────────────────────────────────────
    g.node(
        'config',
        'Config (TOML)\nversion 2.0',
        shape='note',
        fillcolor='#FFFFCC',
        URL=src_url('config/_config.py'),
        target='_blank',
        tooltip='Config class: params, star, orbit, struct, atmos_clim, atmos_chem, escape, interior, outgas, delivery, observe',
    )
    g.edge('config', 'init', style='dotted', label='parsed config', arrowhead='open')

    # ── Legend ─────────────────────────────────────────────────────────
    with g.subgraph(name='cluster_legend') as s:
        s.attr(
            label='Legend',
            style='rounded',
            color='#999999',
            fontcolor='#555555',
            penwidth='1',
            rank='sink',
        )
        s.node(
            'leg_module',
            'Physics module\n(wrapper)',
            fillcolor=C_ATMOS,
            fontcolor='white',
            shape='box',
            style='filled,rounded',
        )
        s.node(
            'leg_backend',
            'Backend\nimplementation',
            fillcolor=C_BACKEND,
            shape='box',
            style='filled',
        )
        s.node('leg_decision', 'Decision', fillcolor=C_DECISION, shape='diamond')
        s.node('leg_data', 'Data flow', shape='plaintext', fillcolor='white', style='')
        s.edge('leg_module', 'leg_backend', style='dashed', label='selects')
        s.edge('leg_decision', 'leg_data', style='invis')

    return g


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

HTML_TEMPLATE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PROTEUS Framework – Architecture Flowchart</title>
<style>
  body {{
    font-family: Helvetica, Arial, sans-serif;
    margin: 2rem auto;
    max-width: 1400px;
    background: #fafafa;
    color: #333;
  }}
  h1 {{ text-align: center; }}
  .description {{
    max-width: 800px;
    margin: 0 auto 1.5rem;
    line-height: 1.5;
    font-size: 0.95rem;
    color: #555;
  }}
  .chart-container {{
    text-align: center;
    overflow-x: auto;
  }}
  .chart-container svg {{
    max-width: 100%;
    height: auto;
  }}
  footer {{
    text-align: center;
    margin-top: 2rem;
    font-size: 0.85rem;
    color: #999;
  }}
</style>
</head>
<body>
<h1>PROTEUS Framework &ndash; Architecture Flowchart</h1>
<div class="description">
  <p>
    This diagram shows the coupling loop and module architecture of the
    <a href="https://github.com/FormingWorlds/PROTEUS">PROTEUS</a> framework.
    <strong>Click on any node</strong> to navigate to the corresponding source
    file on GitHub.  Solid arrows indicate the main execution flow; dashed
    arrows connect wrapper functions to their selectable backend
    implementations.  Edge labels describe the key variables exchanged between
    modules.
  </p>
  <p>
    <em>Regenerate this chart:</em>
    <code>python tools/generate_flowchart.py --output-dir docs/assets</code>
  </p>
</div>
<div class="chart-container">
{svg_content}
</div>
<footer>
  Auto-generated by <code>tools/generate_flowchart.py</code>
</footer>
</body>
</html>
""")


def wrap_html(svg_content: str) -> str:
    """Embed SVG content inside a self-contained HTML page."""
    return HTML_TEMPLATE.format(svg_content=svg_content)


# ---------------------------------------------------------------------------
# SVG post-processing
# ---------------------------------------------------------------------------


def _embed_images_as_base64(svg_path: Path) -> None:
    """Replace file-path image references in the SVG with base64 data URIs.

    This makes the SVG self-contained so it renders correctly when served
    from any location (including inside the HTML wrapper).
    """
    svg_text = svg_path.read_text(encoding='utf-8')

    def _replace_href(match: re.Match) -> str:
        attr = match.group(1)  # 'xlink:href' or 'href'
        file_path = match.group(2)
        img = Path(file_path)
        if not img.exists():
            return match.group(0)  # leave unchanged
        suffix = img.suffix.lower()
        mime = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
        }.get(suffix, 'application/octet-stream')
        b64 = base64.b64encode(img.read_bytes()).decode('ascii')
        return f'{attr}="data:{mime};base64,{b64}"'

    # Match both xlink:href="..." and href="..." pointing to local files
    svg_text = re.sub(
        r'((?:xlink:)?href)="(/[^"]+\.(?:png|jpg|jpeg|gif|svg))"',
        _replace_href,
        svg_text,
    )

    svg_path.write_text(svg_text, encoding='utf-8')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Generate PROTEUS architecture flowchart.',
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=Path,
        default=Path('docs/assets'),
        help='Directory for output files (default: docs/assets)',
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['svg', 'pdf', 'png'],
        default='svg',
        help='Image format (default: svg). HTML wrapper is only generated for SVG.',
    )
    parser.add_argument(
        '--github-base',
        default=GITHUB_BASE,
        help='Base URL for GitHub source links (default: %(default)s)',
    )
    parser.add_argument(
        '--logo',
        default=str(DEFAULT_LOGO),
        help='Path to logo PNG (default: %(default)s). Set to empty string to disable.',
    )
    args = parser.parse_args()

    # Resolve logo path
    logo_path = Path(args.logo) if args.logo else None
    if logo_path and not logo_path.exists():
        print(f'WARNING: Logo file not found at {logo_path}, skipping logo.')
        logo_path = None

    # Build graph
    g = build_graph(github_base=args.github_base, logo_path=logo_path)
    g.format = args.format

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Render
    out_path = args.output_dir / 'proteus_flowchart'
    g.render(filename=str(out_path), cleanup=True)
    rendered_file = Path(str(out_path) + '.' + args.format)

    # Embed logo as base64 in SVG so it is self-contained
    if args.format == 'svg' and logo_path:
        _embed_images_as_base64(rendered_file)

    print(f'Rendered: {rendered_file}')

    # HTML wrapper (SVG only)
    if args.format == 'svg':
        svg_text = rendered_file.read_text(encoding='utf-8')
        html_path = args.output_dir / 'proteus_flowchart.html'
        html_path.write_text(wrap_html(svg_text), encoding='utf-8')
        print(f'HTML:     {html_path}')


if __name__ == '__main__':
    main()
