<h1 align="center">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
</h1>

<p align="center">
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/tests.yaml">
    <img src="https://github.com/FormingWorlds/PROTEUS/actions/workflows/tests.yaml/badge.svg">
  </a>
  <a href="https://fwl-proteus.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/fwl-proteus/badge/?version=latest">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <a href="https://github.com/FormingWorlds/PROTEUS/actions/workflows/tests.yaml">
    <img src="https://gist.githubusercontent.com/stefsmeets/b4ee7dab92e20644bcb3a5ad09f71165/raw/covbadge.svg">
  </a>
  <a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JE008576"><img src="https://img.shields.io/badge/DOI-10.1029%2F2024JE008576-blue"></a>
</p>

<b>PROTEUS</b> (/ˈproʊtiəs, PROH-tee-əs) is a modular Python framework that simulates the coupled evolution of the atmospheres and interiors of rocky planets and exoplanets. Inspired by the Greek god of elusive sea change, who could change his form at will, PROTEUS is designed to be flexible and adaptable to a wide range of planetary environments. It can foretell the future, but answers only to those who are capable of asking the right questions.<br>

<p align="center">
      <img src="assets/schematic.png" style="max-width: 82%; height: auto;"></br>
      <b>Schematic of PROTEUS components and corresponding modules.</b> </br>
</p>

See the [model description](model.html) for an outline of the code and its structure. See the [installation guide](installation.html) and [usage guide](usage.html) for initial steps and troubleshooting advice. Only attempt to run PROTEUS after you have followed all of the installation instructions.

## Contributors

| Name                    | Email address                               |
| -                       | -                                           |
| Tim Lichtenberg         | tim.lichtenberg[at]rug.nl                   |
| Harrison Nicholls       | harrison.nicholls[at]physics.ox.ac.uk       |
| Laurent Soucasse        | l.soucasse[at]esciencecenter.nl             |
| Mariana Sastre          | m.c.villamil.sastre[at]rug.nl               |
| Emma Postolec           | e.n.postolec[at]rug.nl                      |
| Flavia Pascal           | f.pascal[at]student.rug.nl                  |
| Ben Riegler             | ben.riegler[at]tum.de                       |
| Hanno Spreeuw           | h.spreeuw[at]esciencecenter.nl              |
| Dan J. Bower            | dbower[at]ethz.ch                           |
| Mark Hammond            | mark.hammond[at]physics.ox.ac.uk            |
| Stef Smeets             | s.smeets[at]esciencecenter.nl               |
| Raymond Pierrehumbert   | raymond.pierrehumbert[at]physics.ox.ac.uk   |

## Citation and credit

If you make use of PROTEUS, please reference the following papers, state the code version used, and include an acknowledgement. We provide a suggested acknowledgement in the [contributing page](CONTRIBUTING.html#licensing-and-credit)

```bibtex

@ARTICLE{Nicholls_2025_MNRAS,
       author = {{Nicholls}, Harrison and {Pierrehumbert}, Raymond T. and {Lichtenberg}, Tim and {Soucasse}, Laurent and {Smeets}, Stef},
        title = "{Convective shutdown in the atmospheres of lava worlds}",
      journal = {\mnras},
     keywords = {Astrophysics - Earth and Planetary Astrophysics},
         year = 2025,
        month = jan,
       volume = {536},
       number = {3},
        pages = {2957-2971},
          doi = {10.1093/mnras/stae2772},
archivePrefix = {arXiv},
       eprint = {2412.11987},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.536.2957N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Nicholls_2024_JGRP,
       author = {{Nicholls}, Harrison and {Lichtenberg}, Tim and {Bower}, Dan J. and {Pierrehumbert}, Raymond},
        title = "{Magma Ocean Evolution at Arbitrary Redox State}",
      journal = {Journal of Geophysical Research (Planets)},
     keywords = {magma oceans, lava planets, exoplanets, atmospheres, simulation, convection, Astrophysics - Earth and Planetary Astrophysics},
         year = 2024,
        month = dec,
       volume = {129},
       number = {12},
        pages = {2024JE008576},
          doi = {10.1029/2024JE008576},
archivePrefix = {arXiv},
       eprint = {2411.19137},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024JGRE..12908576N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Lichtenberg_2021_JGRP,
       author = {{Lichtenberg}, Tim and {Bower}, Dan J. and {Hammond}, Mark and {Boukrouche}, Ryan and {Sanan}, Patrick and {Tsai}, Shang-Min and {Pierrehumbert}, Raymond T.},
        title = "{Vertically Resolved Magma Ocean-Protoatmosphere Evolution: H$_{2}$, H$_{2}$O, CO$_{2}$, CH$_{4}$, CO, O$_{2}$, and N$_{2}$ as Primary Absorbers}",
      journal = {Journal of Geophysical Research (Planets)},
     keywords = {Atmosphere origins, exoplanets, magma oceans, planet composition, planet formation and evolution, planetary surface, Astrophysics - Earth and Planetary Astrophysics, Physics - Atmospheric and Oceanic Physics, Physics - Geophysics},
         year = 2021,
        month = feb,
       volume = {126},
       number = {2},
          eid = {e06711},
        pages = {e06711},
          doi = {10.1029/2020JE006711},
archivePrefix = {arXiv},
       eprint = {2101.10991},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021JGRE..12606711L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
