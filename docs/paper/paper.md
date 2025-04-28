---
title: 'The PROTEUS Framework for Planetary Evolution'
tags:
  - astronomy
  - exoplanets
  - planetary science
  - geophysics
  - atmospheric science
  - geochemistry
  - geodynamics
  - planetary evolution
  - lava planets
authors:
  - name: PROTEUS Collaboration
  - name: Tim Lichtenberg
    orcid: 0000-0002-3286-7683
    corresponding: true
    affiliation: 1
  - name: Harrison Nicholls
    orcid: 0000-0002-8368-4641
    affiliation: 2
  - name: Mariana Sastre
    orcid: 0009-0008-7799-7976
    affiliation: 1
  - name: Emma Postolec
    affiliation: 1
  - name: Laurent Soucasse
    orcid: 0000-0002-5422-8794
    affiliation: 3
  - name: Dan J. Bower
    orcid: 0000-0002-0673-4860
    affiliation: 4
  - name: Stef Smeets
    orcid: 0000-0002-5413-9038
    affiliation: 3
  - name: Mark Hammond
    orcid: 0000-0002-6893-522X
    affiliation: 2
  - name: Shang-Min Tsai
    orcid: 0000-0002-8163-4608
    affiliation: 5
  - name: Raymond T. Pierrehumbert
    orcid: 0000-0002-5887-1197
    affiliation: 2
affiliations:
 - name: Kapteyn Astronomical Institute, University of Groningen, Groningen, The Netherlands
   index: 1
 - name: Department of Physics, University of Oxford, Oxford, United Kingdom
   index: 2
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 3
 - name: Institute of Geochemistry and Petrology, ETH Zurich, Zurich, Switzerland
   index: 4
 - name: Institute of Astronomy and Astrophysics, Academia Sinica, Taipei, Taiwan
   index: 5

date: June 2025
bibliography: paper.bib

---

# Summary

Novel astronomical instrumentation such as the James Webb Space Telescope (JWST) enable the spectral characterization of low-mass extrasolar planets, so-called super-Earths and sub-Neptunes. Many of these planets orbit close to their star under high irradiation and on eccentric orbits. These conditions create thermodynamic environments that are potentially similar to the climatic and geodnamic regime of the primitive Earth right after its formation, enabling direct observational access to highly molten phases of planetary evolution, so-called magma ocean epochs. Characterizing the thermodynamic and climatic properties of these worlds may yield critical insights to better understand the geologic and climatic conditions that enabled the origin of life on our own world.

The atmospheric, surficial, and geologic conditions during magma ocean epochs arise from feedback between multiple coupled and non-linear processes, including mantle melting, geochemical composition, outgassing, greenhouse forcing, condensation, and atmospheric escape, which makes it difficult to robustly identify long-term evolutionary trends that can be compared against astronomical observations. PROTEUS[^1] is an adaptive and modular numerical framework designed to solve this challenge of the long-term coupled evolution of the atmospheres and interiors of rocky  planets and exoplanets. It couples the iterative solution of several physical and chemical modules, each specifically designed to model a specific component or phenomenon of the integrated planetary system. Through this method it enables robust physical and numerical tests against known solutions of individual sub-processes, and thus enables advanced simulation of the geophysical evolution and outgassing and escaping atmospheres of lava worlds and temperate rocky exoplanets [@lichtenberg21a; @nicholls24a, @nicholls25a].

PROTEUS is typically used as an executable program, where it reads TOML configuration files from the disk and outputs data files and figures to a specified directory. The documentation can be read online [here](https://fwl-proteus.readthedocs.io/en/latest/).

[^1]: The PROTEUS framework can be found on GitHub [here](https://github.com/FormingWorlds/PROTEUS).

# Statement of need

Scientific context. Scientifc objectives. State of the art.

![Caption for example figure.\label{fig:example}](figure.png)

# Description of the platform

| Module | Component | Description | Code | Reference |
| ------ | --------- | ----------- | ---- | --------- |
| JANUS | atmosphere | implements the multicomponent non‐dilute pseudoadiabat | Python | [@graham21] |
| AGNI | atmosphere | solves the atmosphere energy balance using a Newton-Raphson algorithm | Julia | [@nicholls25a]  |
| SOCRATES | atmosphere | computes radiative fluxes from atmospheric temperature and composition | Fortran | [@manners24-tech] |
| SPIDER | interior | solves interior heat transport of partially molten planets using an entropy formulation | C++ | [@bower18] |
| ARAGOG | interior | solves interior heat transport of partially molten planets using a temperature formulation | Python | [@bower18] |
| MORS | star | models the evolution of rotation and high energy emission of stars  | Python | [@johstone21] |

# Comparison with other codes


# Acknowledgements

Tim Lichtenberg acknowledges support from the Netherlands eScience Center under grant number NLESC.OEC.2023.017.

# References
