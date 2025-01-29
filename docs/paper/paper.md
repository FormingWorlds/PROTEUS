---
title: 'PROTEUS: A radiative-convective model for lava planet atmospheres.'
tags:
  - astronomy
  - physics,
  - exoplanets
  - planets
authors:
  - name: Tim Lichtenberg
    orcid: 0000-0002-3286-7683
    affiliation: 1
  - name: Mariana V. Sastre
    affiliation: 1
  - name: Emma Postolec
    affiliation: 1
  - name: Harrison Nicholls
    orcid: 0000-0002-8368-4641
    affiliation: 2
  - name: Laurent Soucasse
    orcid: 0000-0002-5422-8794
    affiliation: 3
  - name: Stef Smeets
    affiliation: 3
  - name: Dan J. Bower
    affiliation: 4, 5
affiliations:
 - name: Kapteyn Astronomical Institute, University of Groningen, P.O. Box 800, 9700 AV Groningen, The Netherlands
   index: 1
 - name: Department of Physics, University of Oxford, Parks Road, Oxford OX1 3PU, UK
   index: 2
 - name: Netherlands eScience Center, Science Park 402, NL-1098 XH Amsterdam, The Netherlands
   index: 3
 - name: Center for Space and Habitability, University of Bern, Gesellschaftsstrasse 6, 3012 Bern, Switzerland
   index: 4
 - name: Institute of Geochemistry and Petrology, ETH Zurich, Clausiusstrasse 25, 8092 Zürich, Switzerland
   index: 5
date: January 2025
bibliography: paper.bib

---

# Summary

With the James Webb Space Telescope (JWST), astronomy can now characterize hot rocky extrasolar planets that are potentially similar to the conditions of the primitive Earth right after its formation. Characterizing the thermodynamic and climatic properties of these worlds will enable us to better understand the geologic conditions that enabled the origin of life on our own world.

PROTEUS[^1] is an adaptive numerical framework designed to solve the long-term evolution of lava planets. It couples multiple physical codes, each designed to model a specific component or phenomenon (interior dynamics, atmosphere dynamics, stellar evolution...). It enables advanced simulation of the geophysical evolution and outgassing atmospheres of lava planets [@lichtenber21; @nicholls24a].

[^1]: The PROTEUS framework can be found on GitHub [here](https://github.com/FormingWorlds/PROTEUS).

# Statement of need

Scientific context. Scientifc objectives. State of the art.

# Description of the platform

| Module | Component | Description | Code | Reference |
| ------ | --------- | ----------- | ---- | --------- |
| JANUS | atmosphere | implements the multicomponent non‐dilute pseudoadiabat | Python | [@graham21] |
| AGNI | atmosphere | solves the atmosphere energy balance using a Newton-Raphson algorithm | Julia | [@nicholls24b]  |
| SOCRATES | atmosphere | computes radiative fluxes from atmospheric temperature and composition | Fortran | [@manners24-tech] |
| SPIDER | interior | solves interior heat transport of partially molten planets using an entropy formulation | C++ | [@bower18] |
| ARAGOG | interior | solves interior heat transport of partially molten planets using a temperature formulation | Python | [@bower18] |
| MORS | star | models the evolution of rotation and high energy emission of stars  | Python | [@johstone21] |



# Acknowledgements

Tim Lichtenberg acknowledges support from the Netherlands eScience Center under grant number NLESC.OEC.2023.017.

# References
