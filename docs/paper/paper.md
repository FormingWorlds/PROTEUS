---
title: 'The PROTEUS framework for planetary evolution'
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

[PROTEUS](https://github.com/FormingWorlds/PROTEUS) is an adaptive and modular numerical framework designed to address the interdisciplinary challenge of understanding the long-term coupled evolution of the atmospheres and interiors of rocky planets and exoplanets. It iteratively couples the numerical solution of several physical and chemical modules, each of which are designed to accurately describe a specific component of the planet and its environment, for example atmospheric radiative transfer, stellar evolution, volatile outgassing, or mantle convection. Through this method PROTEUS enables to resolve time-sensitive hysteresis processes on a global, planetary scale that are not accessible through steady-state approaches. Its modularity allows robust physical and numerical tests against known analytical solutions and empirical evidence of individual sub-processes. The current primary use case of PROTEUS is the simulation of the coupled geophysical and climatic evolution of individual or ensembles of rocky (exo-)planets from their primordial magma ocean phase to either global energy balance, mantle solidification, or complete atmospheric escape [@lichtenberg21a; @nicholls24a, @nicholls25a]. Through its modular implementation, PROTEUS offers multiple avenues to extend its functionality and use-cases in the future, for example toward more volatile-rich planets, solid-state geodynamics, prebiotic and biotic evolution, and statistical inference methods.

# Background

Advances in astronomical instrumentation, such as the James Webb Space Telescope (JWST), now enable the spectral characterization of low-mass extrasolar planets, in particular so-called super-Earths and sub-Neptunes (Knutson & Kempton), which are worlds between 2 and 4 Earth radii in size and non-existent in the Solar System. Many of these planets orbit very close to their star under high irradiation and on eccentric orbits (Zhu & Dong). These conditions create thermodynamic environments that are potentially similar to the climatic and geodynamic regime of the primitive Earth right after its formation, enabling direct observational access to highly molten phases of planetary evolution, so-called magma ocean epochs. These geodynamic regimes governed the early, formative phases of the terrestrial planets (Chao 2021), but are inaccessible at present day in the Solar System.

Characterizing the thermodynamic and climatic properties of these exoplanets in fully or partially molten regimes may thus yield critical insights to better understand the planetary conditions that governed the formation of the earliest atmospheres of the terrestrial planets and built the background environment of the origin of life (Lichtenberg & Miguel 2025). Resolving the physical origin of the diversity in the observed exoplanet population is key to building a background understanding of the abiotic landscape of terrestrial worlds. Only through such enhanced understanding of the planetary context can we built the quantiative framework through which exoplanet astronomy will enable the identification of life as we know it on other worlds.

# Statement of need

The atmospheric, surficial, and geologic conditions during magma ocean epochs arise from feedback between multiple coupled and non-linear processes, which include mantle melting, geochemical evolution, outgassing, greenhouse forcing, condensation, and atmospheric escape (Lichtenberg et al. 2023). The interconnectedness of planetary mantle and atmosphere create interconnected feedback loop that lead to hysteresis of the planetary climate and interior on billion-year timescales.

This emergent property of coupled interior-atmosphere systems is illustrated by the discussion surrounding the 'runaway greenhouse' climate state for Earth and Venus, and for extrasolar planets. If rocky planets are modelled in a steady-state with a pre-exisiting water ocean and freely chosen atmospheric compositions (Way, Yang, Selsis, Madhusudhan), it is found that such planets can retain their water inventory and remain 'habitable' on geologic timescales. However, if modelled as starting in a hot magma ocean state, as predicted by planetary formation scenarios and supported by empirical evidence from the Solar System, these solutions are not recovered (Kite, Lichtenberg, Nicholls, Boer, Shorttle); instead long-lived magma oceans are found to be defining features of planetary evolution, in particular under the extreme stellar irradiation of the currently known exoplanet population.

One key reason for this model divergence is the emerging feedback loop between molten mantle and atmosphere: atmospheric volatiles, including H2O and N and S species, are highly soluble in magmatic fluids (Suer, Sossi, Schaefer), hence the rocky, molten mantle acts as a significant sink of atmospheric volatiles, which selectively draws volatiles out of the atmosphere into the mantle (Dorn & Lichtenberg, Shorttle, Nicholls). This changes the energy transfer through the atmosphere by affecting its pressure, opacity, and reflection propertis, in turn changing the heat loss or gain of the planet through secular cooling or stellar irradiation (Nicholls). Time-sensitive effects, such as continuing accretion (Itcovitz, Wogan), internal redox processes (Lichtenberg, Kite, Deng), photoevaporation (Cherubim, Wordsworth) induced by the host star, or mantle crystallisation (Maurice, Lichtenberg), will affect the global planetary equilibrium over time.

Planets with similar atmospheric properties may thus harbour order of magnitude different bulk volatile fractions if their interior (core and mantle) phase state is different. This presents a critical degeneracy for astronomical observations aiming to infer compositional and thermodynamic properties of exoplanets from telescopic data. On the other hand, evolutionary hysteresis processes may contribute to resolving observational degeneracies. For instance, the solubilities of N and S species are highly sensitive to the compositions properties of the planetary mantle (Suer, Shorttle, Shorttle), hence planets with different geochemistries may be distinguished by matching astronomical observations with time-resolved solutions that connect geochemical with atmospheric considerations over geological timescales. Providing these geophysical and climatic predictions and enable quantative comparison with empirical data from exoplanet astronomy is the primary purpose of the PROTEUS framework in its present form.

# Software modularisation & verification

As modern research software environments grow, they typically become more difficult to maintain and verify. In particular, with a growing user and developer base, sufficient documentation and tutorials become challenging to update on a continuously evolving basis. From a scientific perspective, disciplinary rigour needs to be maintained while interdisciplinary problems are tackled. From an organisational perspective, term-limited research projects and contracts and changing institutions of researchers and software engineers present challenges for incoming code contributors.

The PROTEUS framework attempts to tackle these challenges by modularising its software ecosystem: physical and chemical processes and sub-systems are isolated and mainted in separate git repositories, with their own verification and documentation. From a code philosophy perspective, PROTEUS aims to externalise all physics and chemistry to its interoperable sub-modules. The advantages of this are:

(i) Modules can be developed and tested in relative isolation. Each module is self-sufficient and can be executed standalone, which enhances developer experience and usability. (ii) Modules can be exchanged through other modules, either on a standing or temporary basis, for example to test the sensitivity of one approach over the other. (iii) PROTEUS is in principle interoperable with a variety of external models that fit into the framework designation. In some instances, this enables re-use and extension of pre-existing, highly robust code bases that are efficient and well-maintained. This prevents students and researchers alike to continuously having to reinvent the wheel of their particular domain.

![Schematic of PROTEUS components and corresponding modules. \label{fig:schematic}](PROTEUS_schematic.png)

Figure \ref{fig:schematic} shows the current state of the PROTEUS framework at the time of submission, including its ecosystem of modules, as previously described in [@Lichtenberg21,@nicholls24,@nicholls25a,@nicholls25c]. Several of the currently existing modules (in addition to the PROTEUS code itself) have been written from scratch for their primary use as module within PROTEUS. Other modules, however, are pre-exisiting codes, which have been adapted and extended to work with the PROTEUS framework. The following table provides a list and brief description of the purpose of each module.

| Module | Component | Purpose | Language | Reference |
| ------ | --------- | ----------- | ---- | --------- |
| SPIDER | interior | interior heat transport of partially molten planets using an entropy formulation | C | [@bower18] |
| CALLIOPE | interior | escape of the atmosphere to space  | Python | [@bower22,@shorttle23,@nicholls25a] |
| Aragog | interior | interior heat transport of partially molten planets using a temperature formulation | Python | [@bower18] |
|lovepy | interior | tidal dissipation in the interior | Python | [@hay18,@nicholls25c] |
| AGNI | atmosphere | atmosphere energy balance using a Newton-Raphson algorithm | Julia | [@nicholls25a, @nicholls25b]  |
| JANUS | atmosphere | atmosphere energy balance using a multicomponent non‐dilute pseudoadiabat | Python | [@graham21] |
| ZEPHYRUS | atmosphere | escape of the atmosphere to space  | Python | in prep. |
| fastchem | atmosphere | equilibrium atmospheric chemistry  | Python | [@Kitzmann23] |
| VULCAN | atmosphere | disequilibrium atmospheric chemistry  | Python | [@Tsai17] |
| SOCRATES | atmosphere | radiative fluxes from atmospheric temperature and composition | Fortran | [@manners24-tech] |
| MORS | star | evolution of rotation and high energy emission of stars  | Python | [@johstone21] |
| PLATON | observation | synthetic telescopic observations | Python | [@zhang21] |

# Discussion of similar codes

With the advent of increasing precision in exoplanet characterization, and the focus on smaller planets that approach Earth-like radii and densities, the development of coupled atmosphere-interior codes has recently increased the number of research groups working in this direction. Hence, were here focus specifically on the key trait of (a) time evolution of the planet, (b) coupling between the interior (i.e., a change in one system must dynamically affect other system properties), and (c) the planetary mantle and atmosphere must be described in some fashion that enables quantification of mantle crystallization timescales and changes in atmospheric pressure and/or composition. With this definition, a growing number of codes, mostly proprietary, have been developed over the past few years (Schaefer, Hamano, Bower, Krissansen-Totton, Yeng, Kite, Cowan, Cherubim, Maurice, Lebrun/Massol, Salvador, Majumdar, Carone/Barnes, DLR, Korenaga). The majority of these codes built on the principles of [@ElkinsTanton08].

It is critically important for the advancement of science and the exploration of the exoplanet census that a variety of independent models are developed, optimally in an open source fashion, so that individual approaches can be compared against one another, and the community can learn from each other and thus produce better and more robust science. A detailed comparison with these codes would go way beyond the limits of this article. Hence, we here limit our discussion to the traits that we believe are the unique capabilities and implementation aspects of PROTEUS. These are: (i) Its modularised approach, with the advantages described above. (ii) The ability to spatially resolve the planet (so far in 1-D) from the core-mantle boundary to the top of the atmosphere. Critically, this enables the quantification of thermal evolution scenarios that crystallizse inhomogeneously, and not form the bottom-up, as is a common assumption in most other codes. (iii) The wide variety of geochemistries that can be modelled, which are expressed through the redox state at the mantle-atmosphere interface and result in order-of-magnitude varying atmospheric compositions, which are chemically and energetically self-consistently resolved in the atmospheric modules. (iv) The dynamic resolution of interior and atmospheric energy transfers regimes: radiative and convective layers in the atmosphere; conduction and convection in multiple dynamics regimes are resolved. (v) True multi-phase evolution of the mantle, where melt and solid phases are resolved on individual nodes, affecting energy transfer and chemical properties. (vi) Interconnected atmospheric escape that couples to the planetary interior; i.e., the escaping reservoir is linked and/or disconnected from the 'hidden' reservoir in the deep interior, depending on evolutionary state. (vi) Time-resolved changes in the stellar spectrum and energy flux for a wide array of stellar types. (vii) Inclusion of equilibrium and disequilibrium chemistry in the atmosphere. (viii) The inclusion of realistic, measure surface reflection properties for solid and molten surface conditions. (ix) On-the-fyl computation of observational properties, such as emission and transmission spectrum, planet-to-star contrast ratio, bulk density, and other observational properties of interest. (x) Automated testing of individual modules and the PROTEUS framework as a whole using GitHub CLI. And finally, (xi) a usable and growing documentation and tutorials.


# Documentation

 The documentation and tutorials for PROTEUS can be [accessed online](https://fwl-proteus.readthedocs.io/en/latest/).



# Acknowledgements

Tim Lichtenberg acknowledges support from the Netherlands eScience Center under grant number NLESC.OEC.2023.017.

# References
