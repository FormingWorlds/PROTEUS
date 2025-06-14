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
  - name: Tim Lichtenberg
    orcid: 0000-0002-3286-7683
    corresponding: true
    affiliation: 1
  - name: Harrison Nicholls
    orcid: 0000-0002-8368-4641
    affiliation: 2
  - name: Laurent Soucasse
    orcid: 0000-0002-5422-8794
    affiliation: 3
  - name: Mariana Sastre
    orcid: 0009-0008-7799-7976
    affiliation: 1
  - name: Emma Postolec
    orcid: 0009-0009-5036-3049
    affiliation: 1
  - name: Dan J. Bower
    orcid: 0000-0002-0673-4860
    affiliation: 4
  - name: Flavia Pascal
    affiliation: 1
  - name: Hanno Spreeuw
    orcid: 0000-0002-5057-0322
    affiliation: 3
  - name: Mark Hammond
    orcid: 0000-0002-6893-522X
    affiliation: 2
  - name: Stef Smeets
    orcid: 0000-0002-5413-9038
    affiliation: 3
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
 - name: Institute of Astronomy and Astrophysics, Academia Sinica, Taipei 106319, Taiwan
   index: 5

date: June 2025
bibliography: paper.bib

---

# Summary

[PROTEUS](https://github.com/FormingWorlds/PROTEUS) is an adaptive and modular numerical framework designed to tackle the interdisciplinary challenge of understanding the coupled evolution of the atmospheres and interiors of rocky planets and exoplanets over geologic timescales. It iteratively couples the numerical solution of interoperable physical and chemical modules, each of which are designed to describe a specific component of the planet and its environment, for example atmospheric radiative transfer, stellar evolution, volatile in- and outgassing, or mantle convection. Through this method PROTEUS resolves evolutionary hysteresis processes on a global, planetary scale that are not accessible through steady-state approaches. Its modularity allows robust physical and numerical tests against known analytical solutions and empirical evidence on the level of both individual processes and the interconnected planet system. The current primary use case of PROTEUS is the simulation of the coupled geophysical and climatic evolution of individual or ensembles of rocky (exo-)planets from their primordial magma ocean phase to either global energy balance, mantle solidification, or complete atmospheric escape. Through its modular implementation, PROTEUS offers multiple avenues to extend its functionality and use-cases in the future, for example toward more volatile-rich planets, solid-state geodynamics, prebiotic and biotic chemistry, and statistical inference methods.

# Background

Advances in astronomical instrumentation, such as the launch of the James Webb Space Telescope (JWST), now enable the spectral characterization of low-mass extrasolar planets, in particular so-called super-Earths and sub-Neptunes [@kempton24], which are worlds between 2 and 4 Earth radii in size and absent from the Solar System. Many of these exoplanets orbit very close to their star under high irradiation and on eccentric orbits [@zhu21]. These conditions create thermodynamic environments that are potentially similar to the climatic and geodynamic regime of the primitive Earth right after its formation, enabling direct observational access to highly energetic phases of planetary evolution, magma ocean epochs and runaway greenhouse states. These extreme geodynamic and climatic regimes governed the early, formative phases of the terrestrial planets, but are inaccessible at present day in the Solar System [@lichtenberg23].

Characterizing the thermodynamic and climatic properties of these exoplanets in fully or partially molten regimes will thus yield critical insights to better understand the planetary conditions that governed the formation of the earliest atmospheres of the terrestrial planets and built the background environment of the origin of life as we know it [@lichtenberg25]. Resolving the physical origin of the diversity in the observed exoplanet population is key to building a background understanding of the abiotic landscape of terrestrial worlds. Only through such enhanced understanding of the planetary context can we built the quantitative understanding through which exoplanet astronomy will enable the identification of life as we know it on other worlds.

# Statement of need

The atmospheric, surficial, and geologic conditions during magma ocean epochs arise from feedback between multiple coupled and non-linear processes, which include mantle melting and crystallization, geochemical evolution, outgassing, greenhouse forcing, condensation, and atmospheric escape [@lichtenberg23]. In- and outgassing of atmopsheric volatiles and energy transfer through the planetary mantle and atmosphere create interconnected feedback loops that lead to hysteresis of the planetary climate and interior on billion-year timescales.

This emergent property of coupled interior-atmosphere systems is illustrated by the discussion surrounding the 'runaway greenhouse' climate state for Earth and Venus, and for extrasolar planets. If rocky planets are modelled in a steady-state with a pre-exisiting water ocean and freely chosen atmospheric compositions [@yang13; @way16; @selsis23; @madhusudhan23], it is found that such planets can retain their water inventory and remain 'habitable' on geologic timescales. However, if modelled as starting in a hot magma ocean state, as predicted by planetary formation scenarios and supported by empirical evidence from the Solar System, these solutions are not recovered [@kite20a; @kite20b; @lichtenberg21a; @dorn21; @shorttle24; @nicholls25c; @boer25]; instead long-lived magma oceans are found to be defining features of planetary evolution, in particular under the extreme stellar irradiation of the currently known exoplanet population.

One key reason for this model divergence is the emerging feedback loop between molten mantle and atmosphere: atmospheric volatiles, including H2O and N and S species, are highly soluble in magmatic fluids [@suer23; @sossi23; @schaefer16], hence the rocky, molten mantle acts as a significant sink of atmospheric volatiles, which selectively draws volatiles out of the atmosphere into the mantle [@dorn21; @shorttle24; @nicholls25c]. This changes the energy transfer through the atmosphere by affecting its pressure, opacity, and reflection properties, in turn changing the heat loss or gain of the planet through secular cooling and stellar irradiation. Time-sensitive effects, such as continuing accretion [@itcovitz22; @lichtenberg22; @wogan24], internal redox processes [@wordsworth18; @kite20b; @lichtenberg21b @schaefer24], photoevaporation [@rogers21; @cherubim25] induced by the host star, or mantle crystallisation (Maurice, Lichtenberg), will affect the global planetary equilibrium over time.

Planets with similar atmospheric properties may thus harbour order of magnitude different bulk volatile fractions if their interior (core and mantle) phase state is different. This presents a critical degeneracy for astronomical observations aiming to infer compositional and thermodynamic properties of exoplanets from telescopic data. On the other hand, evolutionary hysteresis processes may contribute to resolving observational degeneracies: the magma solubilities of C, N and S species are highly sensitive to the compositional properties of the planetary mantle [@suer23; @lichtenberg21a; @shorttle24; @nicholls25a], hence planets with different geochemistries may be distinguished by matching astronomical observations with time-resolved solutions that connect geochemical with atmospheric considerations over geological timescales. Providing these geophysical and climatic predictions and enable quantative comparison with empirical data from exoplanet astronomy is the primary purpose of the PROTEUS framework in its present form.

# Framework modularisation

As modern research software environments grow, they typically become more difficult to maintain and verify. In particular, with a growing user and developer base, sufficient documentation and tutorials become challenging to update on a continuously evolving basis. From a scientific perspective, disciplinary rigour needs to be upheld while interdisciplinary research problems are tackled. From an organisational perspective, term-limited research projects and changing institutions of researchers present challenges for code consistency.

The PROTEUS framework attempts to tackle these challenges by modularising its software ecosystem: physical and chemical processes and sub-systems are isolated and maintained in separate git repositories, each with their own verification through automated testing and documentation. From a software engineering perspective, PROTEUS aims to externalise all modelled physics and chemistry to interoperable sub-modules. Some of the advantages of this approach are:

 - Modules can be updated and maintained independently. Each module is self-sufficient and can be executed standalone, which enhances developer experience and usability.
 - Modules can be combined in different ways to create different approaches, which enables the framework to be adapted to a wide range of research questions.
 - Modules can be exchanged through other modules to test the sensitivity of different approaches to the same problem, which enables a more robust understanding of the underlying physics and chemistry.

PROTEUS is thus in principle interoperable with a variety of external computer codes that fit into the framework designation. In some instances, this enables integration and extension of pre-existing codes, preventing researchers from continuously 'reinventing the wheel' of their scientific domain.

![Schematic of PROTEUS framework and implemented modules.\label{fig:schematic}](proteus_schematic.png){width=90%}

\autoref{fig:schematic} shows the current state of the PROTEUS framework at the time of submission, including its ecosystem of modules, as previously introduced in [@lichtenberg21a; @nicholls24; @nicholls25a; @nicholls25c]. Several of the currently existing modules (in addition to the PROTEUS framework itself) have been written from scratch for their primary use as module within PROTEUS. Other modules are specialised codes, which were originally developed stand-alone, and have been adapted and extended to work with the PROTEUS framework.

Modules are grouped into four main categories: (i) interior, (ii) atmosphere, (iii) environment, (iv) interpretation.

Interior modules (i) compute the thermal and chemical evolution of the planetary mantle and core, such as mantle energy transport process, melting and crystallization, and in- and outgassing of volatiles. These include:

  - Aragog and SPIDER [@bower18; @sastre25], which describe the interior heat transport of partially molten planets using an entropy and a temperature formalism, respectively.
  - CALLIOPE [@bower22; @shorttle24; @nicholls25a], which describes the escape of the atmosphere to space.
  - LovePy [@hay19; @nicholls25c], which describes solid-phase tidal heating in the planetary mantle.

Atmosphere modules (ii) compute the energy balance of the planetary atmosphere, including radiative transfer, atmospheric chemistry, and escape processes. These include:

  - AGNI [@nicholls25a; @nicholls25b], which describes the atmosphere energy balance using a radiative-convective model.
  - JANUS [@graham21; @graham22], which describes the atmosphere energy balance using a multicomponent non‐dilute pseudoadiabat.
  - ZEPHYRUS [@postolec25], which describes the escape of the atmosphere to space.
  - FASTCHEM [@kitzmann24], which describes equilibrium atmospheric chemistry.
  - VULCAN [@tsai17; @tsai21], which describes disequilibrium atmospheric chemistry.
  - SOCRATES [@manners24-tech], which describes radiative fluxes from atmospheric temperature and composition.

Environment modules (iii) compute the evolution of the host star, including its luminosity and spectral energy distribution:

 - MORS [@johstone21], which describes the evolution of rotation and high energy emission of stars.

Interpretation modules (iv) compute observational properties of the planet, such as emission and transmission spectra, planet-to-star contrast ratio, and bulk density:

 - PLATON [@zhang19; @zhang20], which describes synthetic telescopic observations of exoplanets.

All module repositories are linked to the PROTEUS framework repository, which provides a single entry point for users to access the entire PROTEUS ecosystem.

# Discussion of similar codes

With the advent of increasing precision in exoplanet characterization, and the focus on smaller planets that approach Earth-like radii and densities, the development of coupled atmosphere-interior codes has recently increased the number of research groups working in this direction. Hence, our discussion here focusses specifically on the key traits of (a) time evolution of the planet, (b) coupling between the interior (i.e., a change in one system must dynamically affect other system properties), and (c) the planetary mantle and atmosphere must be described in some fashion that enables quantification of mantle crystallization timescales and changes in atmospheric pressure and/or composition. With this definition, a growing number of codes, mostly proprietary, have been developed over the past few years [@schaefer16; @hamano13; @bower22; @lichtenberg21a; @krissansentotton21; @tang24; @cherubim25; @kite20a; @maurice24; @lebrun13; @salvador17; @carone25; @farhat25]. The majority of these codes built on the principles developed by @elkinstanton08, but each have their own unique implementations and methodologies.

It is critically important for the exploration of the exoplanet census and refined understanding of the deep history of the terrestrial planets that a variety of independent models are developed, optimally in an open source fashion, so that individual approaches can be compared against one another, and the community can learn from each other and thus produce better and more robust science. A detailed comparison with these codes would go beyond the scope of this article. Hence, we here limit our discussion to the traits that we believe are the unique capabilities and implementation aspects of PROTEUS. These are:

  - Its modularised approach, with the advantages described above.
  - The ability to spatially resolve the planet (so far in 1-D) from the core-mantle boundary to the top of the atmosphere. Critically, this enables the quantification of thermal evolution scenarios that crystallize inhomogeneously, and not solely from the bottom-up, as is a common assumption in most other codes that make use of boundary layer theory for their interior description.
  - The wide variety of geochemistries that can be modelled, which are expressed through the redox state at the mantle-atmosphere interface and result in order-of-magnitude varying atmospheric compositions, which are chemically and energetically self-consistently resolved in the atmospheric modules.
  - The dynamic resolution of interior and atmospheric energy transfers regimes: radiative and convective layers in the atmosphere; two-phase energy and compositional transfer in conductive and (turbulent) convective regimes of the interior are resolved.
  - True multi-phase evolution of the mantle, where melt and solid phases are resolved on individual nodes, affecting energy transfer and chemical properties.
  - Interconnected atmospheric escape that couples to the planetary interior; i.e., the escaping reservoir is dynamically linked and/or disconnected from the volatile reservoir in the deep interior, depending on evolutionary state and redox properties.
  - Time-resolved evolution of the stellar spectrum and energy flux for a wide array of stellar types directly imprint on atmospheric energy transfer and escape.
  - Inclusion of equilibrium and disequilibrium chemistry in the atmosphere.
  - The inclusion of realistic, measured surface reflection properties for solid and molten surface conditions.
  - On-the-fly computation of observational properties, such as emission and transmission spectrum, planet-to-star contrast ratio, bulk density, and other observational properties of interest.
  - Automated testing of individual modules and the PROTEUS framework as a whole using the GitHub continuous integration platform.
  - A usable and growing documentation and tutorials.

# Verification & documentation

PROTEUS implements automated testing and documentation building practices. We use GitHub Actions to automatically run a suite of unit tests, each time code is committed to the public repository or a pull request is opened. The growing test base covers both individual modules within their respective repositories, as well as the PROTEUS framework as a whole. Tests are split into *numerical* tests, which ensure the numerical integrity, and *physical* tests, which compare the code against analytical and numerical results, and empirical data from the scientific literature.

The documentation and tutorials for PROTEUS can be [accessed online](https://fwl-proteus.readthedocs.io/en/latest/).

# Acknowledgements

TL acknowledges support from the Netherlands eScience Center under grant number NLESC.OEC.2023.017, the Branco Weiss Foundation, the Alfred P. Sloan Foundation (AEThER project, G202114194), and the United States National Aeronautic and Space Administration’s Nexus for Exoplanet System Science research coordination network (Alien Earths project, 80NSSC21K0593).

# References
