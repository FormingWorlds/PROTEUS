---
title: 'PROTEUS: simulating the coupled interior-atmosphere evolution of rocky planets through deep time'
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
  - name: Mara Attia
    orcid: 0000-0002-7971-7439
    affiliation: 1
  - name: Laurent Soucasse
    orcid: 0000-0002-5422-8794
    affiliation: 3, 4
  - name: Patrick Bos
    orcid: 0000-0002-6033-960X
    affiliation: 1, 5
  - name: Mariana Sastre
    orcid: 0009-0008-7799-7976
    affiliation: 1
  - name: Dan J. Bower
    orcid: 0000-0002-0673-4860
    affiliation: 6
  - name: Robb Calder
    orcid: 0009-0002-9247-2437
    affiliation: 2
  - name: Lorenzo Cesario
    orcid: 0009-0009-7228-7809
    affiliation: 1
  - name: Emeline Decocq
    orcid: 0009-0008-3326-9715
    affiliation: 1
  - name: Marijn R. van Dijk
    orcid: 0009-0005-6677-1929
    affiliation: 1
  - name: Mohammad Farhat
    orcid: 0000-0001-7864-6627
    affiliation: 7, 8
  - name: Mark Hammond
    orcid: 0000-0002-6893-522X
    affiliation: 9
  - name: Leoni Janssen
    orcid: 0009-0002-9902-731X
    affiliation: 10
  - name: Tadahiro Kimura
    orcid: 0000-0001-8477-2523
    affiliation: 11, 1
  - name: Imre Kisvárdai
    orcid: 0009-0009-7323-6755
    affiliation: 1
  - name: Ioannis Panagiotou
    orcid: 0009-0005-9147-0431
    affiliation: 1
  - name: Flavia C. Pascal
    orcid: 0009-0007-4663-1456
    affiliation: 1, 2
  - name: Raymond T. Pierrehumbert
    orcid: 0000-0002-5887-1197
    affiliation: 9
  - name: Emma Postolec
    orcid: 0009-0009-5036-3049
    affiliation: 1
  - name: Ben Riegler
    orcid: 0009-0003-9173-9539
    affiliation: 12
  - name: Sara Seager
    orcid: 0000-0002-6892-6948
    affiliation: 13
  - name: Oliver Shorttle
    orcid: 0000-0002-8713-1446
    affiliation: 2, 14
  - name: Stef Smeets
    orcid: 0000-0002-5413-9038
    affiliation: 4
  - name: Hanno Spreeuw
    orcid: 0000-0002-5057-0322
    affiliation: 4
  - name: Karen Stuitje
    orcid: 0009-0000-6847-4331
    affiliation: 1
  - name: Shang-Min Tsai
    orcid: 0000-0002-8163-4608
    affiliation: 15
  - name: Anna Grace Ulses
    orcid: 0000-0002-9031-0824
    affiliation: 16

affiliations:
 - name: Kapteyn Astronomical Institute, University of Groningen, Groningen, The Netherlands
   index: 1
 - name: Institute of Astronomy, University of Cambridge, Cambridge, United Kingdom
   index: 2
 - name: IMEC, Leuven, Belgium
   index: 3
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 4
 - name: Center for Information Technology, University of Groningen, Groningen, The Netherlands
   index: 5
 - name: Department of Earth and Planetary Sciences, ETH Zurich, Zurich, Switzerland
   index: 6
 - name: Department of Astronomy, University of California, Berkeley, Berkeley, CA, USA
   index: 7
 - name: Department of Earth and Planetary Science, University of California, Berkeley, Berkeley, CA, USA
   index: 8
 - name: Atmospheric, Oceanic and Planetary Physics, University of Oxford, Oxford, United Kingdom
   index: 9
 - name: Leiden Observatory, Leiden University, Leiden, The Netherlands
   index: 10
 - name: UTokyo Organization for Planetary Space Science, University of Tokyo, Tokyo, Japan
   index: 11
 - name: School of Computation, Information and Technology, Technical University of Munich, Munich, Germany
   index: 12
 - name: Department of Earth, Atmospheric and Planetary Sciences, Massachusetts Institute of Technology, Cambridge, MA, USA
   index: 13
 - name: Department of Earth Sciences, University of Cambridge, Cambridge, United Kingdom
   index: 14
 - name: Institute of Astronomy and Astrophysics, Academia Sinica, Taipei, Taiwan
   index: 15
 - name: University of Washington, Seattle, WA, USA
   index: 16

date: 25 September 2026
bibliography: paper.bib

---

# Summary

[PROTEUS](https://github.com/FormingWorlds/PROTEUS) ([proteus-framework.org](https://proteus-framework.org)) is an open-source framework that simulates how rocky planets and molten sub-Neptunes evolve over time. It follows them from the magma ocean left by accretion to a solid mantle or a permanently molten interior, beneath an atmosphere that is outgassed, transformed, or lost. The interior structure, the mantle, the atmosphere, the exchange of volatiles between magma and gas, atmospheric escape, the host star, tides, and giant impacts are each described by an independent numerical module. PROTEUS advances these modules together, so that every component feeds back on the others, and it converts the evolved state into synthetic spectra and phase curves for comparison with telescope data. Laboratory measurements of material properties enter as replaceable data tables. The same physics governed the young terrestrial planets of the Solar System, so the framework also serves the reconstruction of the early Earth, Venus, Mars, and Mercury. PROTEUS is written in Python, configured through one [TOML](https://toml.io/en/) file, and published on PyPI as `fwl-proteus`. Every change to the framework or to one of its modules passes an automated suite of unit, integration, and physics benchmark tests on GitHub Actions before it is merged.

# Statement of need

Telescopes such as JWST now measure the atmospheres of super-Earths and sub-Neptunes [@kreidberg25; @lichtenberg25b]. Mass and radius alone leave the interior degenerate, because one bulk density fits many combinations of core, mantle, and envelope. The structure models that are used to break this degeneracy treat a cold planet whose mantle has fully solidified, although every rocky planet starts its life molten [@lichtenberg25]. Many observed low-mass planets receive enough irradiation to stay partially molten for their whole lifetime [@calder26; @nicholls26b], which makes a solid interior the exception.

The state of such a planet cannot be computed one component at a time. Gases outgassed from the magma set the greenhouse forcing, which sets the surface temperature, which sets the melt fraction of the mantle [@lichtenberg21a; @nicholls24]. The melt fraction and the oxidation state of the magma then set how much of each volatile dissolves and how much is outgassed, which closes the loop [@sastre26]. Stellar evolution and atmospheric escape remove gas from the top of this loop [@postolec26a; @cesario26]. Tides add heat at its base [@nicholls25c; @vandijk26], and radiative layers in the atmosphere throttle the cooling [@nicholls25a]. The transition from a hot to a temperate climate does not retrace the reverse path [@boer25]. Atmosphere models that ignore dissolution into the magma are common [@turbet21; @selsis23]. So are magma ocean models that ignore the greenhouse effect of their own atmosphere [@solomatov15; @monteux16], and each captures only half of the coupling. Because the state that emerges depends on the path, it must be integrated forward in time. The consequence of a choice in one component then appears only when the complete system is closed [@lichtenberg26]. Several behaviours were found only in the coupled system. A self-limiting feedback between irradiation, tidal heating, and mantle rheology prolongs magma oceans [@nicholls25c]. Convective shutdown forms deep radiative layers above a magma ocean [@nicholls25a]. Redox-driven re-inflation expands an atmosphere late in a planet's life [@cesario26]. The interior contracts by about 10 per cent as it crystallises [@lichtenberg26].

Each of these feedbacks rests on a laboratory-measured material property [@sossi23; @suer23; @guimond24]. These are the equations of state of silicate melt and solid, the melting curves, and the opacities of hot gases. They also include the solubility and speciation of C-H-N-O-S volatiles as functions of pressure, temperature, and oxygen fugacity, and the redox buffers. Exoplanets occupy conditions without Solar System counterpart, so new experiments extend these properties [@lichtenberg25], and a model must accept a new measurement without being rewritten.

The same magma ocean stage set the starting conditions of the Solar System terrestrial planets, where later evolution has overwritten its record [@lichtenberg23]; strongly irradiated exoplanets expose this stage today. PROTEUS connects the two: it reproduces the early divergence of Earth and Venus [@chili26] and the onset of habitable conditions on the Hadean Earth [@vandijk26]. Because the mantle oxidation state is a free parameter [@nicholls24], one set-up spans the reduced mantle of Mercury and the oxidised mantle of Mars. Its users are exoplanet astronomers who need predictions for JWST and for the facilities that follow it. These are PLATO [@rauer25], Ariel [@tinetti18], and the Roman Space Telescope [@tamburo23]. They also include the extremely large telescopes, LIFE [@quanz22], and the Habitable Worlds Observatory [@stark24]. Geochemists use it to test a material property at planetary scale, and observing teams to interpret a spectrum as the outcome of a history.

# State of the field

Coupled models of a magma ocean and its atmosphere go back to @elkinstanton08 [@lebrun13; @hamano13; @schaefer16; @salvador17]. Later codes added atmospheric escape, geochemistry, or tidal heating [@barnes20; @kite20a; @krissansentotton21; @lichtenberg21a; @bower22]. Recent work applies such models to super-Earths, sub-Neptunes, and lava worlds [@maurice24; @tang24; @carone25; @cherubim25; @farhat25; @sahu25]. Most are not public, and each fixes one set of physical prescriptions in one code base. The CHILI intercomparison, which we initiated, runs these codes on common Earth, Venus, and exoplanet cases [@lichtenberg26b; @chili26]. For Earth the models agree on the solidification time to within 4 Myr, whereas for Venus they diverge and their outgassed atmospheres differ in surface pressure and dominant species. The spread comes from the treatment of mantle dynamics (boundary-layer scaling against a resolved 1-D mantle) and of radiative transfer (gray against non-gray, with or without deep radiative layers). It also comes from volatile trapping in the solid mantle and from escape. It exceeds the sensitivity to the initial volatile inventory. A framework in which each component can be exchanged for an alternative, inside otherwise identical simulations, converts such disagreement into a controlled experiment. PROTEUS is therefore a framework and not one code.

VPLanet [@barnes20] is open source and modular but describes the magma ocean and the atmosphere with parameterised box models. Magrathea [@huang22] solves static interior structure and MESA [@paxton11] evolves stars and giant planets; neither evolves a coupled mantle and atmosphere. No other open-source framework we know of combines the following. The description is resolved in 1-D from the core-mantle boundary to the top of the atmosphere. The mantle tracks melt and solid on individual nodes, and its structure contracts as it crystallises [@lichtenberg26]. Outgassing of C-H-N-O-S volatiles depends on the redox state and is resolved in the atmospheric energy balance. Escape is coupled to the interior reservoir, and the stellar spectrum evolves. Tidal heating, orbital evolution, and accretion by giant impacts are included. Synthetic observables feed a Bayesian retrieval on the evolutionary model itself [@nicholls26].

# Software design

PROTEUS separates the coupling from the physics, and each module is kept in its own repository with its own tests and documentation. The framework itself holds the coupling loop, the configuration schema, the input and output layer, and the tooling for parameter grids and computer clusters (\autoref{fig:schematic}). In each step of the loop the mantle module returns the temperature profile, melt fraction, and surface heat flux for the current atmospheric boundary condition, and the structure module recomputes radius and gravity. The outgassing module partitions the volatile inventory between melt and atmosphere at the current melt mass and oxygen fugacity, and the escape module removes mass under the current stellar spectrum. The atmosphere module solves radiative-convective equilibrium for the new composition, and the star, tides, and accretion modules update irradiation, tidal heating, and planet mass. The time step adapts to the fastest changing quantity. This operator-split design costs a longer installation, more dependencies, and data exchange across the Python, Julia, Fortran, and C boundaries. In return, modules are developed and verified independently, as single processes and as a coupled system, and existing codes are integrated rather than re-implemented. Modules with the same role are also exchangeable (SPIDER or Aragog, CALLIOPE or atmodeller, JANUS or AGNI), so the sensitivity to a modelling choice is measured within one framework. Most roles also have a "dummy" implementation that runs in seconds, which keeps the coupled test suite fast and isolates the effect of one component.

![Module groups of the PROTEUS framework. Interior structure: Zalmoxis [@lichtenberg26], on the PALEOS equations of state [@attia26]. Mantle energetics: Aragog [@lichtenberg26], SPIDER [@bower18], and a boundary-layer model. Atmosphere: AGNI [@nicholls25a; @nicholls25b] and JANUS [@graham21; @graham22]. Radiative transfer: SOCRATES [@manners2024fast], with measured surface reflection properties [@hammond25]. Chemistry: VULCAN kinetics [@tsai17; @tsai21] and FastChem equilibrium [@kitzmann24]. Outgassing: CALLIOPE [@bower22; @shorttle24; @nicholls24] and atmodeller [@bower25], with LavAtmos rock vapour [@vanbuchem23]. Escape: ZEPHYRUS [@postolec26a]. Star: MORS [@johnstone21]. Tides and orbit: LovePy [@hay19; @nicholls25c] and Obliqua. Accretion: Morrigan [@kimura25]. Observation and inference: petitRADTRANS [@molliere19; @nasedkin24] and an evolutionary retrieval module [@nicholls26] operate on the output and are not shown.\label{fig:schematic}](proteus_modules_schematic.png){width=95%}

Laboratory data enter as versioned external tables rather than as code. Equations of state [@attia26], melting curves, solubility laws, opacities, and stellar spectra are downloaded and verified on first use from Zenodo and the Open Science Framework. A new measurement therefore replaces a table and leaves the framework unchanged. A new module implements the interface of its role and is registered in the configuration schema. One validated TOML file defines a simulation, which makes every published run reproducible from its configuration; the configurations of the CHILI cases and the tutorial cases ship with the repository.

Rigour rests on testing at three levels, and every change to PROTEUS or to a module is tested on GitHub Actions before it is merged. A fast pull-request gate runs unit tests with the physics mocked and short smoke runs of the real binaries. A nightly suite runs the complete set, uploads coverage, and raises the coverage threshold, which can only rise. An automated check rejects a new test that lacks an edge case, a failure path, or a physical invariant. Invariants include conservation of mass or energy, boundedness, and agreement with a published reference value. Physical validation compares the modules against analytic limits and between independent implementations of the same role, such as the energy budget of the magma ocean in SPIDER and in Aragog. It also compares them against published interior structure models to 20 Earth masses [@lichtenberg26] and against the independent codes of the CHILI intercomparison [@chili26]. Finally, the framework is tested against data: the sulfur dioxide on L 98-59 d [@nicholls26b] and the Hadean geological record [@vandijk26] are predictions it can fail. Test counts and coverage of every module are published on the [validation page](https://proteus-framework.org/validation/); the [documentation](https://proteus-framework.org/PROTEUS/) covers installation, configuration, and tutorials. This paper describes PROTEUS version XX.XX.XX.

# Research impact statement

PROTEUS predictions are used inside JWST observing programmes, for the lava-planet survey of @sastre26b and for TOI-561 b [@postolec26b]. It explained the sulfur dioxide detected on L 98-59 d with a permanent magma ocean rich in sulfur and hydrogen [@nicholls26b]. Photochemistry computed on its outgassed atmospheres identifies sulfur species as an observational tracer of mantle redox state [@panagiotou26]. PROTEUS takes part in the CHILI intercomparison and supplied the evolutionary inputs for its static-model comparison [@lichtenberg26b; @chili26]. Its modules are also used on their own, AGNI for lava-planet atmospheres [@nicholls25b] and atmodeller for outgassing chemistry [@bower25]. More than 20 contributors have developed the framework openly on GitHub since 2018. Releases of the framework and of its main modules are tagged, published on PyPI, and archived with a DOI on Zenodo. The Forming Worlds Lab at the University of Groningen maintains the project together with the Netherlands eScience Center.

# AI usage disclosure

Generative AI tools assisted the development of PROTEUS and the preparation of this paper. We used GitHub Copilot (code completion, automated fixes, and agent-generated pull requests) and Claude Code (Anthropic; Claude Opus 4.6, Claude Sonnet 5, Claude Opus 5.5, and Claude Fable 5.1). These tools drafted and refactored code, scaffolded tests, wrote and revised documentation, and copy-edited this manuscript. Gemini 3.1 Pro and Gemini 3.8 Flash (Google) reviewed code and documentation. The human authors reviewed, edited, and validated all AI-assisted output. They made all design decisions and are responsible for the content of the software and of this paper.

# Acknowledgements

TL acknowledges support from the Netherlands eScience Center (PROTEUS project, NLESC.OEC.2023.017), the Branco Weiss Foundation, and the Alfred P. Sloan Foundation (AEThER project, G202114194). TL further acknowledges the United States National Aeronautics and Space Administration's Nexus for Exoplanet System Science research coordination network (Alien Earths project, 80NSSC21K0593). RC and OS acknowledge support from the United Kingdom Science and Technology Facilities Council (grant numbers ST/Y509139/1 and UKRI1184).

# References
