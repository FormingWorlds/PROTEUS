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

[PROTEUS](https://github.com/FormingWorlds/PROTEUS) ([proteus-framework.org](https://proteus-framework.org)) is an open-source framework that simulates how rocky planets and molten sub-Neptunes evolve over time: from the magma ocean left by accretion to a solid mantle, or to a permanently molten interior, beneath an atmosphere that is outgassed, transformed by chemistry and escape, or lost. The interior structure, the mantle, the atmosphere, the exchange of volatiles between magma and gas, atmospheric escape, the host star, tides, and giant impacts are each described by an independent numerical module. PROTEUS advances these modules together, so that every component feeds back on the others, and converts the evolved state into synthetic spectra and phase curves for comparison with telescope data. Laboratory measurements of material properties enter as replaceable data tables. The same physics governed the young terrestrial planets of the Solar System, so the framework serves the interpretation of exoplanet observations and the reconstruction of the early Earth, Venus, Mars, and Mercury alike. PROTEUS is written in Python, configured through one [TOML](https://toml.io/en/) file, published on PyPI as `fwl-proteus`, and tested on every change.

# Statement of need

Telescopes such as JWST now measure the atmospheres of super-Earths and sub-Neptunes [@kreidberg25; @lichtenberg25b]. Mass and radius alone leave the interior degenerate: one bulk density fits many combinations of core, mantle, and envelope. The structure models that are used to break this degeneracy treat a cold planet whose mantle has fully solidified, although every rocky planet starts its life molten [@lichtenberg25]. Many observed low-mass planets receive enough irradiation to keep their interiors partially molten for their whole lifetime [@lichtenberg25; @calder26; @nicholls26b], so a fully solidified interior is probably the less common case among them.

The state of such a planet cannot be computed one component at a time. Gases outgassed from the magma set the greenhouse forcing, which sets the surface temperature, which sets the melt fraction of the mantle; the melt fraction and the oxidation state of the magma then set how much of each volatile dissolves and how much is outgassed, which closes the loop [@lichtenberg21a; @nicholls24; @sastre26]. Stellar evolution and atmospheric escape remove gas from the top of this loop [@postolec26a; @cesario26], tides add heat at its base [@nicholls25c; @vandijk26], radiative layers in the atmosphere throttle the cooling [@nicholls25a], and the transition from a hot to a temperate climate does not retrace the reverse path [@boer25]. An atmosphere model that ignores dissolution into the magma, and a magma ocean model that ignores the greenhouse effect of its own atmosphere, are both common in the literature; each captures one half of the coupling and loses the other [@lichtenberg25]. The state that emerges depends on the path, so it must be integrated forward in time, and the consequence of a choice in one component appears only when the complete system is closed [@lichtenberg26]. Several behaviours were found only in the coupled system: a self-limiting feedback between irradiation, tidal heating, and mantle rheology that prolongs magma oceans [@nicholls25c], convective shutdown that forms deep radiative layers above a magma ocean [@nicholls25a], redox-driven re-inflation of an atmosphere late in a planet's life [@cesario26], and the contraction of the interior by about 10 per cent as it crystallises [@lichtenberg26].

Each of these feedbacks rests on a laboratory-measured material property: equations of state of silicate melt and solid, melting curves, the solubility and speciation of C-H-N-O-S volatiles as functions of pressure, temperature, and oxygen fugacity, redox buffers, and the opacities of hot gases [@sossi23; @suer23; @guimond24]. Exoplanets occupy conditions without Solar System counterpart, so these properties are being extended by new experiments [@lichtenberg25], and a model must accept a new measurement without being rewritten.

The same magma ocean stage set the starting conditions of the Solar System terrestrial planets, where later evolution has overwritten its record [@lichtenberg23]; strongly irradiated exoplanets expose this stage today. PROTEUS connects the two: it reproduces the early divergence of Earth and Venus [@chili26] and the onset of habitable conditions on the Hadean Earth [@vandijk26], and because the mantle oxidation state is a free parameter [@nicholls24], one set-up spans the reduced mantle of Mercury and the oxidised mantle of Mars. Its users are exoplanet astronomers who need predictions for JWST and for PLATO, Ariel, the Roman Space Telescope, the extremely large telescopes, LIFE, and the Habitable Worlds Observatory [@rauer25; @tinetti18; @tamburo23; @quanz22; @stark24], planetary scientists and geochemists who test a material property at planetary scale, and observing teams who interpret a spectrum as the outcome of a history.

# State of the field

Several codes simulate the coupled evolution of a magma ocean and its atmosphere [@lebrun13; @hamano13; @schaefer16; @salvador17; @barnes20; @kite20a; @krissansentotton21; @lichtenberg21a; @bower22; @maurice24; @tang24; @carone25; @cherubim25; @farhat25; @sahu25], most of them descended from @elkinstanton08. Most are not public, and each fixes one set of physical prescriptions in one code base. The CHILI intercomparison, which we initiated, runs these codes on common Earth, Venus, and exoplanet cases [@lichtenberg26b; @chili26]. For Earth the models agree on the solidification time to within 4 Myr; for Venus they diverge, and their outgassed atmospheres differ in surface pressure and dominant species. The spread comes from the treatment of mantle dynamics (boundary-layer scaling against a resolved 1-D mantle), of radiative transfer (gray against non-gray, with or without deep radiative layers), of volatile trapping in the solid mantle, and of escape, and it exceeds the sensitivity to the initial volatile inventory. A framework in which each of these components can be exchanged for an alternative, inside otherwise identical simulations, converts such disagreement into a controlled experiment. This is why we built PROTEUS as a framework rather than extending one code.

VPLanet [@barnes20] is open source and modular but describes the magma ocean and the atmosphere with parameterised box models. Magrathea [@huang22] solves static interior structure and MESA [@paxton11] evolves stars and giant planets; neither evolves a coupled mantle and atmosphere. To our knowledge, PROTEUS is the only open-source framework that combines a 1-D resolved description from the core-mantle boundary to the top of the atmosphere, a mantle that tracks melt and solid on individual nodes and whose structure contracts as it crystallises [@lichtenberg26], redox-dependent outgassing of C-H-N-O-S volatiles resolved in the atmospheric energy balance, escape coupled to the interior reservoir, an evolving stellar spectrum, tidal heating and orbital evolution, accretion by giant impacts, and synthetic observables with Bayesian retrieval on the evolutionary model itself [@nicholls26].

# Software design

PROTEUS separates the coupling from the physics. Each module is kept in its own repository with its own tests and documentation; the framework holds the coupling loop, the configuration schema, the input and output layer, and the tooling for parameter grids and computer clusters (\autoref{fig:schematic}). In each step of the loop the mantle module returns the temperature profile, melt fraction, and surface heat flux for the current atmospheric boundary condition; the structure module recomputes radius and gravity; the outgassing module partitions the volatile inventory between melt and atmosphere at the current melt mass and oxygen fugacity; the escape module removes mass under the current stellar spectrum; the atmosphere module solves radiative-convective equilibrium for the new composition; and the star, tides, and accretion modules update irradiation, tidal heating, and planet mass. The time step adapts to the fastest changing quantity. This operator-split design costs a longer installation, more dependencies, and data exchange across the Python, Julia, Fortran, and C boundaries. It buys three things: modules are developed and verified independently, as single processes and as a coupled system; modules with the same role are exchangeable, so the sensitivity to a modelling choice is measured within one framework (SPIDER or Aragog for the mantle, CALLIOPE or atmodeller for outgassing, JANUS or AGNI for the atmosphere); and existing codes are integrated rather than re-implemented. Most roles also have a "dummy" implementation that runs in seconds, which keeps the coupled test suite fast and isolates the effect of one component.

![Module groups of the PROTEUS framework. Interior: Zalmoxis [@lichtenberg26] (structure, on the PALEOS equations of state of @attia26), Aragog [@lichtenberg26] and SPIDER [@bower18] (mantle energetics), and a boundary-layer model. Atmosphere: AGNI [@nicholls25a; @nicholls25b] and JANUS [@graham21; @graham22], with SOCRATES radiative transfer [@manners2024fast], measured surface reflection properties [@hammond25], VULCAN chemical kinetics [@tsai17; @tsai21], and FastChem equilibrium chemistry [@kitzmann24]. Outgassing: CALLIOPE [@bower22; @shorttle24; @nicholls24] and atmodeller [@bower25], with LavAtmos rock vapour [@vanbuchem23]. Escape: ZEPHYRUS [@postolec26a]. Star: MORS [@johnstone21]. Tides and orbit: LovePy [@hay19; @nicholls25c] and Obliqua. Accretion: Morrigan [@kimura25]. Observation and inference: petitRADTRANS [@molliere19; @nasedkin24] and an evolutionary retrieval module [@nicholls26] operate on the output and are not shown.\label{fig:schematic}](proteus_modules_schematic.png){width=95%}

Laboratory data enter as versioned external tables rather than as code: equations of state [@attia26], melting curves, solubility laws, opacities, and stellar spectra are downloaded and verified on first use from Zenodo and the Open Science Framework, so a new measurement replaces a table and leaves the framework unchanged. A new module implements the interface of its role and is registered in the configuration schema. One validated TOML file defines a simulation, which makes every published run reproducible from its configuration; the configurations of the CHILI cases and the tutorial cases ship with the repository.

Rigour rests on testing at three levels. Every change to PROTEUS or to a module is tested on GitHub Actions before it is merged: a fast pull-request gate runs unit tests, with the physics mocked, and short smoke runs of the real binaries, and a nightly suite runs the complete set, uploads coverage, and raises the coverage threshold so that it can only increase. An automated check rejects a new test that lacks an edge case, a failure path, or a physical invariant such as conservation of mass or energy, boundedness, or agreement with a published reference value. Physical validation compares the modules against analytic limits, between independent implementations of the same role (the energy budget of the magma ocean in SPIDER and in Aragog), against published interior structure models to 20 Earth masses [@lichtenberg26], and against the independent codes of the CHILI intercomparison [@chili26]. The last test is against data: the sulfur dioxide on L 98-59 d [@nicholls26b] and the geological record of the Hadean Earth [@vandijk26] are predictions the framework can fail. Test counts and coverage of every module are published on the [validation page](https://proteus-framework.org/validation/); the [documentation](https://proteus-framework.org/PROTEUS/) covers installation, configuration, and tutorials. This paper describes PROTEUS version XX.XX.XX.

# Research impact statement

PROTEUS predictions are used inside JWST observing programmes: the framework provided the evolutionary predictions for the lava-planet survey of @sastre26b and for TOI-561 b [@postolec26b], and it explained the sulfur dioxide detected on L 98-59 d with a permanent magma ocean rich in sulfur and hydrogen [@nicholls26b]. Photochemistry computed on its outgassed atmospheres identifies sulfur species as an observational tracer of mantle redox state [@panagiotou26]. PROTEUS takes part in the CHILI intercomparison and supplied the evolutionary inputs for its static-model comparison [@lichtenberg26b; @chili26]. Its modules are used on their own: AGNI as an atmosphere model for lava planets [@nicholls25b] and atmodeller for outgassing chemistry [@bower25]. More than 20 contributors have developed the framework openly on GitHub since 2018. Releases of the framework and of its main modules are tagged, published on PyPI, and archived with a DOI on Zenodo, and the Forming Worlds Lab at the University of Groningen maintains the project together with the Netherlands eScience Center.

# AI usage disclosure

Generative AI tools assisted the development of PROTEUS and the preparation of this paper. We used GitHub Copilot (code completion, automated fixes, and agent-generated pull requests) and Claude Code (Anthropic; Claude Opus 4.6, Claude Sonnet 5, Claude Opus 5.5, and Claude Fable 5.1) to draft and refactor code, scaffold tests, write and revise documentation, and copy-edit this manuscript, and Gemini 3.1 Pro and Gemini 3.8 Flash (Google) to review code and documentation. The human authors reviewed, edited, and validated all AI-assisted output, made all design decisions, and are responsible for the content of the software and of this paper.

# Acknowledgements

TL acknowledges support from the Netherlands eScience Center (PROTEUS project, NLESC.OEC.2023.017), the Branco Weiss Foundation, the Alfred P. Sloan Foundation (AEThER project, G202114194), and the United States National Aeronautic and Space Administration's Nexus for Exoplanet System Science research coordination network (Alien Earths project, 80NSSC21K0593). RC and OS acknowledge support from the United Kingdom Science and Technology Facilities Council (grant numbers ST/Y509139/1 and UKRI1184).

# References
