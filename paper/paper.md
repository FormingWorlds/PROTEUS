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

[PROTEUS](https://github.com/FormingWorlds/PROTEUS) ([proteus-framework.org](https://proteus-framework.org)) is an open-source framework that simulates the coupled evolution of rocky planets and molten sub-Neptunes through deep time. A simulation begins in the magma ocean epoch after accretion and ends with a solid mantle or a permanently molten interior, beneath an atmosphere that is outgassed, altered, or lost. Interior structure, mantle energetics, volatile exchange, atmospheric energy balance, escape, the host star, tides, and giant impacts are each described by a separate module. PROTEUS integrates these modules forward in time, so that each responds to the current state of the others, and converts the evolved planet into synthetic spectra and phase curves. Because the young terrestrial planets of the Solar System passed through the same regime, the framework also serves to reconstruct the early Earth, Venus, Mars, and Mercury. PROTEUS is written in Python, configured through a single [TOML](https://toml.io/en/) file, and distributed on PyPI as `fwl-proteus`. Every change to the framework or its modules has to pass automated unit, integration, and physics benchmark tests on GitHub Actions before being merged.

# Statement of need

Telescopes such as JWST now measure the atmospheres of super-Earths and sub-Neptunes [@kreidberg25; @lichtenberg25b]. However, mass and radius alone do not determine the interior, because one bulk density is consistent with many combinations of core, mantle, and envelope. Structure models used to break this degeneracy typically assume a cold, solid mantle, although every rocky planet begins its evolution molten [@lichtenberg25]. Many observed low-mass planets are irradiated strongly enough to stay partially molten for their entire lifetime [@calder26; @nicholls26b], so a solid interior may well be the exception.

The state of such a planet emerges from the interplay between interior and atmosphere, and it cannot be computed one component at a time. Outgassed volatiles set the greenhouse forcing and hence the surface temperature, which governs the melt fraction of the mantle [@lichtenberg21a; @nicholls24]. Melt fraction and magma redox state in turn control how much of each volatile stays dissolved [@sastre26]. Stellar evolution and atmospheric escape remove volatiles from this cycle [@postolec26a; @cesario26]. Tides heat the mantle [@nicholls25c; @vandijk26], while deep radiative layers in the atmosphere can slow its cooling [@nicholls25a]. Moreover, a planet cooling from a hot to a temperate climate does not retrace the path of its warming [@boer25]. Atmosphere models frequently neglect the dissolution of volatiles into the magma [@turbet21; @selsis23]. Magma ocean models, conversely, often omit the greenhouse effect of their own atmosphere [@solomatov15; @monteux16]. The present state of a planet thus reflects a hysteresis in its evolution and cannot be derived from its current conditions alone, so it has to be obtained by integrating the coupled system forward in time.

These feedbacks depend on laboratory measurements of material properties, such as equations of state [@attia26], volatile solubilities [@sossi23; @suer23], and gas opacities. Laboratory experiments increasingly probe the pressures, temperatures, and compositions expected in exoplanet interiors, many of which have no equivalent in the Solar System [@lichtenberg25]. An evolutionary model therefore has to take up each new measurement without changes to its code.

The rocky planets of the inner Solar System passed through the same magma ocean epoch, but their later evolution has largely overprinted its record [@lichtenberg23]. Strongly irradiated exoplanets, in contrast, may reside in this phase today, which opens it to direct observation. With PROTEUS we have reproduced the early divergence of Earth and Venus [@chili26] and the onset of clement surface conditions on the Hadean Earth [@vandijk26]. The interior compositions and geochemistries it covers [@nicholls24; @lichtenberg26] range from the reduced mantle of Mercury to the oxidised mantle of Mars. For exoplanets, the simulations predict the atmospheric composition, radius, and thermal emission that JWST and the observatories that follow it can measure. A measured spectrum or bulk density can thus be interpreted as a snapshot of the evolutionary history of a planet. Because material properties enter as data tables, the observable consequences of a new laboratory measurement can be evaluated at planetary scale.

# State of the field

Coupled models of a magma ocean and its atmosphere date back to @elkinstanton08 [@lebrun13; @hamano13; @schaefer16; @salvador17]. Later codes included atmospheric escape, geochemistry, or tidal heating [@barnes20; @kite20a; @krissansentotton21; @lichtenberg21a; @bower22]. More recent studies extend such models to super-Earths, sub-Neptunes, and lava worlds [@maurice24; @tang24; @carone25; @cherubim25; @farhat25; @sahu25]. However, most of these codes are not public, and each fixes one set of physical prescriptions in one code base. We initiated the CHILI intercomparison to test them on common Earth, Venus, and exoplanet cases [@lichtenberg26b; @chili26]. For Earth the models agree on the solidification time to within 4 Myr, whereas for Venus they diverge and their outgassed atmospheres differ in surface pressure and dominant species. This spread originates from the treatment of mantle dynamics (boundary-layer scaling against a resolved 1-D mantle) and of radiative transfer (gray against non-gray, with or without deep radiative layers). Volatile trapping in the solid mantle and atmospheric escape add to it, and together these choices outweigh the sensitivity to the initial volatile inventory. Such disagreement can be traced to a specific process only if components can be exchanged within otherwise identical simulations, which is the central motivation for the modular design of PROTEUS.

VPLanet [@barnes20] is open source and similarly modular, but it describes the magma ocean and the atmosphere with parameterised box models. Magrathea [@huang22] computes static interior structures, and MESA [@paxton11] evolves stars and giant planets, but neither follows the coupled evolution of a mantle and its atmosphere. To our knowledge, no other open-source framework resolves the full planet in one dimension, from the core-mantle boundary to the top of the atmosphere. The melt fraction is tracked on each node of a mantle that crystallises and contracts [@lichtenberg26]. Redox-dependent outgassing of C-H-N-O-S volatiles is coupled to the atmospheric energy balance. Escape draws on the volatile reservoir of the interior under an evolving stellar spectrum, and tidal heating, orbital evolution, and giant impacts are included. The synthetic observables also feed a Bayesian retrieval on the evolutionary model itself [@nicholls26].

# Software design

PROTEUS keeps the coupling between components separate from their physics, and each module resides in its own repository with its own tests and documentation. The framework itself contains the coupling loop, the configuration schema, the input and output layer, and tools for parameter grids on computer clusters (\autoref{fig:schematic}). In each iteration, the mantle module returns the temperature profile, melt fraction, and surface heat flux for the current atmospheric boundary condition, and the structure module updates radius and gravity. The outgassing module partitions volatiles between melt and atmosphere at the current melt mass and oxygen fugacity, and escape removes mass under the stellar spectrum. Finally, the atmosphere module solves for radiative-convective equilibrium with the new composition, and the star, tides, and accretion modules update irradiation, tidal heating, and planet mass. The time step adapts to the most rapidly changing quantity. This operator-split approach costs a longer installation, more dependencies, and data exchange across Python, Julia, Fortran, and C code. However, it allows modules to be developed and verified independently, both standalone and within the coupled system, and existing codes to be integrated rather than re-implemented. Because modules with the same role are interchangeable (SPIDER or Aragog, CALLIOPE or atmodeller, JANUS or AGNI), the sensitivity to a modelling choice can be measured within one framework. Most roles also have a simplified "dummy" module that runs in seconds, which keeps the coupled test suite fast and isolates the effect of a single component.

![Module groups of the PROTEUS framework. Interior structure: Zalmoxis [@lichtenberg26], on the PALEOS equations of state [@attia26]. Mantle energetics: Aragog [@lichtenberg26], SPIDER [@bower18], and a boundary-layer model. Atmosphere: AGNI [@nicholls25a; @nicholls25b] and JANUS [@graham21; @graham22]. Radiative transfer: SOCRATES [@manners2024fast], with measured surface reflection properties [@hammond25]. Chemistry: VULCAN kinetics [@tsai17; @tsai21] and FastChem equilibrium [@kitzmann24]. Outgassing: CALLIOPE [@bower22; @shorttle24; @nicholls24] and atmodeller [@bower25], with LavAtmos rock vapour [@vanbuchem23]. Escape: ZEPHYRUS [@postolec26a]. Star: MORS [@johnstone21]. Tides and orbit: LovePy [@hay19; @nicholls25c] and Obliqua. Accretion: Morrigan [@kimura25]. Observation and inference: petitRADTRANS [@molliere19; @nasedkin24] and an evolutionary retrieval module [@nicholls26] operate on the output and are not shown.\label{fig:schematic}](proteus_modules_schematic.png){width=95%}

Laboratory data enter as versioned external tables rather than as code, including equations of state [@attia26], melting curves, solubility laws, opacities, and stellar spectra. The tables are downloaded from Zenodo and the Open Science Framework and verified on first use, so a new measurement replaces a table and leaves the code unchanged. A new module implements the interface of its role and is registered in the configuration schema. A single validated TOML file defines each simulation, which makes every published run reproducible; the configurations of the CHILI cases and the tutorials ship with the repository.

Each pull request to PROTEUS or a module runs a fast set of unit tests, with the physics mocked, and short runs of the actual solvers. A nightly suite runs the complete set, uploads coverage, and raises the coverage threshold, which is never lowered. An automated check rejects new tests that lack an edge case, a failure path, or a physical invariant such as conservation of mass or energy, boundedness, or agreement with a reference value. Physically, the modules are validated against analytic limits and against each other where two implementations of one role exist, such as the magma ocean energy budget in SPIDER and in Aragog. The interior structure is compared with published models up to 20 Earth masses [@lichtenberg26]. The coupled evolution is benchmarked against the other codes in the CHILI intercomparison [@chili26]. Finally, the framework is tested against observations such as the sulfur dioxide on L 98-59 d [@nicholls26b] and the Hadean geological record [@vandijk26]. Test counts and coverage of every module are published on the [validation page](https://proteus-framework.org/validation/), and the [documentation](https://proteus-framework.org/PROTEUS/) covers installation, configuration, and tutorials. This paper describes PROTEUS version \textcolor{red}{XX.XX.XX}.^[Will be updated to the latest release on resubmission.]

# Research impact statement

Simulations with PROTEUS have revealed several phenomena that arise only when planetary subsystems are coupled. A self-limiting feedback between irradiation, tidal heating, and mantle rheology prolongs magma ocean epochs [@nicholls25c]. On the Hadean Earth, the tides of the young Moon may have had the same effect [@vandijk26]. Above a magma ocean, convection can shut down and leave deep radiative layers in the atmosphere [@nicholls25a]. On young planets, the balance between outgassing and escape determines whether a bare rock or a water- or carbon-rich atmosphere remains [@postolec26a]. Most rocky sub-Neptunes are expected to retain molten interiors [@calder26], and a crystallising interior contracts by about 10 per cent [@lichtenberg26]. Late in their evolution, the atmospheres of ultra-short-period super-Earths can re-inflate through redox-driven processes [@cesario26].

Predictions from PROTEUS are used in JWST observing programmes, including the lava-planet survey of @sastre26b and TOI-561 b [@postolec26b]. PROTEUS simulations can explain the sulfur dioxide detected on L 98-59 d with a permanent magma ocean rich in sulfur and hydrogen [@nicholls26b]. Photochemical models of the atmospheres outgassed in PROTEUS identify sulfur species as an observational tracer of the mantle redox state [@panagiotou26]. PROTEUS participates in the CHILI intercomparison and provided the evolutionary inputs for its comparison of static models [@lichtenberg26b; @chili26]. Individual modules are also used standalone, AGNI for lava-planet atmospheres [@nicholls25b] and atmodeller for outgassing chemistry [@bower25]. More than 20 contributors have developed the framework openly on GitHub since 2018. Releases of the framework and of its main modules are tagged, published on PyPI, and archived with a DOI on Zenodo. The Forming Worlds Lab at the University of Groningen maintains the project together with the Netherlands eScience Center.

# AI usage disclosure

Generative AI tools have assisted the development of PROTEUS since early 2026, and were used in the preparation of this paper. We use Claude Code (Anthropic; Fable, Opus, and Sonnet model families), GitHub Copilot, and Gemini (Google; Pro and Flash model families) to draft and refactor code, scaffold tests, revise and extend documentation, and copy-edit this manuscript. Human authors steer, review, edit, and validate all AI-assisted output, make all design decisions, and are responsible for the content of the software and of this paper.

# Acknowledgements

TL acknowledges support from the Netherlands eScience Center (PROTEUS project, NLESC.OEC.2023.017), the Branco Weiss Foundation, and the Alfred P. Sloan Foundation (AEThER project, G202114194). TL further acknowledges the United States National Aeronautics and Space Administration's Nexus for Exoplanet System Science research coordination network (Alien Earths project, 80NSSC21K0593). RC and OS acknowledge support from the United Kingdom Science and Technology Facilities Council (grant numbers ST/Y509139/1 and UKRI1184).

# References
