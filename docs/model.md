# Model

## Overview

George Box famously put that "all models are wrong, but some are useful". We must bear this in mind when developing a tool that self-consistently models the pertinent processes of planetary evolution, which span many orders of magnitude in spatial- and time-scales. We therefore leverage a modular and hierarchical modelling approach.

PROTEUS is a modular framework for simulating the time evolution of small (exo)planets. This new modelling framework is designed to be flexible, in a sense reflecting the broad diversity of planetary conditions already discovered, with the view of being updated to incorporate additional physics as the need arises. This approach stands in contrast to common monolithic models in the literature. Furthermore, PROTEUS is free and open-source, which permits external scrutiny of its workings and ensures that conclusions eventually presented in the literature reasonably reflect the assumptions in the modelling approach. PROTEUS is directly based upon the model of Lichtenberg et al. (2021) although the code has evolved substantially from that state.

Although PROTEUS aims to treat the problem of *planetary* evolution, it must necessarily also handle external processes which act upon the planet (e.g. tidal heating). We are therefore also modelling the combined system of a planet, the relevant interactions with neighbouring planets, its orbital mechanics, and the evolution of its host star. In PROTEUS, the planet itself is conceptually sub-divided into a vaporised *atmosphere* component, which sits above an *interior* component containing a silicate mantle and metallic core. The schematic below shows a cartoon of the problem under consideration, depicting the most important components. In subdividing the system, PROTEUS acts to facilitate communication between individual software *modules* which each implement a model for a specific part of the overall system. For example: the interior module of PROTEUS simulates the time-evolution of the planet's mantle and core, their cooling, and potential solidification. Conceptually, PROTEUS modules (e.g. the interior) are 'slots' which are filled by specific implementations: the 'models' (e.g. Aragog).

<p align="center">
      <img src="assets/schematic.png" style="max-width: 70%; height: auto;"></br>
      <b>Schematic of PROTEUS components and corresponding modules.</b> </br>
</p>

We implement multiple independent models to perform the role of a given module within the PROTEUS framework. Each model can be used stand-alone, independently of the other models. Hierarchical modelling allows an inter-comparison of simple and complex models, taking advantage of the easy comprehension of the simple in order to diagnose and validate the qualitative behaviour of the complex. We emphasise that the _dummy_ modules are not designed to make quantitatively meaningful calculations of planetary evolution, but only to qualitatively capture end-member behaviours and set expectations from the physically-representative implementations.

Only the interior and star modules have an explicit notion of time-evolution. All other modules are applied at equilibrium, such that the quantities calculated by these modules are effectively updated instantaneously at each time-step. We are therefore assuming that the physical processes handled by these equilibrium modules are able to reach steady-state on time-scales shorter than the physics considered by interior and stellar evolution modules.

For further information on the model, we refer the reader to the model's bibliography below.

## Bibliography

**Works describing PROTEUS**

* Nicholls et al. (2025a) –– [doi:10.1093/mnras/stae2772](https://doi.org/10.1093/mnras/stae2772) –– [arXiv PDF](https://arxiv.org/pdf/2412.11987)
* Nicholls et al. (2024) –– [doi:10.1029/2024JE008576](https://doi.org/10.1029/2024JE008576) –– [arXiv PDF](https://arxiv.org/pdf/2411.19137)
* Lichtenberg et al. (2021) –– [doi:10.1029/2020JE006711](https://doi.org/10.1029/2020JE006711) –– [arXiv PDF](https://arxiv.org/pdf/2101.10991)

**Works depended-upon by PROTEUS**

* TBC
