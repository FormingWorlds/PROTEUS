# Usage

This section describes how to use PROTEUS. The framework can be run three ways:

- **Standalone**: a single forward model from one configuration file.
- **As a grid (ensemble)**: many forward models sweeping over parameters.
- **Within a retrieval framework**: PROTEUS as the forward model in a parameter inference loop.

In all cases you configure the model through a configuration file, described in the [configuration guide](config.md). If you run into problems, see the [troubleshooting](troubleshooting.md) page.

!!! tip "Quick start"
    ```console
    proteus start -c input/all_options.toml
    ```
    Results appear in `output/all_options/`. See [Running and output](usage_running.md#output-and-results) for details.

## Where to go next

| Page | What it covers |
|---|---|
| [Running and output](usage_running.md) | Launching a single run from the terminal, running on remote machines, where results are written, and archiving output. |
| [Parameter grids](usage_grids.md) | Defining and dispatching ensembles of simulations, with or without Slurm. |
| [Postprocessing and chemistry](usage_postprocessing.md) | Atmospheric chemistry with VULCAN, synthetic observations, and multi-angle thermal profiles. |
| [Bayesian inference](inference.md) | Using PROTEUS as the forward model in a Bayesian-optimisation retrieval. |

Related pages: the [configuration file](config.md) reference, [diagnosing and updating your installation](doctor.md), and the worked [tutorials](../Tutorials/quick_start_dummy.md).

## Tutorials

For guided, end-to-end walkthroughs, see the tutorials:

- [Quick start (all-dummy)](../Tutorials/quick_start_dummy.md): a fast first run with placeholder modules.
- [Earth analogue](../Tutorials/earth_analogue.md): a physically realistic Earth-like case.
- [Parameter grid sweep](../Tutorials/parameter_grid.md): building and running an ensemble.
- [Solar System CHILI intercomparison](../Tutorials/chili_intercomparison.md): reproducing the CHILI benchmark cases.

---

**See also:** [Running and output](usage_running.md) | [Parameter grids](usage_grids.md) | [Postprocessing and chemistry](usage_postprocessing.md) | [Configuration file](config.md) | [Diagnose and update](doctor.md) | [Troubleshooting](troubleshooting.md)
