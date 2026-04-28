"""Generate the 8 PALEOS v1.1.0 validation configs (4 masses x 2 resolutions).

Mirrors today's T2.1i (1 M_E) + T2.3 (3, 5, 10 M_E) Newton runs verbatim
except for `mass_tot`, `tcmb_init`, `tcenter_init`, `path`, and `mantle_eos`.
Output dir + filename: chili_paleos_v1_1_0_<M>me_<R>res(.toml).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path("/Users/timlichtenberg/git/PROTEUS")
TEMPLATE = REPO / "output" / "chili_dry_coupled_stage2_t2_1i_newton" / "init_coupler.toml"
OUTDIR = REPO / "input" / "chili"

# (mass_tot, tcmb_init, tcenter_init) — copied from T2.1i (1 M_E) and T2.3 (3/5/10 M_E)
IC_PER_MASS = {
    1:  (7199.0, 6000.0),
    3:  (9800.0, 11100.0),
    5:  (11000.0, 13500.0),
    10: (12600.0, 16700.0),
}

EOS_PER_RES = {
    150: "PALEOS-2phase:MgSiO3",
    600: "PALEOS-2phase:MgSiO3-highres",
}


def build_config(mass: int, res: int, template_text: str) -> tuple[str, str]:
    tcmb, tcenter = IC_PER_MASS[mass]
    eos = EOS_PER_RES[res]
    name = f"chili_paleos_v1_1_0_{mass}me_{res}res"
    out = template_text
    out = out.replace(
        'path = "chili_dry_coupled_stage2_t2_1i_newton"',
        f'path = "{name}"',
    )
    out = out.replace(
        "mass_tot = 1.0",
        f"mass_tot = {float(mass)}",
    )
    out = out.replace(
        "tcmb_init = 7199.0",
        f"tcmb_init = {tcmb}",
    )
    out = out.replace(
        "tcenter_init = 6000.0",
        f"tcenter_init = {tcenter}",
    )
    out = out.replace(
        'mantle_eos = "PALEOS-2phase:MgSiO3"',
        f'mantle_eos = "{eos}"',
    )
    return name, out


def main() -> None:
    template = TEMPLATE.read_text()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mass in (1, 3, 5, 10):
        for res in (150, 600):
            name, text = build_config(mass, res, template)
            target = OUTDIR / f"{name}.toml"
            target.write_text(text)
            print(f"wrote {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
