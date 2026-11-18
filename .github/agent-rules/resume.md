# Resume with an accretion module

Read this before you change resume, `restore_accretion_state` or the state files of an accretion module.

With an accretion module selected, resume refuses a last row that records accreted rock (`M_accreted_rock > 0`) but no `n_impacts_applied`, and, when the run directory holds the resolved `impact_timeline.csv`, a counter below the number of impacts at or before the resume time, or above it unless every surplus impact is one of the next impacts after the resume time and lies in the init stage (`t <= 1 yr`) (`restore_accretion_state` in `src/proteus/accretion/wrapper.py`).
