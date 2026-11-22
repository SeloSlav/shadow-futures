# Replication: Measuring Comparison Opportunities in Adaptive Markets

The finite-history simulations use synthetic allocations. The separate MusicLab analysis uses recorded downloads from Experiment 1 and published condition-level listening totals.
Python 3 and NumPy are the only dependencies; recorded runs used NumPy 2.3.5.
Run the scripts from this directory with a Python executable:

```
python comparison_experiments.py
python calibration_experiments.py
python validate_experiments.py
```

The first command writes `comparison_results.json`. It executes the 18-condition
reinforcement/separation/reset study, four response/persistence stresses, complete
short-history enumeration, and the specified affinity-cost design calculations.
Three seeds each generate 2,000 records per parameter law. Horizons reuse simulated
paths and configurations share recipient random numbers, so comparisons between
conditions must not be treated as independent estimates. The audit uses a separate
seed stream with offset 900,000. The full run took approximately five minutes on
the authoring machine; runtime depends on hardware.

The second command writes `calibration_results.json` and `calibration_banks.npz`.
It uses three independent calibration banks with 999 null trajectories at each
of 20 joint candidate points. Validation uses 1,000 fresh histories per truth and
bank, for single histories and genuinely independent length-sixteen markets.
Null and validation streams are separate. Conditional validation standard errors
aggregate within-bank Bernoulli variances; bank-specific rates, bank ranges, and
the standard error over three bank means describe additional bank variability.
Wilson intervals are descriptive, conditional on the realized calibration banks.
Finite-grid rank coverage is unconditional over both calibration and observation
randomness, conditional on the declared initial state and design.

The third command reads existing results and the sibling article `../paper2.json`
without changing simulation outputs, then writes `validation.json`. Its independent
weight-form history enumerator and chronological likelihood calculation check the
kernel, budget, Fisher-information components, finite KL bound, affinity bound,
reset arithmetic, response identity, null-rank ties, calibration banks, and all
four manuscript tables. It has no DOCX, rendering, browser, or desktop dependency.

The sibling `paper2.json` is an archived input for validation of Tables 1–4. It predates Exhibit A and the MusicLab section. If the replication supplement is distributed without its
sibling article JSON, retain that JSON beside this directory or adjust the
validator's article-source path. Runtime fields change on reruns; substantive
numeric outputs are deterministic for the specified seeds and numerical library.
No computation here certifies almost-sure infinite-budget finiteness, continuous
parameter confidence coverage, real-market identification, or causal invariance.

## One continuing market with the same limiting share

Run `python same_share.py` to reproduce Exhibit A. The nine-point joint grid, fixed seeds, nested horizons, 999 calibration histories per null and 2,000 independent validation histories per truth are in the script. The output includes classification errors, conditional validation standard errors, projected-set coverage, cardinality and mean span. Sets refer to the stated finite grid. The rank test includes ties.

## MusicLab Experiment 1

From this directory, run `python musiclab.py ../musiclab`. The ZIP includes nine original anonymized download-event files, renamed world1.txt through world9.txt. The script checks file structure and chronology and records the SHA-256 hashes. The first eight worlds displayed social information. World 9 supplied the hidden-count baseline. All 48 song columns are kept in file order. Zero-mark rows are counted and excluded. Files are chronological, and ties retain release order. No listening event is inferred from an absent download. The separate listening totals come from the authors’ Table S2.

Dataset credit: Matthew J. Salganik, *Success and Failure in Cultural Markets*, Princeton University, [dataset](https://doi.org/10.34770/y56c-ym90), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The nine source files were distributed by [University of Michigan CSS Labs at commit 86dd114](https://github.com/UM-CSS/CSSLabs-Experiments/tree/86dd11432be87f0674da169571d372d19814da32/external/CM/musiclab_data). Original filenames are dynamics_downloads_download_w1_v1.txt through dynamics_downloads_download_w9_v1.txt. Only the filenames were changed for this package.

Experiment and separate listening counts: M. J. Salganik, P. S. Dodds and D. J. Watts, *Science* 311 (2006), 854–856, [article](https://doi.org/10.1126/science.1121066) and [Table S2](https://www.princeton.edu/~mjs3/salganik_dodds_watts06_som.pdf). The held-out scores evaluate prediction of song identity conditional on a download. They do not identify the effect of popularity on listening or conversion.
