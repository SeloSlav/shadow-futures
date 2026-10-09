Shadow Futures Paper 2 replication materials

All observations in this package are synthetic. The benchmark is an exact
two-recipient reinforced allocation experiment, with beta equal to -0.5 or
+0.5, not a field dataset. The model, seeds, sample sizes, and uncertainty
interpretation are given in Paper 2 and paper2_results.json.

Run in a Python environment containing the versions in requirements.txt:
  python paper2_benchmark.py
  python paper2_validate.py
  python paper2_plot.py

The benchmark writes paper2_results.json beside the script. It uses seeds
20261009 and 20261010, NumPy 2.3.5, and 5,000 datasets under each parameter
for each design. A rerun replaces that results file; preserve the distributed
copy first if you want to compare results. runtime_seconds naturally varies.

The standalone validator uses an independent log-odds expression and checks
all 65,536 length-16 histories, exact likelihood ratios, reset arithmetic,
and the recorded exact classification risk. It does not require manuscript
render directories or overwrite the benchmark results.

The optional figure is generated from the recorded outputs. The plot script
also creates a PDF as its rendering intermediate. The manuscript contains a
native Word results table. The figure is supplementary scientific material.

Monte Carlo uncertainty is distinct from model uncertainty and economic
population uncertainty. Zero observed errors do not establish zero true
risk. A finite computation does not prove an infinite-horizon theorem.
