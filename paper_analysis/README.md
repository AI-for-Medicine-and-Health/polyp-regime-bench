# Paper analysis scripts

Scripts that produce the rank-agreement analysis (Kendall's tau), Fig. 2, Fig. 5 and the
main-text/appendix tables of the manuscript from the exported result files.

Inputs (from the original `generated/` folder):
- `generated/results_data_ap001_operating020_vote001.json`
- `generated/table2_aug10x.tex`, `generated/table3_conditions.tex`,
  `generated/table5_inference_cost.tex`, `generated/table4_rtdetrx_errors.tex`,
  `generated/table1_datasets.tex`, `generated/appendix_table_s1.tex` ... `s3.tex`

Run from the directory that contains `generated/`:

    mkdir -p figs tables
    python analyze.py   # writes analysis.json (rankings, tau, ensemble summaries)
    python figs.py      # writes figs/fig2_rank_by_regime.pdf, figs/fig3_ensemble_gain_cost.pdf
    python tables.py    # writes tables/*.tex

Note: figs.py names the ensemble figure fig3_ensemble_gain_cost.pdf; in the manuscript it is
Fig. 5 (figs/fig5_ensemble_gain_cost.pdf). Requires numpy, scipy and matplotlib.
