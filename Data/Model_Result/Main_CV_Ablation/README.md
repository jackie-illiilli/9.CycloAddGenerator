# Main CV and ablation result caches

`notebooks/03_modeling/02_main_cv_ablation.ipynb` is the primary random-row five-fold
CV and descriptor-ablation notebook. All retraining switches default to False.

## Separate stages

- `main_cv/`: the complete PhysOrg + SVO + broad-data predicted-reaction-energy model.
  Set only `RETRAIN_MAIN_CV = True` to run this stage first.
- `ablation/`: PhysOrg, PhysOrg + SVO, and active-only predicted-reaction-energy
  variants. Later set `RETRAIN_MAIN_CV = False` and `RETRAIN_ABLATION = True`.
  This stage reuses the complete model's saved results instead of fitting it again.
- `reaction_energy/`: the separate first-stage reaction-free-energy experiments,
  controlled by `RETRAIN_REACTION_ENERGY`.
- Full-data feature importance is controlled by `RETRAIN_FEATURE_IMPORTANCE` and
  saved under `main_cv/`. It does not represent held-out performance.

Each CV stage saves `folds.csv`, `summary.csv`, and `run_settings.json`. All three
CV stages also save `oof_predictions.csv` with reaction IDs, aligned row index,
variant/target or reaction-energy dataset, fold, observed values, and out-of-fold predictions. Their
run settings record the common split seed, model parameters, and input hashes.
The notebook checks matching run settings when comparing two newly trained stages.
All four D/I/activation targets use the same folds for a given feature variant.

The full-model scatter cell explicitly calls
`model_train.plot_scatter_with_metrics`, passing predicted values first and
observed values second, as in legacy notebook cells 27-30. Per-target limits and
ticks are configured in `main_plot_settings`; interaction values are negated on
both axes. Printed metrics still use observed values as the reference. Plots are saved under
`main_cv/plots/`. The plotting function is imported only when OOF values exist, so
reading summary-only caches does not require the full training/plotting environment.

The thermodynamic prediction section also calls
`model_train.plot_scatter_with_metrics`, matching legacy notebook cells 72-76.
Its four reaction-free-energy plots use the red density style and are saved in
`reaction_energy/plots/` as `rxn_physorg_svo.png`, `active_physorg.png`,
`active_physorg_svo.png`, and `active_broad_transfer.png`. The reaction-energy
training section saves the underlying observed/predicted `deltaG` values to
`reaction_energy/oof_predictions.csv`. After one training run, set
`RETRAIN_REACTION_ENERGY = False` to plot from this CSV; use
`PLOT_REACTION_ENERGY = False` to skip the plots.

## Initial saved data

The root-level CSVs are retained as historical combined snapshots. The current
stage folders were initialized by splitting these existing tables, without
retraining: 4 complete-model summary rows, 12 ablation summary rows, 4
reaction-energy summary rows, and 40 top-10 feature-importance rows. These values
were recovered from saved HTML tables in the executed notebook and rounded to
six decimals by its original display. No fold metrics or predictions were invented.
A deliberate training run replaces the relevant stage caches with full-precision
results. Existing OOF plots cannot be reconstructed from summary metrics alone.
