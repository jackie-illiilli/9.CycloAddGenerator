# Modeling workflow

These notebooks cover model fitting, representation benchmarks, generalization tests, and candidate comparisons. Run them from the repository root or use the setup cells to resolve repository-relative paths.

| Notebook | Task |
| --- | --- |
| 01_energy_landscape.ipynb | Inspect the energy and descriptor inputs and generate diagnostic plots |
| 02_main_cv_ablation.ipynb | Fit the primary models and compare descriptor combinations |
| 03_fingerprint_generation.ipynb | Prepare alternative molecular representations |
| 04_fingerprint_benchmark.ipynb | Compare representations and models; includes fixed-parameter and nested TPE benchmarks |
| 05_generalization_tests.ipynb | Evaluate grouped and reactant-holdout splits |
| 06_candidate_dft_comparison.ipynb | Compare model predictions with selected DFT records |
| 07_svo_variant_generation.ipynb | Generate the alternative SVO descriptor map |
| 08_svo_variant_benchmark.ipynb | Evaluate the SVO map against the primary representation |
| 09_training_fraction_seeds.ipynb | Assess training-fraction and random-seed effects |
| 10_ChemProp.ipynb | Run the ChemProp reaction-graph baseline |

The notebooks use the saved energy tables in Data/DFT_Result and descriptor maps in Data/Descriptors. Configure paths and random seeds in each notebook before running it. The default environment is `main_py3_12` (Python 3.12), with dependencies in [requirements.txt](../../requirements.txt), including Hyperopt, ChemProp, PyTorch, and Lightning.

**Run and test `03_fingerprint_generation.ipynb` only in `main` (Python 3.8).**
Its DScribe-based descriptor code uses the existing legacy environment, including
DScribe 1.2.1, ASE 3.22.0, Mordred 1.2.0, and NumPy 1.21.0. Select the `main`
Python 3.8 notebook kernel and confirm `sys.executable` and `sys.version` before
execution. Do not use `main_py3_12` for this notebook or install the root
requirements into `main`. See the root [environment instructions](../../README.md#environment)
for kernel registration and the current Python 3.12 environment's dependency
conflicts. External conformers and pretrained maps are still needed for the
optional representations.

`02_main_cv_ablation.ipynb` defaults to reading cached CSV results from
`Data/Model_Result/Main_CV_Ablation` without loading descriptors or training models.
Train the complete PhysOrg + SVO + broad-data predicted-reaction-energy model
first with `RETRAIN_MAIN_CV = True`; its metrics and OOF predictions go to `main_cv/`.
Then set `RETRAIN_MAIN_CV = False` and `RETRAIN_ABLATION = True` to run only the three
reduced variants and save them to `ablation/`, reusing the full-model results.
Both stages use identical five-fold splits and record their run settings and input
hashes. Reaction-energy CV and feature importance have separate switches,
`RETRAIN_REACTION_ENERGY` and `RETRAIN_FEATURE_IMPORTANCE`.
The full-model plot cell directly calls `model_train.plot_scatter_with_metrics`
on saved OOF predictions. The initial caches contain summary tables and top 10
features recovered from saved outputs at six-decimal precision; individual OOF
predictions were not saved in that run, so these plots require the CSV produced
by a subsequent full-model training run.
The thermodynamic reaction-free-energy section also saves its own OOF prediction
CSV and provides the four legacy density plots through the same plotting function.
These use `PLOT_REACTION_ENERGY`, read `reaction_energy/oof_predictions.csv`, and
save figures under `reaction_energy/plots/` independently of main-model CV.

The nested TPE search in 04_fingerprint_benchmark.ipynb requires Hyperopt, which is pinned in requirements.txt. By default, it tunes the PhysOrg + SVO representation with five-fold inner and outer validation; edit the notebook's selection settings to change the scope. This search is compute-intensive.

## Reproduction limits

The supplied filtered descriptors support the main modeling workflows. The complete conformer collection and some pretrained molecular representations used by fingerprint comparisons are external and are not included. Consequently, the full representation benchmark cannot be rebuilt from this repository alone. Some candidate and SVO analyses also require external structure or calculation files; the notebook setup cells identify those inputs. Matching published outputs additionally requires the package versions and prepared input tables used for the original analysis.
