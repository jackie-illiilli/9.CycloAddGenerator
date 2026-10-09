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

The notebooks use the saved energy tables in Data/DFT_Result and descriptor maps in Data/Descriptors. Configure paths and random seeds in each notebook before running it. ChemProp workflows require ChemProp and PyTorch Lightning in addition to the packages listed in requirements.txt.

The nested TPE search in 04_fingerprint_benchmark.ipynb requires Hyperopt, which is not pinned in requirements.txt (`pip install hyperopt`). By default, it tunes the PhysOrg + SVO representation with five-fold inner and outer validation; edit the notebook's selection settings to change the scope. This search is compute-intensive.

## Reproduction limits

The supplied filtered descriptors support the main modeling workflows. The complete conformer collection and some pretrained molecular representations used by fingerprint comparisons are external and are not included. Consequently, the full representation benchmark cannot be rebuilt from this repository alone. Some candidate and SVO analyses also require external structure or calculation files; the notebook setup cells identify those inputs. Matching published outputs additionally requires the package versions and prepared input tables used for the original analysis.
