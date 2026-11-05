# AFAG: Adaptive Feature Selection with Adaptive Graph Learning

This repository contains the code and tuned parameter files for the manuscript **“Advanced Feature Selection with Adaptive Graph Learning in Unsupervised Setting”** (Manuscript ID: Math-30001).

The experiments compare AFAG with eight baseline and comparison methods on six benchmark datasets. The repository also includes scripts for parameter tuning, ablation analysis, and convergence diagnostics.

## Repository Contents

| File or directory | Description |
|---|---|
| `Running.py` | Runs the main comparison experiments and saves results and plots. |
| `algorithms.py` | Implements conventional feature selection methods and clustering evaluation. |
| `algorithms_adaptive.py` | Implements AFAG, UDLA, CNAFS, SOGFS, and shared adaptive-graph utilities. |
| `data_loader.py` | Loads and prepares the benchmark datasets. |
| `tune_afag_wide.py` | Searches AFAG parameter settings. |
| `tune_udla_wide.py` | Searches UDLA parameter settings. |
| `tune_comparison_methods.py` | Searches parameter settings for the comparison methods. |
| `ablation.py` | Runs the ablation analysis. |
| `convergence_and_ki.py` | Produces convergence and `k_i` diagnostic analyses. |
| `tuning_afag_wide/` | Stores AFAG parameter files and tuning logs, organized by dataset. |
| `tuning_udla_wide/` | Stores UDLA parameter files and tuning logs, organized by dataset. |
| `tuning_comparison_methods/` | Stores comparison-method parameter files and tuning logs, organized by dataset. |
| `dataset/` | Local directory for the benchmark data files; the data files are not included in this repository. |

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- scikit-learn
- Matplotlib

Install the dependencies with:

```bash
python -m pip install numpy scipy scikit-learn matplotlib
```

For exact reproduction, use the package versions from the environment used to generate the reported results.

## Datasets

The datasets are not included in this repository. Download the MATLAB files from the links below and place them in a local `dataset/` directory at the repository root.

| Dataset | Samples | Features | Classes | Download |
|---|---:|---:|---:|---|
| warpAR10P | 130 | 2,400 | 10 | [warpAR10P.mat](https://jundongl.github.io/scikit-feature/files/datasets/warpAR10P.mat) |
| ORL | 400 | 1,024 | 40 | [ORL.mat](https://jundongl.github.io/scikit-feature/files/datasets/ORL.mat) |
| Yale | 165 | 1,024 | 15 | [Yale.mat](https://jundongl.github.io/scikit-feature/files/datasets/Yale.mat) |
| CLL-SUB-111 | 111 | 11,340 | 3 | [CLL_SUB_111.mat](https://jundongl.github.io/scikit-feature/files/datasets/CLL_SUB_111.mat) |
| TOX-171 | 171 | 5,748 | 4 | [TOX_171.mat](https://jundongl.github.io/scikit-feature/files/datasets/TOX_171.mat) |
| GLIOMA | 50 | 4,434 | 4 | [GLIOMA.mat](https://jundongl.github.io/scikit-feature/files/datasets/GLIOMA.mat) |

Expected local dataset layout:

```text
dataset/
├── warpAR10P.mat
├── ORL.mat
├── Yale.mat
├── CLL_SUB_111.mat
├── TOX_171.mat
└── GLIOMA.mat
```

Please follow the dataset providers’ terms when downloading and using the data. Do not redistribute data files unless their terms permit it.

## Tuned Parameter Files

The main experiment loads the saved parameter files from these paths:

```text
tuning_afag_wide/<dataset>/best_params.json
tuning_udla_wide/<dataset>/best_params.json
tuning_comparison_methods/<dataset>/best_params.json
```

Each tuning directory should contain a subdirectory for each of the six datasets. The comparison-method JSON files contain entries for JELSR, NDFS, SOGFS, and CNAFS. Keep the directory names, dataset names, and JSON structure unchanged because `Running.py` uses them to load the parameters. The tuning scripts can be used to search for parameters again; the saved JSON files allow the main experiment to use the recorded settings directly.

## Running the Main Experiments

Run commands from the repository root after installing the requirements, downloading the datasets, and placing them in `dataset/`.

Run a single dataset:

```bash
python Running.py warpAR10P
```

Run all six datasets:

```bash
python Running.py --all
```

The main runner uses the random seed and preprocessing settings specified in `Running.py`. It writes per-dataset results, configuration files, and plots to the `results/` directory. Running the script without a dataset argument displays the parameter settings and usage information.

## Additional Analysis Scripts

- `tune_afag_wide.py`, `tune_udla_wide.py`, and `tune_comparison_methods.py` perform parameter searches for their respective methods.
- `ablation.py` runs the ablation analysis.
- `convergence_and_ki.py` generates convergence and `k_i` diagnostics.

These analyses may require substantial computation. Refer to the options and configuration in each script before running it.

## Reproducibility Notes

- The main experiment configuration uses random seed 52 and 20 clustering evaluation runs.
- The methods, preprocessing settings, feature-count grid, and saved parameter provenance are defined or recorded by the scripts.
- To reproduce reported results, use the same dataset files, parameter JSON files, and package versions as the original experiment environment.
- Tuning logs, when present, are provided as records of the parameter search; `Running.py` loads `best_params.json` to configure the main comparison.

## License and Attribution

See the [`LICENSE`](LICENSE) file for the license applying to this repository. Please also cite the AFAG manuscript and the original publications for the comparison methods. Preserve any source-code attribution and license notices included with third-party implementations.
