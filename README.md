# Polyamide Transformer Project

This repository contains a Transformer-based molecular generation workflow for polyamide-related SMILES data.

## Main entry points

- `train.py`: compatibility wrapper for model training.
- `generate.py`: generate SMILES from a trained checkpoint.
- `eval.py`: compatibility wrapper for metrics evaluation.
- `run.py`: compatibility wrapper for the end-to-end pipeline.
- `config.py`: central project paths and default locations.

Real script implementations live under `scripts/`. `sample.py` is kept as a compatibility wrapper around `generate.py`.

## Project layout

```text
.
|-- config.py
|-- train.py
|-- generate.py
|-- run.py
|-- eval.py
|-- dataset.py
|-- script_utils.py
|-- models_storage.py
|-- interfaces.py
|-- utils.py
|-- SA.py
|-- scripts/
|   |-- train.py
|   |-- eval.py
|   `-- run.py
|-- metrics/
|   |-- NP_score/
|   |   `-- npscorer.py
|   `-- SA_score/
|       |-- metrics.py
|       |-- sascorer.py
|       `-- utils_evl.py
|-- Transformer/
|-- data/
|   |-- PolyInfo_train.csv
|   |-- PolyInfo_test.csv
|   `-- resources/
|       |-- fpscores.pkl.gz
|       |-- mcf.csv
|       `-- wehi_pains.csv
|-- checkpoints/
`-- outputs/
```

## Git tracking guidance

Track these files:

- source code under the repository root and `Transformer/`
- `data/PolyInfo_train.csv`
- `data/PolyInfo_test.csv`
- `data/resources/` files required by evaluation and SA scoring
- `README.md`, `requirements.txt`, `.gitignore`

Ignore these files:

- `checkpoints/`
- `outputs/`
- `__pycache__/`, `.idea/`, `.vscode/`
- virtual environments and Python cache files
- generated `.pt`, `.pth`, `.ckpt`, `.bin`, `.log`
- generated statistics such as `data/*_stats.npz`

## Basic usage

Train:

```bash
python train.py transformer --model_save checkpoints/transformer_model.pt --config_save checkpoints/transformer_config.pt --vocab_save checkpoints/transformer_vocab.pt --device cpu
```

Generate:

```bash
python generate.py transformer --model_load checkpoints/transformer_model.pt --config_load checkpoints/transformer_config.pt --vocab_load checkpoints/transformer_vocab.pt --gen_save outputs/generated.csv --n_samples 1000 --device cpu
```

## Notes

- `train.py`, `eval.py`, and `run.py` remain available at the repository root as compatibility wrappers.
- The current environment used for reorganization did not have project dependencies installed, so runtime imports requiring `numpy`, `torch`, `rdkit`, and related packages could not be fully executed during verification.
