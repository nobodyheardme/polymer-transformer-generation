# Polyamide Transformer Project

This repository contains a Transformer-based molecular generation workflow for polyamide-related SMILES data.

## Main entry points

- `train.py`: train a model.
- `generate.py`: generate SMILES from a trained checkpoint.
- `config.py`: central project paths and default locations.

`sample.py` is kept as a compatibility wrapper around `generate.py`.

## Project layout

```text
.
|-- config.py
|-- train.py
|-- generate.py
|-- run.py
|-- eval.py
|-- dataset.py
|-- metrics.py
|-- script_utils.py
|-- models_storage.py
|-- interfaces.py
|-- utils.py
|-- utils_evl.py
|-- sascorer.py
|-- npscorer.py
|-- SA.py
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

- `run.py` remains available as a higher-level pipeline script.
- The current environment used for reorganization did not have project dependencies installed, so runtime imports requiring `numpy`, `torch`, `rdkit`, and related packages could not be fully executed during verification.
