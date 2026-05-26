# Transformer-based Polymer Molecular Generation

A beginner-friendly and reusable Transformer-based framework for polymer SMILES generation and AI-driven molecular design learning.

This repository provides a simplified end-to-end workflow for applying Transformer/GPT-style sequence models to polymer molecular generation. It is designed for students and beginners who want to understand how machine learning-based molecular design code is organized, how SMILES strings can be modeled as sequences, and how generated molecules can be evaluated after generation.

This project is not intended to claim a new state-of-the-art generative model. Instead, it focuses on building a readable, modifiable, and reusable learning framework for Transformer-based molecular generation.

## Project Motivation

Machine learning-based molecular design has become an important research direction in chemistry, materials science, polymer informatics, and AI-driven molecular discovery. Among different deep learning architectures, Transformer models have become one of the most influential model families because of their strong sequence modeling ability and flexibility.

However, for beginners, it is often difficult to connect the following parts together:

1. how molecular structures are represented as SMILES strings;
2. how a Transformer model learns sequence patterns;
3. how a trained model generates new molecular strings;
4. how generated molecules are checked and evaluated;
5. how the whole molecular generation pipeline is organized in code.

This project was created to bridge that gap.

Existing molecular generation benchmark repositories provide useful baselines and evaluation tools, but a simple Transformer-based polymer SMILES generation workflow is not always presented in a beginner-friendly and easy-to-modify form. Therefore, this repository reorganizes the molecular generation pipeline into a clear framework that beginners can run, understand, modify, and extend.

## Main Idea

This repository treats polymer SMILES generation as a character-level sequence modeling task.

The general workflow is:

```text
Polymer SMILES dataset
        ↓
Vocabulary construction
        ↓
Sequence encoding
        ↓
Transformer/GPT-style model training
        ↓
SMILES generation
        ↓
Validity checking
        ↓
Molecular evaluation
        ↓
Generated molecule output
```

The Transformer model is the default model used in this project. At the same time, the outer workflow is designed to be reusable. Users can keep the same data processing, training, generation, filtering, and evaluation pipeline while replacing the model architecture.

## Main Features

- Transformer/GPT-style implementation for polymer SMILES generation
- Character-level vocabulary construction for SMILES strings
- End-to-end training and generation workflow
- Temperature and top-k sampling for molecular generation
- Molecular validity checking using RDKit
- SA score and NP score evaluation utilities
- Reorganized metric modules for clearer project structure
- Beginner-friendly code organization
- Reusable framework design that allows model replacement

## Preliminary Generation Performance

The Transformer/GPT-style model has been tested on a polymer SMILES dataset. In a preliminary generation experiment, the model generated 30,000 polymer SMILES strings, among which 21,335 were valid after RDKit checking.

```text
Total generated SMILES: 30,000
Valid generated SMILES: 21,335
Validity ratio: 71.1%
```

This result suggests that the Transformer-based sequence model can effectively learn polymer SMILES syntax and generate a large number of valid molecular strings under the current training and sampling settings.

The result is encouraging compared with my own experiments using traditional molecular generative models. However, it should be regarded as a preliminary result rather than a strict benchmark comparison. A fair comparison with models from existing molecular generation benchmarks requires the same dataset, train/test split, generation number, sampling strategy, evaluation metrics, and post-processing rules.

## Why This Repository May Be Useful

This repository may be useful for beginners who want to:

- understand how Transformer models can be used for SMILES generation;
- learn the connection between language modeling and molecular generation;
- run a complete polymer molecular generation pipeline;
- modify the model architecture or sampling strategy;
- evaluate generated molecules using common molecular design metrics;
- build a starting point for more advanced AI-based molecular design projects.

The code is intentionally kept simple and modular. In most cases, users can keep the outer pipeline unchanged and only modify or replace the model definition when testing a different generative architecture.

## Reusable Code Framework

This repository is designed not only as a Transformer implementation, but also as a reusable framework for machine learning-based molecular generation.

The outer workflow is model-independent and can be reused for different generative models. Users can keep the same data processing, vocabulary construction, training loop, molecule generation, validity filtering, and evaluation pipeline while replacing only the model architecture.

For example, the Transformer model in this project can be replaced by other sequence-based generative models, such as:

- RNN
- GRU
- LSTM
- Transformer variants
- other customized neural network architectures

as long as the new model follows the same input-output format required by the training and generation scripts.

This design makes the repository useful as a starting template for beginners who want to understand how molecular generation code is organized before building more advanced models.

## What Can Be Reused

The following parts of this repository are designed to be reusable:

- data loading and preprocessing
- SMILES vocabulary construction
- sequence encoding and decoding
- training and validation workflow
- molecule generation script
- temperature and top-k sampling strategy
- molecular validity checking
- SA score, NP score, and other evaluation utilities
- generated molecule filtering and result saving

In most cases, users only need to modify or replace the model definition if they want to test a different generative architecture.

## Project Structure

```text
.
├── README.md
├── train.py
├── run.py
├── eval.py
├── SA.py
├── model.py
├── utils.py
├── config.py
├── metrics/
│   ├── __init__.py
│   ├── NP_score/
│   │   ├── __init__.py
│   │   └── npscorer.py
│   └── SA_score/
│       ├── __init__.py
│       ├── metrics.py
│       ├── sascorer.py
│       └── utils_evl.py
└── scripts/
```

The root-level scripts such as `train.py`, `eval.py`, and `run.py` are kept as convenient entry points. The `metrics/` directory reorganizes molecular evaluation utilities into clearer submodules. The `examples/` directory contains sample generated results and example trained-model artifacts for quick inspection and reuse.

## Examples

The `examples/` directory provides ready-to-check sample files from the Transformer workflow, including:

- generated polymer SMILES CSV files;
- filtered result files such as `valid_novel_SA` outputs;
- example checkpoint, config, and vocabulary files in `.pt` format.

These files are useful if you want to inspect the output format or test downstream analysis steps before running training and generation yourself.

## Model Replacement Guide

The project is organized so that users can replace the model while keeping the rest of the pipeline.

A new model should generally support:

1. tokenized SMILES input;
2. sequence prediction over the vocabulary;
3. loss calculation during training;
4. autoregressive generation during sampling.

The default Transformer/GPT-style model can therefore be used as a reference implementation. If another model follows the same basic input-output format, the existing training, generation, and evaluation scripts can be reused with minimal modification.

This makes the repository suitable for comparing different molecular generative models under the same outer workflow.

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n polymer_transformer python=3.9
conda activate polymer_transformer
```

Install common dependencies:

```bash
pip install torch pandas numpy tqdm rdkit
```

Depending on your local environment, RDKit may also be installed through conda:

```bash
conda install -c conda-forge rdkit
```

## Usage

### 1. Prepare the dataset

Prepare a CSV file containing polymer SMILES strings. The dataset should contain a column for SMILES strings, for example:

```text
smiles
*CCCCC(=O)NCCCNC(=O)c1ccc(C(=O)N*)cc1
...
```

The exact input path and column name can be configured in the corresponding configuration or script files.

### 2. Train the model

```bash
python train.py
```

The training script reads the polymer SMILES dataset, builds the vocabulary, encodes the sequences, and trains the Transformer/GPT-style model.

### 3. Generate polymer SMILES

```bash
python run.py
```

The generation script samples new polymer SMILES strings from the trained model. Sampling parameters such as temperature and top-k can be adjusted to control generation diversity and validity.

### 4. Evaluate generated molecules

```bash
python eval.py
```

The evaluation script checks generated molecules and calculates molecular evaluation metrics such as validity, SA score, and NP score.

## Sampling Parameters

During generation, two important parameters are commonly used.

### Temperature

Temperature controls the randomness of sampling.

A lower temperature makes the model more conservative and more likely to generate high-probability tokens. A higher temperature increases diversity but may also increase the chance of invalid SMILES.

### Top-k Sampling

Top-k sampling restricts the next-token sampling space to the top k most likely tokens.

This can reduce unreasonable token choices while still allowing diversity in generated molecules.

## Relation to Existing Molecular Generation Benchmarks

This project was inspired by existing molecular generation benchmarks, especially MOSES and Polymer-Generative-Models-Benchmark.

Polymer-Generative-Models-Benchmark provides useful references for polymer generative modeling workflows and benchmark-style evaluation. Compared with that repository, this project focuses on a simplified Transformer/GPT-style implementation for polymer SMILES generation and provides a reusable code framework for beginners.

In my preliminary experiments, the Transformer-based workflow showed promising generation performance on polymer SMILES, especially in terms of RDKit validity. This suggests that Transformer-based sequence modeling may be a useful complementary baseline for polymer molecular generation.

A rigorous comparison with the models in existing benchmark repositories would require reproducing all models under the same dataset, train/test split, generation number, sampling strategy, evaluation metrics, and post-processing rules.

Therefore, this repository should be understood as a complementary learning resource rather than a replacement for existing benchmark projects.

## Acknowledgements and Third-party Code

This project builds on several excellent open-source projects.

### minGPT

The Transformer/GPT-style model implementation was adapted from Andrej Karpathy's minGPT:

- https://github.com/karpathy/minGPT

minGPT provides a minimal and educational PyTorch implementation of GPT. In this project, the GPT-style sequence modeling idea was modified and reorganized for character-level polymer SMILES generation.

### MOSES

Some molecular evaluation utilities were adapted from the MOSES benchmark:

- https://github.com/molecularsets/moses

MOSES provides widely used tools and metrics for molecular generation benchmarking. In this repository, related scoring and evaluation utilities are reorganized under the `metrics/` directory.

### Polymer-Generative-Models-Benchmark

This project was also inspired by Polymer-Generative-Models-Benchmark:

- https://github.com/ytl0410/Polymer-Generative-Models-Benchmark

That repository provides useful references for polymer molecular generation workflows and benchmark-style evaluation. This project focuses on a simplified Transformer-based workflow and reusable code framework for beginners.

The main contribution of this repository is to reorganize these ideas into a beginner-friendly, end-to-end Transformer workflow for polymer molecular generation. The code structure, training scripts, generation scripts, and evaluation workflow were simplified and modified to make the project easier to understand, reproduce, and extend.

All third-party code remains credited to its original authors. Any modifications, simplifications, or mistakes in this repository are my own.

## Notes on Code Reuse

This repository contains code adapted from or inspired by third-party open-source projects. Users should check the original repositories and their licenses before reusing substantial portions of the code.

Third-party components retain their original licenses and copyrights.

## Limitations

This project is mainly designed for learning and research practice.

Current limitations include:

- The model is a simplified Transformer/GPT-style implementation rather than a fully optimized molecular generation model.
- Generated SMILES validity depends on dataset quality, model training, and sampling parameters.
- Molecular validity does not guarantee synthetic feasibility or real experimental accessibility.
- Polymer SMILES generation may require additional domain-specific filtering before experimental use.
- The current framework is focused on sequence-based molecular generation and does not directly model 3D molecular structures or reaction feasibility.
- The preliminary performance result is not a strict benchmark comparison against all existing molecular generation models.

## Future Work

Possible extensions include:

- adding more generative models under the same framework;
- improving SMILES tokenization;
- adding SELFIES representation;
- adding property prediction models;
- adding reinforcement learning or goal-directed generation;
- adding more polymer-specific evaluation metrics;
- improving molecular filtering and post-processing;
- building a cleaner model interface for easier architecture replacement;
- performing strict benchmark comparisons under unified datasets and evaluation protocols.

## License

This repository is for academic learning and research purposes.

Third-party components retain their original licenses and copyrights. Please refer to the original repositories for their license terms:

- minGPT: https://github.com/karpathy/minGPT
- MOSES: https://github.com/molecularsets/moses
- Polymer-Generative-Models-Benchmark: https://github.com/ytl0410/Polymer-Generative-Models-Benchmark

If you use or modify this repository, please also preserve the corresponding third-party acknowledgements.
