# AlphaVacc

[![DOI](https://zenodo.org/badge/979973636.svg)](https://doi.org/10.5281/zenodo.17004419)

A Monte Carlo Tree Search (MCTS)–driven deep learning framework for tumor immunotherapy vaccine design. AlphaVacc intelligently mutates antigen peptide sequences—taking into account patient‑specific or randomized starting peptides and HLA alleles—to build an expanded library of high‑immunogenicity vaccine candidates.

---

## Table of Contents

- [AlphaVacc](#alphavacc)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Training (Supervised Fine‑Tuning)](#training-supervised-finetuning)
    - [Prediction (Peptide Generation)](#prediction-peptide-generation)
  - [Argument Reference](#argument-reference)
  - [Logging \& Outputs](#logging--outputs)
  - [License](#license)

---

## Features

* **MCTS‑based peptide mutation** for guided exploration of high‑immunogenicity sequences.
* **Neural network coach** that iteratively learns from self‑play.
* **Patient‑specific or random starting peptides**, with support for HLA allele targets via IEDB data.
* **Easy-to-use CLI**: separate scripts for training (`main.py`) and sequence generation (`predict.py`).

---

## Prerequisites

* Python 3.9+
* anaconda
* `coloredlogs`
* Other dependencies: see `environment.yml` 

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/AlphaVacc.git
   cd AlphaVacc
   ```

2. **Install dependencies**

   ```bash
   conda env create -f environment.yml
   conda activate AI
   ```

3. **Prepare data**

   * Place your IEDB target file (`IEDB-target-9res.txt`) under `data/IEDB/`.

---

## Directory Structure

```
AlphaVacc/
├── Coach.py
├── NNetwrapper.py
├── Search.py
├── peptideMutGame.py
├── peptideMutLogic.py
├── peptideMutNNet.py
├── ...
├── utils.py
├── main.py
├── predict.py
├── data/
│   └── IEDB/
│       └── IEDB-target-9res.txt
├── temp/                   # checkpoint directory
├── Predict_data/           # output directory for predictions
└── environment.yml 
```

---

## Configuration

Both `main.py` and `predict.py` read a `dotdict` named `args` containing hyperparameters and paths. Major fields include:

* `pep_length` (int): peptide length (default 9)
* `res_type` (int): number of residue types (20)
* **MCTS settings:**

  * `numIters`: total training iterations
  * `numEps`: self‑play games per iteration
  * `numMCTSSims`: MCTS simulations per move
  * `cpuct`, `tempThreshold`, `updateThreshold`
* **Checkpoint & model loading:**

  * `checkpoint` (path)
  * `load_model` (bool)
  * `load_folder_file`: tuple (folder, filename)
* **IEDB data:**

  * `IEDBdir`, `IEDBtargetdatabase`
* **Mutation & scoring:**

  * `mutation_rate`, `half_life`, `T_init`

See the [Argument Reference](#argument-reference) below for the full list.

---

## Usage

### Training (Supervised Fine‑Tuning)

```bash
python main.py
```

This will:

1. Initialize logging (`coloredlogs` at DEBUG).
2. Rotate old record files, create new `Startrecord-YYYYMMDDHHMM.txt` and `Manualrecord-YYYYMMDDHHMM.txt`.
3. Load IEDB targets.
4. Instantiate the game (`peptideMutGame`) and neural net (`NNetWrapper`).
5. Optionally load a checkpoint (`--load_model`).
6. Start the learning loop via `Coach.learn()`.

You can modify hyperparameters directly in `main.py`’s `args`, or extend it to accept CLI flags.

---

### Prediction (Peptide Generation)

```bash
python predict.py <checkpoint_filename>
```

Example:

```bash
python predict.py checkpoint_11.pth.tar
```

This script:

1. Rotates old update files in `Predict_data/` and creates a new `update-<checkpoint>-<startpeptide>-YYYYMMDDHHMM.txt`.
2. Loads the trained model checkpoint.
3. Runs `Search.predict_usingNN(...)` for `optimizationSTEP` iterations.
4. Appends each generated peptide sequence to the update file, marking win/loss.

---

## Argument Reference

Below is a non‑exhaustive list of `args` fields shared by both scripts:

| Argument                       | Default                               | Description                                  |
| ------------------------------ | ------------------------------------- | -------------------------------------------- |
| `pep_length`                   | `9`                                   | Length of peptide sequences                  |
| `res_type`                     | `20`                                  | Number of amino acid types                   |
| **MCTS**                       |                                       |                                              |
| `numIters`                     | `100`                                 | Number of training iterations                |
| `numEps`                       | `200`                                 | Self‑play games per iteration                |
| `numMCTSSims`                  | `200`                                 | MCTS simulations per move                    |
| `cpuct`                        | `0.3`                                 | MCTS exploration constant                    |
| **Thresholds**                 |                                       |                                              |
| `tempThreshold`                | `15`                                  | Temperature cut‑off for move selection       |
| `updateThreshold`              | `0.55`                                | Win‑rate threshold for new net acceptance    |
| **Checkpoint**                 |                                       |                                              |
| `checkpoint`                   | `'./temp/'`                           | Directory to save/load checkpoints           |
| `load_model`                   | `False` (`True` in predict.py\`)      | Whether to load existing model               |
| `load_folder_file`             | `('./temp/','checkpoint_11.pth.tar')` | Tuple of (folder, filename)                  |
| **IEDB Data**                  |                                       |                                              |
| `IEDBdir`                      | `'./data/IEDB'`                       | Path to IEDB directory                       |
| `IEDBtargetdatabase`           | `'IEDB-target-9res.txt'`              | Filename of target peptides                  |
| **Mutation & Scoring**         |                                       |                                              |
| `mutation_rate`                | `'3-1'`                               | Mutation rate string                         |
| `half_life`                    | `500`                                 | Hypothetical half‑life parameter for scoring |
| `T_init`                       | `20`                                  | Initial temperature for scoring              |
| `optimizationSTEP` *(predict)* | `50`                                  | Iterations for peptide optimization          |

---

## Logging & Outputs

* **Training logs** are printed to console via `coloredlogs`.
* **Record files** stored in working directory:

  * `Startrecord-YYYYMMDDHHMM.txt`
  * `Manualrecord-YYYYMMDDHHMM.txt`
* **Prediction outputs** appended under `Predict_data/` as `update-<checkpoint>-<startpeptide>-<timestamp>.txt`.

---

## License

This project is licensed under the [MIT License](LICENSE).
