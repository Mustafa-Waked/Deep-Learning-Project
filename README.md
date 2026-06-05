# Deep Learning Project

Siamese neural network for **artist verification**: given two painting images, predict whether they were created by the **same artist** (binary classification).

## What this project does

University of Haifa deep learning coursework. The pipeline loads painting metadata from CSV, builds positive/negative image pairs, trains a **convolutional Siamese network** with a frozen **ResNet101** backbone (ImageNet weights), and evaluates pair classification accuracy.

## Technologies

- Python 3.9–3.12, Jupyter Notebook
- TensorFlow / Keras
- ResNet101 (`tf.keras.applications.resnet.ResNet101`)
- NumPy, Pandas, scikit-learn
- Pillow, torchvision (image loading/transforms)
- Matplotlib (training curves)

## Features

- **Data cleaning** (`DataCleaning.ipynb`): remove corrupted entries using `replacements_for_corrupted_files.zip`, output `all_data_info_1.csv`
- **Pair generation**: same-artist pairs (label 1) vs different-artist pairs (label 0)
- **Preprocessing**: resize to 224×224, ResNet v2 preprocessing
- **Siamese model**: shared encoder, absolute feature difference, sigmoid output
- **Training**: RMSprop, binary cross-entropy, early stopping on validation accuracy
- **Visualizations**: sample image pairs, train/val loss and accuracy plots

## Project structure

```
Deep-Learning-Project/
├── DataCleaning.ipynb
├── ProjectCode.ipynb
├── DLProjectReport.pdf
├── requirements.txt
├── scripts/validate_notebooks.py
├── .gitignore
└── README.md
```

### Expected data layout (not in this repo)

Place these next to the notebooks before running:

```
Deep-Learning-Project/
├── all_data_info.csv
├── replacements_for_corrupted_files.zip
├── all_data_info_1.csv          # produced by DataCleaning.ipynb
├── train/train_crop/
└── test/test_crop/
```

## Prerequisites

- **Python 3.9–3.12** (TensorFlow does not support Python 3.13+ on most platforms)
- Jupyter Notebook or JupyterLab
- Course painting dataset (CSV + images) — not included due to size
- GPU recommended for training

## Installation

```bash
git clone https://github.com/Mustafa-Waked/Deep-Learning-Project.git
cd Deep-Learning-Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Build

No separate compile step. Install dependencies with `pip install -r requirements.txt`.

## Run

### Step 0 — Validate environment

```bash
python scripts/validate_notebooks.py
```

### Step 1 — Clean the dataset

1. Open `DataCleaning.ipynb`
2. Place `all_data_info.csv` and `replacements_for_corrupted_files.zip` in the project root
3. Run all cells → creates `all_data_info_1.csv`

### Step 2 — Train and evaluate

1. Open `ProjectCode.ipynb`
2. Ensure `all_data_info_1.csv` and image folders exist:
   - `train/train_crop/`
   - `test/test_crop/`
3. Run cells in order: load metadata → pair generators → build Siamese ResNet101 model → train → plot curves

## Example workflow

```text
DataCleaning.ipynb  →  all_data_info_1.csv
ProjectCode.ipynb   →  pair batches from train/train_crop & test/test_crop
                    →  Siamese net training (same artist = 1, different = 0)
```

## Model summary

| Component | Detail |
|-----------|--------|
| Backbone | ResNet101 (frozen, ImageNet weights) |
| Inputs | Two images 224×224×3 |
| Distance | `abs(encoded_a - encoded_b)` |
| Head | Global max pooling → dropout → Dense(1, sigmoid) |
| Loss | `binary_crossentropy` |
| Optimizer | RMSprop (lr=0.0001) |

## Expected output

- `all_data_info_1.csv` after data cleaning
- Training logs with validation accuracy (recorded run reached ~**66%** before early stopping)
- Loss/accuracy plots when notebook cells are re-executed

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `No matching distribution found for tensorflow` | Python 3.13+ | Create a venv with Python 3.9–3.12. |
| `FileNotFoundError: all_data_info.csv` | Dataset not downloaded | Obtain course CSV/images locally; see layout above. |
| OOM during training | Batch size / GPU memory | Reduce batch size in `ProjectCode.ipynb` (default 32). |
| Slow training on CPU | Large ResNet101 backbone | Use a GPU runtime or Colab. |
| Empty plots in repo | Outputs cleared for git size | Re-run notebook cells to regenerate figures. |

## Notes / limitations

- **Painting images and CSV/zips are not included** in GitHub (too large).
- Results depend on data, hardware, and random pair sampling.
- Training is slow on CPU; GPU recommended.
- Notebook cell outputs were cleared in the repo to reduce file size.

## Author

Mustafa Waked — University of Haifa
