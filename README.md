# Deep Learning Project

Siamese neural network for **artist verification**: given two painting images, predict whether they were created by the **same artist** (binary classification).

## Description

University of Haifa deep learning coursework. The pipeline loads painting metadata from CSV, builds positive/negative image pairs, trains a **convolutional Siamese network** with a frozen **ResNet101** backbone (ImageNet weights), and evaluates pair classification accuracy.

## Technologies

- Python, Jupyter Notebook
- TensorFlow / Keras
- ResNet101 (`tf.keras.applications.resnet.ResNet101`)
- NumPy, Pandas, scikit-learn
- Pillow, torchvision (image loading/transforms)
- Matplotlib (training curves)

## Features

- **Data cleaning** (`DataCleaning.ipynb`): remove corrupted entries using `replacements_for_corrupted_files.zip`, output `all_data_info_1.csv`
- **Pair generation**: same-artist pairs (label 1) vs different-artist pairs (label 0)
- **Preprocessing**: resize to 224×224, ResNet preprocessing
- **Siamese model**: shared encoder, absolute feature difference, sigmoid output
- **Training**: RMSprop, binary cross-entropy, early stopping on validation accuracy (20 epochs max, stopped at epoch 7 in recorded run)
- **Visualizations**: sample image pairs, train/val loss and accuracy plots

## Project structure

```
Deep-Learning-Project/
├── DataCleaning.ipynb      # CSV cleaning pipeline
├── ProjectCode.ipynb       # Preprocessing, model, training, evaluation
├── DLProjectReport.pdf     # Written project report
├── requirements.txt
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
├── train/train_crop/            # training images
└── test/test_crop/              # test/validation images
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Use a Python environment compatible with your TensorFlow version (the original project used TensorFlow 2.x with Keras).

## How to run

### Step 1 — Clean the dataset

1. Open `DataCleaning.ipynb`
2. Ensure `all_data_info.csv` and `replacements_for_corrupted_files.zip` are in the project root
3. Run all cells → creates `all_data_info_1.csv`

### Step 2 — Train and evaluate

1. Open `ProjectCode.ipynb`
2. Ensure `all_data_info_1.csv` and image folders exist:
   - `train/train_crop/`
   - `test/test_crop/`
3. Run cells in order:
   - Load metadata and build train/test artist dictionaries
   - Create pair generators (batch size **32**)
   - Build and compile Siamese ResNet101 model
   - Train with early stopping
   - Plot accuracy/loss curves

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

## Notes / limitations

- **Painting images and CSV/zips are not included** in GitHub (too large). You need the course dataset locally.
- Recorded training reached ~**66%** validation accuracy before early stopping; results depend on data and hardware.
- `ProjectCode.ipynb` markdown still mentions VGG16 in places; the implemented backbone is **ResNet101**.
- Training is slow on CPU (~20+ minutes per epoch in the saved run); GPU recommended.
- Notebook cell outputs were cleared in the repo to reduce file size; re-run notebooks to regenerate plots.

## Author

Mustafa Waked — University of Haifa
