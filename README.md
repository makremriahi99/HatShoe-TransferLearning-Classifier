# Hat vs Shoe — Transfer Learning Classifier

Binary image classifier that distinguishes **hats from shoes** using **Transfer Learning** with MobileNetV2 pre-trained on ImageNet.

## What it does

- Loads a custom dataset of hat/shoe images from Google Drive
- Uses **MobileNetV2** (frozen base) as a feature extractor
- Adds a custom classification head (GlobalAveragePooling → Dense → Sigmoid)
- Fine-tunes top layers with `Adam` optimizer
- Achieves high accuracy with minimal training data thanks to pre-trained features
- Evaluates on a held-out test set with accuracy and loss curves

## Architecture

```
MobileNetV2 (frozen, ImageNet weights)
    └─ GlobalAveragePooling2D
    └─ Dense(128, ReLU)
    └─ Dropout(0.3)
    └─ Dense(1, Sigmoid)   ← binary output
```

## Tech stack

- Python 3
- `TensorFlow` / `Keras` — model definition and training
- `MobileNetV2` — pre-trained backbone
- `matplotlib` — training curve visualization

## Usage

Open `notebook.ipynb` in Google Colab (GPU recommended). Ensure the dataset folder is accessible via the provided Drive path.

## Topics

`python` `tensorflow` `keras` `transfer-learning` `mobilenetv2` `image-classification` `deep-learning`
