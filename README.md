# Classificatore Cappello vs Scarpa — Transfer Learning con MobileNetV2

Classificatore binario di immagini che distingue **cappelli da scarpe** usando **Transfer Learning** con MobileNetV2 pre-addestrato su ImageNet.

**Nessun dataset esterno richiesto** — il modello genera automaticamente 200+ immagini per classe tramite **data augmentation** a partire da una singola foto per classe.

## Architettura

```
Input 224x224x3
    └─ MobileNetV2 (pesi ImageNet congelati) → feature vector
    └─ Global Average Pooling
    └─ Dropout (0.3)
    └─ Dense 128 neuroni (ReLU)
    └─ Dropout (0.2)
    └─ Output 1 neurone (Sigmoid) → cappello / scarpa
```

## Come si usa

Apri il notebook su **Google Colab** (consigliato per la GPU):

1. Carica `notebook.ipynb` su Google Colab
2. Aggiungi una foto di un cappello e una di una scarpa nella cartella del progetto
3. Esegui tutte le celle — il notebook genera le immagini augmentate e addestra il modello

```bash
# In locale
pip install tensorflow numpy matplotlib scikit-learn
jupyter notebook notebook.ipynb
```

## Pipeline

1. **Data Augmentation** — genera 200+ immagini da 1 foto per classe (rotazioni, zoom, flip, contrasto)
2. **Transfer Learning** — MobileNetV2 con pesi ImageNet congelati
3. **Fine-tuning** — addestramento della testa di classificazione con early stopping
4. **Valutazione** — accuracy, matrice di confusione, grafici loss/accuracy

## Tecnologie

- `TensorFlow / Keras` — framework deep learning
- `MobileNetV2` — rete pre-addestrata su ImageNet
- Data Augmentation — `ImageDataGenerator`
- Google Colab con GPU

## Tag

`python` `tensorflow` `transfer-learning` `mobilenetv2` `image-classification` `deep-learning` `data-augmentation` `keras`
