# Classificatore Cappello vs Scarpa — Transfer Learning con MobileNetV2

Classificatore binario di immagini che distingue **cappelli da scarpe** usando il **Transfer Learning** con MobileNetV2 pre-addestrato su ImageNet.

## Cosa fa

- Caricamento e preparazione del dataset di immagini
- Fine-tuning di MobileNetV2 (pesi ImageNet congelati + testa di classificazione custom)
- Addestramento e valutazione del modello
- Predizione su nuove immagini

## Come si usa

```bash
pip install tensorflow numpy matplotlib
python classifier.py
```

## Tecnologie

- `TensorFlow / Keras` — framework deep learning
- `MobileNetV2` — rete pre-addestrata su ImageNet
- Transfer Learning + Fine-tuning

## Tag

`python` `tensorflow` `transfer-learning` `mobilenetv2` `image-classification` `deep-learning` `keras`
