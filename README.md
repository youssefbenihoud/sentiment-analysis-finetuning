# Sentiment Analysis Fine-Tuning

Dieses Projekt demonstriert das Fine-Tuning eines vortrainierten BERT-Modells für Sentiment Analysis auf dem IMDB-Datensatz.

## Inhaltsverzeichnis
- [Projektbeschreibung](#projektbeschreibung)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Ergebnisse](#ergebnisse)

## Projektbeschreibung
Das Ziel des Projekts ist es, ein vortrainiertes BERT-Modell (`bert-base-uncased`) für Sentiment Analysis (positive/negative) zu fine-tunen. Der Datensatz stammt von [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Installation
1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/DEIN_USERNAME/sentiment-analysis-finetuning.git
   cd sentiment-analysis-finetuning

2. Installieren Sie die benötigten Bibliotheken:
    ```bash
    pip install -r requirements.txt
    
## Verwendung
 
 Um das Modell mit neuen Daten zu testen, führen Sie das folgende Skript aus:
    ```bash
    python src/inference.py --text "Dein Text hier"

## Ergebnisse
- Trainingsdaten: 80% des IMDB-Datensatzes
- Testdaten: 20% des IMDB-Datensatzes
- Accuracy nach Fine-Tuning: