import pandas as pd

# Laden des Datensatzes
data = pd.read_csv("../data/IMDB Dataset.csv")

# Überprüfen der ersten Zeilen
print(data.head())

# Aufteilung in Trainings- und Testdaten
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Speichern der aufgeteilten Daten
train_data.to_csv("../data/train.csv", index=False)
test_data.to_csv("../data/test.csv", index=False)

print("Daten wurden erfolgreich vorbereitet und gespeichert")