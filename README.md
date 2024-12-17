# f.l.a.m.e 🔥
Machine Learning project to use satellite imagery to analyze and predict forest fires by using a combination of weather data and image classification.

## Progress so far:
- **Data Cleaning:**
  - Cleaned up corrupted data using the script Überprüfung.py (library: toco).

- **Initial Model Implementation:**
  - Built a basic CNN model as a starting point.
  - Challenges:
    - **Training Data Issues:** Similar-looking images make classification difficult.
    - **Performance Bottleneck:** Adding more than 30 CNN layers results in diminishing returns and degrading accuracy due to vanishing gradients.

- **Solution Attempts:**
  - **Deeper Neural Networks:** Tried increasing the depth of the network but faced challenges in finding the optimal configuration.
  - **Interim Solution:** Adopted a preconstructed network, ResNet50, for initial experiments.

## About ResNet50:
- **Architecture Overview:**
  - **50 Layers:** Deep architecture designed to handle complex feature hierarchies.
  - **Residual Blocks:**
    - Prevent vanishing gradients by using shortcut connections.
    - Allow direct propagation of the input to deeper layers, ensuring better training efficiency and convergence.

- **Key Advantages:**
  - Handles vanishing gradient problems effectively, even with high depth.

- **Implementation Notes:**

## Next Steps:
- **Experiment with ResNet50's hyperparameters:**
  - Fine-tune ResNet50 on our dataset.
  - Explore techniques like data augmentation to enhance the model's robustness against variability in input images.
  - Learning rate adjustments.
  - Freeze/unfreeze specific layers for transfer learning.
  - Integrate weather data as an additional feature input to improve predictions.
  - Develop ensemble approaches (e.g., combining ResNet50 with a temporal model like LSTM for sequential weather data).
  - Evaluate performance using metrics like precision, recall, specific to fire prediction accuracy.



## Strukturwunsch: 
`
ML_Project/
│
├── data/                      # Datenverzeichnis
│   ├── raw/                   # Rohdaten (unbearbeitet)
│   ├── processed/             # Vorverarbeitete Daten
│   ├── AnimalFaces/           # Spezifische Projektdaten (AnimalFaces)
│   ├── Wildfire/              # Spezifische Projektdaten (Wildfire)
│   └── Wildfire_test/         # Testdaten (Wildfire)
│
├── notebooks/                 # Jupyter Notebooks für Exploratory Data Analysis (EDA)
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── models/                    # Gespeicherte Modelle und Checkpoints
│   ├── saved_model/           # Exportierte Modelle
│   └── model_checkpoints/     # Trainings-Checkpoints
│
├── src/                       # Quellcode des Projekts
│   ├── __init__.py            # Markiert src als Paket
│   ├── data_processing.py     # Datenvorverarbeitung (z.B. Cleaning, Augmentation)
│   ├── model_training.py      # Model-Trainingslogik
│   ├── model_evaluation.py    # Evaluierungslogik
│   └── utils.py               # Hilfsfunktionen
│
├── tests/                     # Tests für Code und Modelle
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   └── test_utils.py
│
├── scripts/                   # Python-Skripte für Automatisierung
│   ├── train_model.py         # Trainings-Startskript
│   ├── evaluate_model.py      # Evaluations-Startskript
│   └── predict.py             # Skript zur Vorhersage
│
├── requirements.txt           # Python-Abhängigkeiten
├── README.md                  # Projektbeschreibung und Anweisungen
├── LICENSE                    # Lizenzdatei
├── .gitignore                 # Dateien, die Git ignorieren soll
└── Überprüfung.py             # Falls benötigt, integrieren oder neu benennen
`
