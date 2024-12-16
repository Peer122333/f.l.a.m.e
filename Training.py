import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Datenverzeichnis
dataset_dir = "Wildfire"  # Passe den Pfad an

# Überprüfen, ob das Dataset existiert
if not os.path.exists(dataset_dir):
    print(f"Fehler: Das Verzeichnis '{dataset_dir}' wurde nicht gefunden.")
else:
    # Trainings- und Validierungsdatensätze erstellen
    train_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(128, 128),
        batch_size=32
    )

    val_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=32
    )
    
    class_names = train_dataset.class_names
    print("Klassen:", class_names)

    # Modell erstellen
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')  # Eine Ausgabe pro Klasse
    ])

    # Modell kompilieren
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Modell trainieren
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )

    # Genauigkeit auf Validierungsdaten
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validierungsgenauigkeit: {val_accuracy:.2f}")

    # Modell speichern
    save_path = "saved_model/animal_classifier.h5"
    model.save(save_path)
    print(f"Modell erfolgreich gespeichert unter: {save_path}")

    # Klassennamen speichern
    with open("saved_model/class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print("Klassennamen erfolgreich gespeichert.")
