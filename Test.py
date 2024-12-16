import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Modell und Klassennamen laden
try:
    model = load_model("saved_model/animal_classifier.keras")
    print("Modell erfolgreich geladen!")
    model.summary()  # Zeigt eine Zusammenfassung des Modells an
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")

# Klassennamen aus der gespeicherten Datei laden (falls vorhanden)
# Wenn du die Klassennamen direkt aus dem Modell beziehst, kannst du den folgenden Code überspringen
with open("saved_model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Angenommene Bildpfad
img_path = "/Users/leon/Documents/Tensorflow_apple/AnimalFaces/DogHead/dog000046a.jpg"  # Beispielbildpfad

# Bild vorverarbeiten
img = image.load_img(img_path, target_size=(128, 128))  # Bildgröße wie beim Training
img_array = image.img_to_array(img)  # Bild in Array umwandeln
img_array = np.expand_dims(img_array, axis=0)  # Für das Batch-Modell vorbereiten
img_array = img_array / 255.0  # Normierung auf [0, 1], wenn nötig

# Vorhersage durchführen
predictions = model.predict(img_array)

# Debugging: Überprüfen der Form von predictions
print("Vorhersage-Array:", predictions)
print("Vorhersage-Array Form:", predictions.shape)

# Sicherstellen, dass predictions eine erwartete Form hat
if predictions.ndim == 2 and predictions.shape[0] == 1:  # Überprüfen, ob es nur eine Zeile gibt
    predictions = predictions[0]  # Nur das erste (und einzige) Element extrahieren

# Debugging: Nach dem Extrahieren der Wahrscheinlichkeiten
print("Extrahiertes Vorhersage-Array:", predictions)

# Überprüfen, ob predictions leer ist oder nicht die richtige Länge hat
if len(predictions) == 0:
    print("Fehler: Keine Vorhersage erhalten.")
else:
    # Vorhersage anzeigen
    predicted_class_index = np.argmax(predictions)
    print(f"Vorhersage-Index: {predicted_class_index}")
    print(f"Vorhergesagte Klasse: {class_names[predicted_class_index]}")
