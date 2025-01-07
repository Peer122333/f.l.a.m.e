import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Lade das vortrainierte Modell
model = tf.keras.models.load_model('saved_model/animal_classifier.h5')

# Lese die Klassennamen aus der Datei
class_names_path = '/Users/leon/Documents/Tensorflow_apple/saved_model/class_names.txt'
with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()  # Jede Zeile als ein Element in einer Liste

# Bild laden und preprocessen
img_path = '/Users/leon/Documents/Tensorflow_apple/Wildfire/nowildfire/-73.8358,45.482147.jpg'  # Bildpfad
img = image.load_img(img_path, target_size=(128, 128), color_mode='rgb')  # Farbmodus auf 'rgb' setzen
img_array = image.img_to_array(img)  # Bild in Array umwandeln
img_array = np.expand_dims(img_array, axis=0)  # Eine Dimension für den Batch hinzufügen
img_array = img_array.astype('float32') / 255  # Bild normalisieren

# Vorhersage treffen
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Ausgabe der Vorhersage mit Klassennamen
print(f"Predicted class: {class_names[predicted_class]}")  # Ausgabe der Klasse mit Namen

# (Optional) Ausgabe der Wahrscheinlichkeiten
print(f"Class probabilities: {prediction}")
