import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

# Überprüfen, ob die Metal-API aktiv ist
print("Einsatz von Metal-API:", tf.config.list_physical_devices('GPU'))

# Lade das vortrainierte Modell
model = tf.keras.models.load_model('saved_model/animal_classifier.h5')

# Lese die Klassennamen aus der Datei
class_names_path = '/Users/leon/Documents/Tensorflow_apple/saved_model/class_names.txt'
with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()  # Jede Zeile als ein Element in einer Liste

# Pfad zum Verzeichnis mit Bildern
dataset_dir = '/Users/leon/Documents/Tensorflow_apple/Wildfire_test'

# Funktion zum Laden und Preprocessen von Bildern
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255  # Normalisierung
    return img_array

# Listen für tatsächliche und vorhergesagte Klassen
y_true = []
y_pred = []
counter = 0

# Gehe durch alle Bilder in den Unterverzeichnissen
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Preprocessen und Vorhersage treffen
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            
            # Tatsächliche und vorhergesagte Klasse speichern
            y_true.append(class_index)
            y_pred.append(predicted_class)
            print(f'{class_dir} {y_true[counter]} {y_pred[counter]}')
            counter += 1

# Ausgabe der Modellgenauigkeit und des Klassifikationsberichts
print("\nKlassifikationsbericht:")
print(classification_report(y_true, y_pred, target_names=class_names))
