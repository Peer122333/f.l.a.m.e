import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

def is_valid_image(file_path):
    try:
        # Versuche, das Bild zu lesen und zu dekodieren
        image_data = tf.io.read_file(file_path)
        image_decoded = tf.image.decode_jpeg(image_data, channels=3)  # Stelle sicher, dass das Bild 3 Farbkanäle hat
        return True
    except tf.errors.InvalidArgumentError:
        # Fehler beim Dekodieren des Bildes, es ist beschädigt
        return False

def clean_invalid_images(dataset_dir):
    # Gehe durch alle Unterordner im Dataset-Verzeichnis
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(('.jpg', '.jpeg', '.png')):  # Nur Bilddateien überprüfen
                if not is_valid_image(file_path):
                    print(f"Ungültiges Bild gefunden und gelöscht: {file_path}")
                    os.remove(file_path)

# Pfad zu deinem Dataset
dataset_dir = "Wildfire_test"  # Passe den Pfad an

# Überprüfe und lösche beschädigte Bilder
clean_invalid_images(dataset_dir)

# Jetzt kannst du das Dataset wie gewohnt laden
train_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
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
