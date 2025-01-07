import importlib

# Liste der zu prüfenden Bibliotheken
libraries = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.preprocessing",
    "tensorflow.keras.utils",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn"
]

def check_library(library):
    try:
        # Versuche die Bibliothek zu importieren
        importlib.import_module(library)
        print(f"✅ {library} wurde erfolgreich importiert.")
    except ImportError:
        print(f"❌ {library} ist nicht installiert.")
    except Exception as e:
        print(f"⚠️ Ein Fehler ist bei {library} aufgetreten: {e}")

if __name__ == "__main__":
    print("Überprüfe Bibliotheken...\n")
    for lib in libraries:
        check_library(lib)
    print("\nÜberprüfung abgeschlossen.")
