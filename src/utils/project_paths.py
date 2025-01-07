from pathlib import Path

def find_project_root(start_path: Path, root_name: str = "f.l.a.m.e") -> Path:
    """
    Suche den Projekt-Root-Ordner (z. B. 'f.l.a.m.e'), indem du die Hierarchie nach oben gehst.
    :param start_path: Pfad, von dem aus die Suche beginnt
    :param root_name: Name des Root-Ordners
    :return: Der Pfad zum Projekt-Root
    """
    current_path = start_path.resolve()
    while current_path.name != root_name:
        if current_path.parent == current_path:  # Root des Dateisystems erreicht
            raise FileNotFoundError(f"Projekt-Root '{root_name}' wurde nicht gefunden.")
        current_path = current_path.parent
    return current_path
