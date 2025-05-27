# app/bootstrap.py
"""
Ajoute la racine du projet à sys.path pour qu'on puisse faire
`from src import ...` dans toutes les pages.
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent  # dossier racine
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
