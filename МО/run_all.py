"""Последовательно запускает и сохраняет шесть ноутбуков с результатами."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient

BASE = Path(__file__).resolve().parent
# Ядро Jupyter должно использовать Python текущего окружения.
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
for path in sorted(BASE.glob('Практика_*.ipynb')):
    print('Запускаю', path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(BASE)}}).execute()
    nbformat.write(nb, path)
print('Все шесть ноутбуков выполнены без ошибок.')
