import sys
import json
import hashlib
import shutil
import traceback
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QFileDialog, QMessageBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal

# -------------------- Utils --------------------
def save_figure_pickle(fig: Figure, path: Path):
    with open(path, "wb") as f:
        pickle.dump(fig, f)

def load_figure_pickle(path: Path) -> Figure:
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------- SaveProgressWindow --------------------
class SaveProgressWindow(QDialog):
    def __init__(self, title="Сохранение проекта..."):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)

        self.cancelled = False
        self.is_finished = False
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self._cancel)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.text_log)
        layout.addWidget(self.cancel_button)

    def _cancel(self):
        if self.is_finished:
            self.accept()
        else:
            self.cancelled = True
            self.log("Отмена сохранения запрошена...")

    def log(self, message):
        print(f"LOG: {message}")
        sys.stdout.flush()
        self.text_log.append(message)

    def set_progress(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def finish(self, message):
        self.is_finished = True
        self.cancel_button.setText("Закрыть")
        self.log(message)

# -------------------- SaveWorker --------------------

class SaveWorker(QObject):
    step = pyqtSignal(int, str)
    done = pyqtSignal(str, str)

    def __init__(self, project_dir, state, progress_window: SaveProgressWindow):
        super().__init__()
        self.project_dir = Path(project_dir)
        self.state = state
        self.figures_dir = self.project_dir / "figures"
        self.progress_window = progress_window

    def run(self):
        try:
            self.project_dir.mkdir(exist_ok=True)
            self.figures_dir.mkdir(exist_ok=True)

            # Только slides и figures идут в project.json
            state_to_save = {
                "slides": self.state["slides"],
                "figures": self.state["figures"]
            }

            self._log("Сохранение project.json")
            with open(self.project_dir / "project.json", "w", encoding="utf-8") as f:
                json.dump(state_to_save, f, indent=4, ensure_ascii=False)

            total_figures = sum(len(columns) for columns in self.state.get("figures", {}).values())
            current = 0

            for fpath, columns in self.state.get("figures", {}).items():
                for col, rel_path in columns.items():
                    fig = self.state["figures_data"][fpath][col]
                    save_path = self.figures_dir / rel_path
                    save_path.parent.mkdir(exist_ok=True)
                    save_figure_pickle(fig, save_path)
                    current += 1
                    self._log(f"Сохранена фигура: {save_path}")

            self.done.emit("ok", "Проект успешно сохранён.")
        except Exception as e:
            tb = traceback.format_exc()
            self.done.emit("err", f"Ошибка сохранения:\n{tb}")

    def _log(self, message):
        self.progress_window.log(message)
        print(f"LOG: {message}")
        sys.stdout.flush()
