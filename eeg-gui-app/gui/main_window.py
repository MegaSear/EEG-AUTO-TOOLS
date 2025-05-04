from __future__ import annotations

import sys, json, hashlib, shutil, traceback, pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QFileDialog, QMessageBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal

from gui.slides.slide1_intro import Slide1Intro
from gui.slides.slide2_file_selection import Slide2FileSelection
from gui.slides.slide3_qc import Slide3QC
from gui.slides.slide4_preprocessing import Slide4Preprocessing
from gui.slides.slide5_Classic_analysis import Slide5Analysis
from gui.slides.slide6_ML_analysis import Slide6MLAnalysis
from gui.slides.slide7_results import Slide7Results
from gui.save_project import SaveProgressWindow, SaveWorker, load_figure_pickle
import os 


# -------------------- MainWindow --------------------
class MainWindow(QMainWindow):
    def __init__(self, project_dir=None):
        super().__init__()
        self.setWindowTitle("EEG Auto Tools GUI")
        self.setGeometry(100, 100, 1200, 800)
        self.project_dir = Path(project_dir) if project_dir else None

        self.save_thread = None
        self.save_worker = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        nav_layout = QHBoxLayout()
        layout.addLayout(nav_layout)

        titles = ["Manual", "DataBase", "QC", "Preprocessing", "Classic Analysis", "ML Analysis", "Dataset Stats"]
        for i, title in enumerate(titles):
            btn = QPushButton(title)
            btn.clicked.connect(lambda _, index=i: self.navigate_to(index))
            nav_layout.addWidget(btn)

        save_btn = QPushButton("Сохранить проект")
        save_btn.clicked.connect(self.save_project)
        nav_layout.addWidget(save_btn)

        load_btn = QPushButton("Загрузить проект")
        load_btn.clicked.connect(lambda: self.load_project())
        nav_layout.addWidget(load_btn)
        self._init_slides() 

    def _init_slides(self):
        if self.project_dir:
            cache_dir = self.project_dir / "cache"
            cache_dir.mkdir(exist_ok=True)

        self.slides = [
            Slide1Intro(), Slide2FileSelection(), Slide3QC(cache_dir=str(cache_dir)), Slide4Preprocessing(),
            Slide5Analysis(), Slide6MLAnalysis(), Slide7Results()
        ]
        for slide in self.slides:
            self.stack.addWidget(slide)
        self.navigate_to(0)

    def initialize_project(self, folder: Path):
        self.project_dir = folder
        if not hasattr(self, "slides") or not self.slides:
            self._init_slides()
        else:
            self.navigate_to(0)
            
    def navigate_to(self, index):
        self.stack.setCurrentIndex(index)

    def _collect_state(self):
        table = self.slides[1].table
        figures_map = {}
        figures_data = {}

        for fpath, colmap in table.data_storage.items():
            for col, val in colmap.items():
                if isinstance(val, Figure):
                    uid = hashlib.sha1(fpath.encode()).hexdigest()[:8]
                    rel_path = f"{uid}/{col}.pkl"
                    figures_map.setdefault(fpath, {})[col] = rel_path
                    figures_data.setdefault(fpath, {})[col] = val

        slides_state = {}
        for idx, slide in enumerate(self.slides):
            key = f"slide{idx+1}"
            if hasattr(slide, "serialize"):
                slides_state[key] = slide.serialize()
            else:
                slides_state[key] = {}

        return {
            "slides": slides_state,
            "figures": figures_map,
            "figures_data": figures_data,
        }

    def save_project(self):
        if not self.project_dir:
            folder = QFileDialog.getExistingDirectory(self, "Выберите папку для проекта")
            if not folder:
                return
            self.initialize_project(Path(folder))

        state = self._collect_state()
        progress = SaveProgressWindow()
        progress.show()

        self.save_thread = QThread()
        self.save_worker = SaveWorker(self.project_dir, state, progress)
        self.save_worker.moveToThread(self.save_thread)

        self.save_worker.done.connect(lambda status, msg: (progress.finish(msg), self.save_thread.quit()))
        self.save_thread.finished.connect(self._on_save_finished)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_thread.start()

    def _on_save_finished(self):
        self.save_thread = None
        self.save_worker = None

    def load_project(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Выберите папку проекта")
            if not folder:
                return
        folder = Path(folder)
        self.initialize_project(folder)

        project_json = folder / "project.json"
        figures_dir = folder / "figures"

        if not project_json.exists():
            QMessageBox.critical(self, "Ошибка", "Файл project.json не найден!")
            return

        with open(project_json, encoding="utf-8") as f:
            state = json.load(f)

        figures = state.get("figures", {})
        for fpath, colmap in figures.items():
            for col, rel_path in colmap.items():
                fig_path = figures_dir / rel_path
                if fig_path.exists():
                    fig = load_figure_pickle(fig_path)
                    self.slides[1].table.set_data(fpath, col, fig)
                else:
                    print(f"LOG: Фигура {fig_path} не найдена при загрузке проекта.")

        self._apply_state(state)
        QMessageBox.information(self, "Готово", "Проект успешно загружен!")

    def _apply_state(self, state: dict):
        slides = state.get("slides", {})
        for idx, slide in enumerate(self.slides):
            key = f"slide{idx+1}"
            if key in slides and hasattr(slide, "deserialize"):
                slide.deserialize(slides[key])
