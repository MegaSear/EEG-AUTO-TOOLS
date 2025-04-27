# gui/main_window.py

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QStackedWidget, QLabel,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
import sys
import json
from gui.slides.slide1_intro import Slide1Intro
from gui.slides.slide2_file_selection import Slide2FileSelection
from gui.slides.slide3_qc import Slide3QC
from gui.slides.slide4_preprocessing import Slide4Preprocessing
from gui.slides.slide5_Classic_analysis import Slide5Analysis
from gui.slides.slide6_ML_analysis import Slide6MLAnalysis
from gui.slides.slide7_results import Slide7Results

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Auto Tools GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной макет
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Стек виджетов для слайдов
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Инициализация слайдов
        self.slides = [
            Slide1Intro(),
            Slide2FileSelection(),
            Slide3QC(),
            Slide4Preprocessing(),
            Slide5Analysis(),
            Slide6MLAnalysis(),
            Slide7Results()
        ]
        self.slides_names = ["Manual", "DataBase", "QC", "Preprocessing", "Classic Analysis", "ML Analysis", "Dataset Stats"]

        for slide in self.slides:
            self.stack.addWidget(slide)

        # Навигационная панель
        nav_layout = QHBoxLayout()
        main_layout.addLayout(nav_layout)

        self.nav_buttons = []
        for i, slide_name in enumerate(self.slides_names):
            btn = QPushButton(slide_name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, index=i: self.navigate_to(index))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        # Кнопки для сохранения и загрузки проекта
        self.save_button = QPushButton("Сохранить проект")
        self.save_button.clicked.connect(self.save_project)
        nav_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Загрузить проект")
        self.load_button.clicked.connect(self.load_project)
        nav_layout.addWidget(self.load_button)

        self.navigate_to(0)

    def navigate_to(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", "", "EEG Project Files (*.eegproj)"
        )
        if not file_path:
            return

        # Собираем данные со всех слайдов
        project_data = {
            "slides": {}
        }

        # Slide1Intro (ничего не сохраняем, так как это статический слайд)
        project_data["slides"]["slide1"] = {}

        # Slide2FileSelection
        project_data["slides"]["slide2"] = self.slides[1].serialize()

        # Slide3QC
        project_data["slides"]["slide3"] = self.slides[2].serialize()

        # Slide4Preprocessing (пока заглушка, добавим позже)
        project_data["slides"]["slide4"] = self.slides[3].serialize() if hasattr(self.slides[3], 'serialize') else {}

        # Slide5Analysis, Slide6MLAnalysis, Slide7Results (заглушки)
        project_data["slides"]["slide5"] = self.slides[4].serialize() if hasattr(self.slides[4], 'serialize') else {}
        project_data["slides"]["slide6"] = self.slides[5].serialize() if hasattr(self.slides[5], 'serialize') else {}
        project_data["slides"]["slide7"] = self.slides[6].serialize() if hasattr(self.slides[6], 'serialize') else {}

        # Сохраняем проект в файл
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4)
            QMessageBox.information(self, "Успех", "Проект успешно сохранён!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить проект: {e}")

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить проект", "", "EEG Project Files (*.eegproj)"
        )
        if not file_path:
            return

        # Загружаем данные из файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить проект: {e}")
            return

        # Очищаем текущие данные всех слайдов
        for slide in self.slides:
            if hasattr(slide, 'clear'):
                slide.clear()

        # Загружаем данные в слайды
        if "slides" in project_data:
            # Slide2FileSelection
            if "slide2" in project_data["slides"]:
                self.slides[1].deserialize(project_data["slides"]["slide2"])

            # Slide3QC
            if "slide3" in project_data["slides"]:
                self.slides[2].deserialize(project_data["slides"]["slide3"])

            # Slide4Preprocessing (пока заглушка)
            if "slide4" in project_data["slides"] and hasattr(self.slides[3], 'deserialize'):
                self.slides[3].deserialize(project_data["slides"]["slide4"])

            # Slide5Analysis, Slide6MLAnalysis, Slide7Results (заглушки)
            if "slide5" in project_data["slides"] and hasattr(self.slides[4], 'deserialize'):
                self.slides[4].deserialize(project_data["slides"]["slide5"])
            if "slide6" in project_data["slides"] and hasattr(self.slides[5], 'deserialize'):
                self.slides[5].deserialize(project_data["slides"]["slide6"])
            if "slide7" in project_data["slides"] and hasattr(self.slides[6], 'deserialize'):
                self.slides[6].deserialize(project_data["slides"]["slide7"])

        QMessageBox.information(self, "Успех", "Проект успешно загружен!")

# Точка входа
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())