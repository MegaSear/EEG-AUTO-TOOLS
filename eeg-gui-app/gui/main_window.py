# gui/main_window.py

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QStackedWidget, QLabel
)
from PyQt5.QtCore import Qt
import sys
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
            btn.clicked.connect(lambda checked, index = i: self.navigate_to(index))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        self.navigate_to(0)

    def navigate_to(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)


# Точка входа
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
