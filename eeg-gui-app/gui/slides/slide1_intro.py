# gui/slides/slide1_intro.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt

class Slide1Intro(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Добро пожаловать в EEG Auto Tools GUI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setStyleSheet("font-size: 14px;")
        instructions.setText(
            "Это приложение предназначено для обработки и анализа ЭЭГ данных с использованием библиотеки eeg-auto-tools.\n\n"
            "Шаги работы:\n"
            "1. Ознакомьтесь с руководством пользователя.\n"
            "2. Загрузите один или несколько ЭЭГ файлов.\n"
            "3. Настройте последовательность блоков для проверки качества (Рекомендуется настройка по умолчанию).\n"
            "4. Настройте последовательность блоков для предобработки данных.\n"
            "5. Выполните анализ данных с использованием методов машинного обучения.\n"
            "6. Просмотрите и сохраните результаты анализа.\n\n"
            "Используйте кнопки навигации внизу для перехода между этапами обработки ваших данных."
        )
        layout.addWidget(instructions)
