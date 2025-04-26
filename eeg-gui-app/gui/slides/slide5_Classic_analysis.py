# gui/slides/slide5_analysis.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QStackedWidget, QFormLayout, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt

class Slide5Analysis(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Анализ данных")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Выбор метода анализа
        self.analysis_selector = QComboBox()
        self.analysis_selector.addItems(["ERP", "ERSP", "Извлечение признаков", "Классификация (ML/DL)"])
        self.analysis_selector.currentIndexChanged.connect(self.switch_analysis_form)
        layout.addWidget(self.analysis_selector)

        # Стек виджетов для форм настройки
        self.form_stack = QStackedWidget()
        layout.addWidget(self.form_stack)

        # Формы настройки для каждого метода анализа
        self.erp_form = self.create_erp_form()
        self.ersp_form = self.create_ersp_form()
        self.feature_form = self.create_feature_form()
        self.classifier_form = self.create_classifier_form()

        self.form_stack.addWidget(self.erp_form)
        self.form_stack.addWidget(self.ersp_form)
        self.form_stack.addWidget(self.feature_form)
        self.form_stack.addWidget(self.classifier_form)

        # Кнопка запуска анализа
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)

    def switch_analysis_form(self, index):
        self.form_stack.setCurrentIndex(index)

    def create_erp_form(self):
        form = QWidget()
        layout = QFormLayout()
        form.setLayout(layout)

        self.erp_event_code = QLineEdit()
        self.erp_time_window = QLineEdit()
        layout.addRow("Код события:", self.erp_event_code)
        layout.addRow("Окно времени (мс):", self.erp_time_window)

        return form

    def create_ersp_form(self):
        form = QWidget()
        layout = QFormLayout()
        form.setLayout(layout)

        self.ersp_frequency_range = QLineEdit()
        self.ersp_time_window = QLineEdit()
        layout.addRow("Диапазон частот (Hz):", self.ersp_frequency_range)
        layout.addRow("Окно времени (мс):", self.ersp_time_window)

        return form

    def create_feature_form(self):
        form = QWidget()
        layout = QFormLayout()
        form.setLayout(layout)

        self.feature_type = QLineEdit()
        self.feature_parameters = QLineEdit()
        layout.addRow("Тип признаков:", self.feature_type)
        layout.addRow("Параметры:", self.feature_parameters)

        return form

    def create_classifier_form(self):
        form = QWidget()
        layout = QFormLayout()
        form.setLayout(layout)

        self.classifier_type = QLineEdit()
        self.classifier_parameters = QLineEdit()
        layout.addRow("Тип классификатора:", self.classifier_type)
        layout.addRow("Параметры:", self.classifier_parameters)

        return form

    def run_analysis(self):
        current_index = self.form_stack.currentIndex()
        if current_index == 0:
            event_code = self.erp_event_code.text()
            time_window = self.erp_time_window.text()
            # Здесь должна быть логика запуска ERP-анализа с указанными параметрами
            QMessageBox.information(self, "ERP", f"Запуск ERP-анализа с кодом события {event_code} и окном {time_window} мс.")
        elif current_index == 1:
            freq_range = self.ersp_frequency_range.text()
            time_window = self.ersp_time_window.text()
            # Здесь должна быть логика запуска ERSP-анализа с указанными параметрами
            QMessageBox.information(self, "ERSP", f"Запуск ERSP-анализа с диапазоном частот {freq_range} Hz и окном {time_window} мс.")
        elif current_index == 2:
            feature_type = self.feature_type.text()
            parameters = self.feature_parameters.text()
            # Здесь должна быть логика извлечения признаков с указанными параметрами
            QMessageBox.information(self, "Извлечение признаков", f"Извлечение признаков типа {feature_type} с параметрами {parameters}.")
        elif current_index == 3:
            classifier_type = self.classifier_type.text()
            parameters = self.classifier_parameters.text()
            # Здесь должна быть логика запуска классификации с указанными параметрами
            QMessageBox.information(self, "Классификация", f"Запуск классификации с типом {classifier_type} и параметрами {parameters}.")
