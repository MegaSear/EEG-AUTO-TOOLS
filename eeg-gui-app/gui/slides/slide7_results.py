# gui/slides/slide6_results.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog, QHBoxLayout, QTextEdit
)
from PyQt5.QtCore import Qt
import os

class Slide7Results(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.results = []  # Список результатов анализа

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Результаты анализа")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Таблица для отображения результатов
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Имя файла", "QC отчёт", "Предобработка", "ML отчёт"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellClicked.connect(self.display_report)
        layout.addWidget(self.table)

        # Кнопка для экспорта результатов
        export_button = QPushButton("Экспортировать результаты")
        export_button.clicked.connect(self.export_results)
        layout.addWidget(export_button)

        # Область для отображения подробного отчёта
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        layout.addWidget(self.report_view)

    def add_result(self, file_name, qc_report, preprocessing_report, ml_report):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        self.table.setItem(row_position, 0, QTableWidgetItem(file_name))
        self.table.setItem(row_position, 1, QTableWidgetItem("Просмотр"))
        self.table.setItem(row_position, 2, QTableWidgetItem("Просмотр"))
        self.table.setItem(row_position, 3, QTableWidgetItem("Просмотр"))

        self.results.append({
            "file_name": file_name,
            "qc_report": qc_report,
            "preprocessing_report": preprocessing_report,
            "ml_report": ml_report
        })

    def display_report(self, row, column):
        report_types = ["qc_report", "preprocessing_report", "ml_report"]
        if column == 0:
            return  # Имя файла, ничего не делаем
        report = self.results[row].get(report_types[column - 1], "Отчёт недоступен.")
        self.report_view.setText(report)

    def export_results(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for result in self.results:
                        f.write(f"Файл: {result['file_name']}\n")
                        f.write("QC отчёт:\n")
                        f.write(f"{result['qc_report']}\n")
                        f.write("Предобработка:\n")
                        f.write(f"{result['preprocessing_report']}\n")
                        f.write("ML отчёт:\n")
                        f.write(f"{result['ml_report']}\n")
                        f.write("\n" + "-"*50 + "\n\n")
            except Exception as e:
                self.report_view.setText(f"Ошибка при сохранении: {str(e)}")
