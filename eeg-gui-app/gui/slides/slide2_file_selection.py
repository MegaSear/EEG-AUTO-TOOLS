# gui/slides/slide2_file_selection.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QLabel, QAbstractItemView,
    QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QEvent
import mne
from gui.slides.slide3_qc import Slide3QC
# from gui.slides.slide4_preprocessing import Slide4Preprocessing  # Добавим позже
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import os
import numpy as np

class QCFilterDialog(QDialog):
    def __init__(self, available_keys, thresholds=None):
        super().__init__()
        self.setWindowTitle("Настройка фильтра QC")
        self.thresholds = thresholds or {}
        self.available_keys = available_keys
        layout = QFormLayout(self)
        self.inputs = {}

        for key in self.available_keys:
            value = self.thresholds.get(key, "")
            input_field = QLineEdit(str(value))
            layout.addRow(key, input_field)
            self.inputs[key] = input_field

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_thresholds(self):
        thresholds = {
            k: float(v.text()) if v.text().replace('.', '', 1).isdigit() else float('inf')
            for k, v in self.inputs.items() if v.text()
        }
        return thresholds

class Slide2FileSelection(QWidget):
    def __init__(self):
        super().__init__()
        self.files = []
        self.thresholds = {}
        self.report_columns = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Добавить файлы")
        self.add_button.clicked.connect(self.open_file_dialog)
        btn_layout.addWidget(self.add_button)
        self.import_bids_button = QPushButton("Импорт из BIDS")
        self.import_bids_button.clicked.connect(self.import_bids_dataset)
        btn_layout.addWidget(self.import_bids_button)
        self.filter_button = QPushButton("Настроить фильтр QC")
        self.filter_button.clicked.connect(self.open_filter_dialog)
        btn_layout.addWidget(self.filter_button)
        layout.addLayout(btn_layout)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Имя файла", "Длительность\n(сек)", "Частота дискретизации\n(Гц)",
            "Количество каналов", "QC статус", "Preprocessing статус"
        ])
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.table)
        run_buttons_layout = QHBoxLayout()
        self.run_qc_button = QPushButton("Запустить QC")
        self.run_qc_button.clicked.connect(self.run_qc_for_selected)
        run_buttons_layout.addWidget(self.run_qc_button)
        self.run_preproc_button = QPushButton("Запустить Preprocessing")
        self.run_preproc_button.clicked.connect(self.run_preproc_for_selected)
        run_buttons_layout.addWidget(self.run_preproc_button)
        layout.addLayout(run_buttons_layout)
        self.setFocusPolicy(Qt.StrongFocus)
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            self.delete_selected_files()
        return super().eventFilter(source, event)

    def delete_selected_files(self):
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        msg = QMessageBox()
        msg.setWindowTitle("Подтверждение удаления")
        msg.setText(f"Вы уверены, что хотите удалить информацию об обработке {len(selected_rows)} файлов?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec_() == QMessageBox.Yes:
            for row in selected_rows:
                self.table.removeRow(row)
                self.files.pop(row)

    def bids_path_exists(self, bids_path):
        try:
            _ = bids_path.fpath
            return bids_path.fpath.exists()
        except Exception:
            return False

    def import_bids_dataset(self):
        bids_root = QFileDialog.getExistingDirectory(self, "Выберите BIDS-директорию", "")
        if not bids_root:
            return
        subjects = get_entity_vals(bids_root, 'subject')
        tasks = get_entity_vals(bids_root, 'task')
        all_runs = get_entity_vals(bids_root, 'run')
        for subject in subjects:
            for task in tasks:
                runs = []
                for run in all_runs:
                    bids_path = BIDSPath(subject=subject, task=task, run=run, root=bids_root)
                    if self.bids_path_exists(bids_path):
                        runs.append(run)
                if not runs:
                    runs = [None]
                for run in runs:
                    bids_path = BIDSPath(subject=subject, task=task, run=run, root=bids_root)
                    try:
                        raw = read_raw_bids(bids_path=bids_path, verbose=False)
                        file_path = raw.filenames[0]
                        if file_path not in self.files:
                            self.files.append(file_path)
                            self.add_file_to_table(file_path)
                    except Exception as e:
                        print(f"Ошибка чтения {subject}, {task}, run={run}: {e}")

    def open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите ЭЭГ-файлы",
            "",
            "Поддерживаемые (*.vhdr *.edf *.set);;Все файлы (*)"
        )
        for file in files:
            if file not in self.files:
                self.files.append(file)
                self.add_file_to_table(file)

    def add_file_to_table(self, file_path):
        try:
            raw = mne.io.read_raw(file_path, preload=False, verbose=False)
            duration = raw.times[-1]
            sfreq = raw.info['sfreq']
            n_channels = raw.info['nchan']
        except Exception as e:
            duration = "Ошибка"
            sfreq = "Ошибка"
            n_channels = "Ошибка"
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        def set_item(col, value):
            item = QTableWidgetItem(str(value))
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row_position, col, item)
        set_item(0, os.path.basename(file_path))
        set_item(1, duration)
        set_item(2, sfreq)
        set_item(3, n_channels)
        set_item(4, "None")
        set_item(5, "None")

    def open_filter_dialog(self):
        available_keys = [k for k in self.report_columns.keys() if k.startswith("QC_")]
        dialog = QCFilterDialog(available_keys, self.thresholds)
        if dialog.exec_():
            self.thresholds = dialog.get_thresholds()
            self.apply_qc_filter()

    def apply_qc_filter(self):
        for row in range(self.table.rowCount()):
            try:
                n_channels = float(self.table.item(row, 3).text()) if self.table.item(row, 3).text().replace('.', '', 1).isdigit() else 1
                passed = True
                for key, threshold in self.thresholds.items():
                    if key in self.report_columns:
                        col = self.report_columns[key]
                        value = float(self.table.item(row, col).text()) if self.table.item(row, col) and self.table.item(row, col).text().replace('.', '', 1).isdigit() else float('inf')
                        if "channels" in key.lower():
                            value = value / n_channels * 100
                        if value > threshold:
                            passed = False
                            break
                result = "✔" if passed else "✖"
            except Exception:
                result = "✖"
            item = QTableWidgetItem(result)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, item)

    def run_qc_for_selected(self):
        files_to_qc = [self.files[i] for i in range(len(self.files))
                       if self.table.item(i, 4).text() == "None"]
        if not files_to_qc:
            print("⚠ Нет файлов для QC.")
            return
        print("🔍 Запуск QC для:", files_to_qc)
        from gui.main_window import MainWindow
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[2], Slide3QC):
            mw.slides[2].set_input_files(files_to_qc)
            mw.slides[2].start_processing()

    def run_preproc_for_selected(self):
        files_to_proc = [self.files[i] for i in range(len(self.files))
                         if self.table.item(i, 4).text() == "✔"]
        if not files_to_proc:
            print("⚠ Нет файлов, прошедших QC.")
            return
        print("⚙ Запуск Preprocessing для:", files_to_proc)
        from gui.main_window import MainWindow
        from gui.slides.slide4_preprocessing import Slide4Preprocessing
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[3], Slide4Preprocessing):
            mw.slides[3].set_input_files(files_to_proc)
            mw.slides[3].start_processing()

    def update_reports(self, reports_per_file, transform_names):
        all_keys = set()
        for file_path, report in reports_per_file.items():
            for t_name, repo_data in report.items():
                for key in repo_data.keys():
                    all_keys.add(f"{t_name}_{key}")
        current_columns = set(self.report_columns.keys())
        new_columns = all_keys - current_columns
        for col_name in new_columns:
            col_index = self.table.columnCount()
            self.table.insertColumn(col_index)
            self.table.setHorizontalHeaderItem(col_index, QTableWidgetItem(col_name))
            self.report_columns[col_name] = col_index
        for row, file_path in enumerate(self.files):
            report = reports_per_file.get(file_path, {})
            for t_name, repo_data in report.items():
                for key, value in repo_data.items():
                    col_name = f"{t_name}_{key}"
                    if col_name in self.report_columns:
                        item = QTableWidgetItem(str(value))
                        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                        self.table.setItem(row, self.report_columns[col_name], item)
            if report:
                # Если отчёт не пустой, проверяем QC статус
                has_qc = any(col_name.startswith("QC_") for col_name in report)
                if has_qc:
                    item = QTableWidgetItem("✔")
                else:
                    item = QTableWidgetItem("None")
            else:
                # Если отчёта вообще нет, оставляем None
                item = QTableWidgetItem("None")
            status_col = 4
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, status_col, item)

    def clear(self):
        """Очищает все данные слайда."""
        self.files.clear()
        self.thresholds.clear()
        self.report_columns.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Имя файла", "Длительность\n(сек)", "Частота дискретизации\n(Гц)",
            "Количество каналов", "QC статус", "Preprocessing статус"
        ])

    def serialize(self):
        """Сериализует данные слайда для сохранения в файл."""
        data = {
            "files": self.files,
            "thresholds": self.thresholds,
            "report_columns": self.report_columns,
            "table_data": []
        }
        # Получаем заголовки столбцов
        headers = []
        for col in range(self.table.columnCount()):
            header_item = self.table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else "")
        
        # Сохраняем данные таблицы
        for row in range(self.table.rowCount()):
            row_data = {}
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                row_data[headers[col]] = item.text() if item else ""
            data["table_data"].append(row_data)
        return data

    def deserialize(self, data):
        """Загружает данные слайда из словаря."""
        self.clear()
        self.files = data.get("files", [])
        self.thresholds = data.get("thresholds", {})
        self.report_columns = data.get("report_columns", {})
        
        # Восстанавливаем столбцы таблицы
        if self.report_columns:
            self.table.setColumnCount(len(self.report_columns) + 6)
            headers = [
                "Имя файла", "Длительность\n(сек)", "Частота дискретизации\n(Гц)",
                "Количество каналов", "QC статус", "Preprocessing статус"
            ] + list(self.report_columns.keys())
            self.table.setHorizontalHeaderLabels(headers)

        # Восстанавливаем данные таблицы
        table_data = data.get("table_data", [])
        # Получаем текущие заголовки столбцов
        headers = []
        for col in range(self.table.columnCount()):
            header_item = self.table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else "")
        
        for row_data in table_data:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, header in enumerate(headers):
                value = row_data.get(header, "")
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if col in (4, 5):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
