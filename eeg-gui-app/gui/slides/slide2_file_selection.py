from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox, QTableWidgetItem,
    QListWidget, QLabel, QTextEdit, QTreeWidget, QTreeWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QEvent, QTimer
from gui.table import FileTableWidget
from gui.utils import EEGFileManager
import os
import mne
import json
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from gui.log_dialog import LogDialog
from gui.image_view_dialog import ImageViewDialog
from gui.channel_list_dialog import ChannelListDialog
from gui.qc_filter_dialog import QCFilterDialog
import hashlib
from pathlib import Path
class Slide2FileSelection(QWidget):
    def __init__(self):
        super().__init__()
        self.table = FileTableWidget()  # TABLE_COLUMNS defined in table.py
        self.open_dialogs = []
        self.log_dialog = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Импорт файлов")
        self.add_button.clicked.connect(self.add_files)
        btn_layout.addWidget(self.add_button)
        
        self.import_bids_button = QPushButton("Импорт BIDS")
        self.import_bids_button.clicked.connect(self.import_bids_dataset)
        btn_layout.addWidget(self.import_bids_button)

        self.filter_button = QPushButton("Настроить фильтр QC")
        self.filter_button.clicked.connect(self.open_filter_dialog)
        btn_layout.addWidget(self.filter_button)
        
        self.delete_button = QPushButton("Delete Selected Files")
        self.delete_button.clicked.connect(self.delete_selected_files)
        btn_layout.addWidget(self.delete_button)

        self.delete_cache_button = QPushButton("Delete Computed Data")
        self.delete_cache_button.clicked.connect(self.delete_cache)
        btn_layout.addWidget(self.delete_cache_button)

        self.view_logs_button = QPushButton("View Logs")
        self.view_logs_button.clicked.connect(self.show_log_dialog)
        btn_layout.addWidget(self.view_logs_button)

        layout.addLayout(btn_layout)
        layout.addWidget(self.table)
        
        run_layout = QHBoxLayout()
        self.run_qc_button = QPushButton("Запустить QC")
        self.run_qc_button.clicked.connect(self.start_qc_processing)
        run_layout.addWidget(self.run_qc_button)
        
        self.run_preproc_button = QPushButton("Запустить Preprocessing")
        self.run_preproc_button.clicked.connect(self.start_preprocessing)
        run_layout.addWidget(self.run_preproc_button)
        
        layout.addLayout(run_layout)
        self.installEventFilter(self)
        self.table.cell_clicked.connect(self.handle_cell_clicked)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите ЭЭГ-файлы", "",
            "Поддерживаемые (*.vhdr *.edf *.set);;Все файлы (*)"
        )
        for file_path in files:
            from gui.table import normalize_path
            if normalize_path(file_path) in self.table.data_files:
                QMessageBox.warning(
                    self, "Предупреждение", f"Файл {file_path} уже добавлен."
                )
                continue
            try:
                file_info = EEGFileManager.read_file_info(file_path)
                data = [
                    os.path.basename(file_path),
                    file_info["duration"],
                    file_info["sfreq"],
                    file_info["n_channels"],
                    "View",
                    "None",
                    "None"
                ]
                self.table.add_row(file_path, data)
                self.table.set_channels(file_path, file_info["ch_names"])
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Не удалось добавить файл {file_path}: {e}"
                )

    def import_bids_dataset(self):
        bids_root = QFileDialog.getExistingDirectory(self, "Выберите BIDS-директорию", "")
        if not bids_root:
            return
        files = EEGFileManager.import_bids(bids_root)
        for file_path in files:
            if file_path in self.table.data_files:
                QMessageBox.warning(
                    self, "Предупреждение", f"Файл {file_path} уже добавлен."
                )
                continue
            try:
                file_info = EEGFileManager.read_file_info(file_path)
                data = [
                    os.path.basename(file_path),
                    file_info["duration"],
                    file_info["sfreq"],
                    file_info["n_channels"],
                    "View",
                    "None",
                    "None"
                ]
                self.table.add_row(file_path, data)
                self.table.set_channels(file_path, file_info["ch_names"])
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Не удалось добавить файл {file_path}: {e}"
                )

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            self.delete_selected_files()
        return super().eventFilter(source, event)

    def delete_selected_files(self):
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        if not selected_rows:
            return
        msg = QMessageBox()
        msg.setWindowTitle("Подтверждение удаления")
        msg.setText(f"Удалить {len(selected_rows)} файлов?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if msg.exec_() == QMessageBox.Yes:
            self.table.remove_selected_rows()
            self.table.remove_empty_columns()

    def delete_cache(self):
        msg = QMessageBox()
        msg.setWindowTitle("Подтверждение удаления")
        msg.setText("Удалить все кэшированные данные?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if msg.exec_() == QMessageBox.Yes:
            try:
                import shutil
                from gui.utils import Worker
                shutil.rmtree(Worker.cache_dir, ignore_errors=True)
                os.makedirs(Worker.cache_dir)
                self.table.logs.clear()
                for row in range(self.table.rowCount()):
                    self.table.set_elem(row, 5, "None", align=Qt.AlignCenter)
                QMessageBox.information(self, "Успех", "Кэш успешно удалён!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить кэш: {e}")

    def show_log_dialog(self):
        if self.log_dialog is None:
            self.log_dialog = LogDialog(self.table.get_logs(), self)
            # Привязываем сигнал обновления логов
            self.table.log_updated.connect(self.log_dialog.update_entry)
            self.log_dialog.finished.connect(self.clear_log_dialog)
        self.log_dialog.show()

    def clear_log_dialog(self):
        self.log_dialog = None

    def open_filter_dialog(self):
        qc_keys = [key for key in self.table.report_columns if key.startswith("QC_")]
        dialog = QCFilterDialog(qc_keys, self.table.get_qc_thresholds())
        if dialog.exec_():
            self.table.set_qc_thresholds(dialog.get_thresholds())
            self.apply_qc_filter()

    def apply_qc_filter(self):
        self.table.update_qc_filter_status()

    def start_qc_processing(self):
        files_to_process = [
            self.table.data_files[i] for i in range(self.table.rowCount())]
        if not files_to_process:
            print("⚠ Нет файлов для QC.")
            return
        print("🔍 Запуск QC для:", files_to_process)
        from gui.main_window import MainWindow
        from gui.slides.slide3_qc import Slide3QC
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[2], Slide3QC):
            mw.slides[2].graph_manager.input_files = files_to_process
            mw.slides[2].start_processing()

    def start_preprocessing(self):
        files_to_process = [
            self.table.data_files[i] for i in range(self.table.rowCount())
            if self.table.item(i, 5).text() == "✔"
        ]
        if not files_to_process:
            print("⚠ Нет файлов, прошедших QC.")
            return
        print("⚙ Запуск Preprocessing для:", files_to_process)
        from gui.main_window import MainWindow
        from gui.slides.slide4_preprocessing import Slide4Preprocessing
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[3], Slide4Preprocessing):
            mw.slides[3].set_input_files(files_to_process)
            mw.slides[3].start_processing()

    def handle_cell_clicked(self, row, col, header):
        file_path = self.table.data_files[row]
        if header == "Список каналов":
            channels = self.table.get_channels(file_path)
            if not channels:
                try:
                    raw = mne.io.read_raw(file_path, preload=False)
                    channels = raw.ch_names
                    self.table.set_channels(file_path, channels)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить каналы: {e}")
                    return
            dialog = ChannelListDialog(channels, self)
            dialog.show()
            self.open_dialogs.append(dialog)
            dialog.finished.connect(lambda: self.open_dialogs.remove(dialog))
        elif self.table.item(row, col).text() == "View":
            value = self.table.get_data(file_path, header)
            if isinstance(value, Figure):
                dialog = ImageViewDialog(figure=value, parent=self)
                dialog.exec_()
            else:
                QMessageBox.critical(self, "Ошибка", f"График не найден для {header}")

    def update_reports(self, reports_per_file, transform_names, logs):
        self.table.set_log(logs)
        new_columns = set()

        # Store new reports in table
        for file_path, report in reports_per_file.items():
            for t_name, data in report.items():
                for key, value in data.items():
                    col_name = f"{t_name}_{key}"
                    new_columns.add(col_name)
                    self.table.set_data(file_path, col_name, value)

        # Add new columns to table
        for col_name in new_columns - set(self.table.report_columns):
            col_index = self.table.add_column(col_name, align=Qt.AlignLeft)
            self.table.report_columns[col_name] = col_index

        # Update table with all reports
        for row, file_path in enumerate(self.table.data_files):
            report = self.table.get_report(file_path)
            for t_name, data in report.items():
                for key, value in data.items():
                    col_name = f"{t_name}_{key}"
                    if col_name in self.table.report_columns:
                        if isinstance(value, Figure):
                            self.table.set_elem(row, self.table.report_columns[col_name], "View", Qt.AlignCenter, is_link=True)
                        else:
                            self.table.set_elem(row, self.table.report_columns[col_name], value, Qt.AlignLeft)

    def clear(self):
        for dialog in self.open_dialogs[:]:
            dialog.close()
        self.open_dialogs.clear()
        self.table.clear_data()

    def serialize(self):
        return {
            "files": self.table.data_files,
            "thresholds": self.table.qc_thresholds,
            "report_columns": self.table.report_columns,
            "logs": self.table.logs,
            "table_data": [
                {
                    self.table.horizontalHeaderItem(col).text(): (
                        {"type": "figure"} if isinstance(self.table.get_data(self.table.data_files[row], self.table.horizontalHeaderItem(col).text()), Figure)
                        else {"type": "value", "data": self.table.item(row, col).text()}
                    )
                    for col in range(self.table.columnCount())
                    if self.table.item(row, col)
                }
                for row in range(self.table.rowCount())
            ]
        }

    def deserialize(self, data):
        self.clear()
        from gui.table import normalize_path

        self.table.data_files = data.get("files", [])
        self.table.qc_thresholds = data.get("thresholds", {})
        self.table.report_columns = data.get("report_columns", {})

        if "QC Фильтр" not in self.table.report_columns:
            col_index = self.table.columnCount()
            self.table.report_columns["QC Фильтр"] = col_index
        
        self.table.logs = data.get("logs", {})

        initial_columns = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount()) if self.table.horizontalHeaderItem(i)]
        self.table.setColumnCount(len(initial_columns) + len(self.table.report_columns))
        headers = initial_columns + list(self.table.report_columns.keys())
        self.table.setHorizontalHeaderLabels(headers)

        base_dir = None
        if self.table.data_files:
            base_dir = os.path.dirname(self.table.data_files[0])

        for row_data in data.get("table_data", []):
            file_name_info = row_data.get("Имя файла", {})
            if isinstance(file_name_info, dict):
                file_name = file_name_info.get("data", "")
            else:
                file_name = file_name_info

            if not file_name:
                continue

            candidates = [p for p in self.table.data_files if os.path.basename(p) == file_name]

            if candidates:
                file_path = candidates[0]
            elif base_dir:
                file_path = normalize_path(os.path.join(base_dir, file_name))
                if file_path not in self.table.data_files:
                    self.table.data_files.append(file_path)
            else:
                continue

            if file_path not in self.table.data_files:
                row = self.table.add_row(file_path)
            else:
                row = self.table.data_files.index(file_path)
                if row >= self.table.rowCount():
                    self.table.insertRow(row)

            qc_filter_value = row_data.get("QC Фильтр", {})
            if isinstance(qc_filter_value, dict):
                value = qc_filter_value.get("data", "None")
            else:
                value = qc_filter_value or "None"

            qc_status_value = row_data.get("QC Статус", {})
            if isinstance(qc_status_value, dict):
                value = qc_status_value.get("data", "None")
            else:
                value = qc_status_value or "None"
            align = Qt.AlignCenter
            self.table.set_elem(row, 5, value, align=align)

            align = Qt.AlignCenter
            col_qc_filter = self.table.report_columns["QC Фильтр"]
            self.table.set_elem(row, col_qc_filter, value, align=align)

            for col, header in enumerate(headers):
                value_info = row_data.get(header, {})
                if isinstance(value_info, dict):
                    value_type = value_info.get("type", "value")
                    if value_type == "figure":
                        # График
                        from gui.save_project import load_figure_pickle
                        project_dir = Path(self.window().project_dir)
                        figures_dir = project_dir / "figures"
                        
                        fpath_normalized = normalize_path(file_path)
                        uid = hashlib.sha1(fpath_normalized.encode()).hexdigest()[:8]
                        fig_path = figures_dir / f"{uid}/{header}.pkl"
                        if fig_path.exists():
                            fig = load_figure_pickle(fig_path)
                            self.table.set_data(file_path, header, fig)
                            align = Qt.AlignCenter
                            self.table.set_elem(row, col, "View", align, is_link=True)
                    else:
                        value = value_info.get("data", "")
                        align = Qt.AlignCenter if col in (4, 5, 6) else Qt.AlignLeft
                        is_link = value == "View"
                        self.table.set_elem(row, col, value, align, is_link=is_link)

                else:
                    value = value_info
                    align = Qt.AlignCenter if col in (4, 5, 6) else Qt.AlignLeft
                    is_link = value == "View"
                    self.table.set_elem(row, col, value, align, is_link=is_link)