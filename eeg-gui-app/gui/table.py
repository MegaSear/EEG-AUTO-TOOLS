from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os

TABLE_COLUMNS = [
    {"label": "Имя файла", "align": Qt.AlignLeft},
    {"label": "Длительность\n(сек)", "align": Qt.AlignLeft},
    {"label": "Частота дискретизации\n(Гц)", "align": Qt.AlignLeft},
    {"label": "Количество каналов", "align": Qt.AlignLeft},
    {"label": "Список каналов", "align": Qt.AlignCenter},
    {"label": "QC статус", "align": Qt.AlignCenter},
    {"label": "Preprocessing статус", "align": Qt.AlignCenter},
    {"label": "QC фильтр", "align": Qt.AlignCenter}
]
base_columns = [TABLE_COLUMNS[i]["label"] for i in range(8)]

def normalize_path(path):
    """Нормализовать путь (абсолютный + нормализованный + без чувствительности к регистру для Windows)."""
    return os.path.normcase(os.path.abspath(path))

class FileTableWidget(QTableWidget):
    cell_clicked = pyqtSignal(int, int, str)
    log_updated = pyqtSignal(str, int, str, dict, str)

    def __init__(self, columns=TABLE_COLUMNS):
        super().__init__()
        self.data_files = []  # List of normalized file paths
        self.data_storage = {}  # {file_path: {col_name: value}} for Figures, channels, etc.
        self.qc_thresholds = {}
        self.report_columns = {}
        self.logs = {}
        self.setup_table(columns)
        self.itemClicked.connect(self.handle_item_clicked)

    def setup_table(self, columns):
        if columns:
            self.setColumnCount(len(columns))
            self.setHorizontalHeaderLabels([col["label"] for col in columns])
        else:
            self.setColumnCount(0)

        self.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionMode(QAbstractItemView.MultiSelection)

    def set_elem(self, row, col, data, align=Qt.AlignLeft, is_link=False):
        item = QTableWidgetItem(str(data))
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        item.setTextAlignment(align)
        if is_link:
            font = QFont()
            font.setUnderline(True)
            item.setFont(font)
            item.setForeground(QColor('blue'))
        self.setItem(row, col, item)

    def has_file(self, file_path):
        norm_path = normalize_path(file_path)
        return any(normalize_path(p) == norm_path for p in self.data_files)

    def add_row(self, file_path, data=None):
        if self.has_file(file_path):
            return None

        row = self.rowCount()
        self.insertRow(row)
        self.data_files.append(file_path)

        if file_path not in self.data_storage:
            self.data_storage[file_path] = {}

        if data:
            for col, value in enumerate(data):
                if col < self.columnCount():
                    align = Qt.AlignCenter if col >= self.columnCount() - 2 else Qt.AlignLeft
                    is_link = value == "View"
                    self.set_elem(row, col, value, align, is_link=is_link)

        return row

    def add_column(self, label, align=Qt.AlignCenter):
        col = self.columnCount()
        self.setColumnCount(col + 1)
        self.setHorizontalHeaderLabels(
            [self.horizontalHeaderItem(i).text() if self.horizontalHeaderItem(i) else "" for i in range(col)] + [label]
        )
        self.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        for row in range(self.rowCount()):
            self.set_elem(row, col, "", align)
        return col

    def remove_file(self, file_path):
        """Убрать файл из data_files, data_storage, logs и таблицы."""
        norm_path = normalize_path(file_path)

        # Найти index по нормализованному пути
        for idx, existing_path in enumerate(self.data_files):
            if normalize_path(existing_path) == norm_path:
                break
        else:
            return  # Файл не найден → ничего не делаем

        # Удаляем строку таблицы
        self.removeRow(idx)
        self.data_files.pop(idx)

        # Удаляем данные
        if existing_path in self.data_storage:
            for col_name, value in self.data_storage[existing_path].items():
                if isinstance(value, Figure):
                    value.close()
            del self.data_storage[existing_path]

        if existing_path in self.logs:
            del self.logs[existing_path]

            
    def remove_selected_rows(self):
        selected_rows = sorted(set(index.row() for index in self.selectedIndexes()), reverse=True)
        for row in selected_rows:
            if row < len(self.data_files):
                file_path = self.data_files[row]
                self.remove_file(file_path)


    def handle_item_clicked(self, item):
        if item.text() == "View":
            row = item.row()
            col = item.column()
            header = self.horizontalHeaderItem(col).text()
            self.cell_clicked.emit(row, col, header)

    def set_data(self, file_path, col_name, value):
        file_path = normalize_path(file_path)
        if file_path not in self.data_storage:
            self.data_storage[file_path] = {}
        self.data_storage[file_path][col_name] = value

    def get_data(self, file_path, col_name):
        file_path = normalize_path(file_path)
        return self.data_storage.get(file_path, {}).get(col_name, None)

    def get_report(self, file_path):
        file_path = normalize_path(file_path)
        report = {}
        for col_name, value in self.data_storage.get(file_path, {}).items():
            if col_name == "channels":
                continue
            parts = col_name.rsplit("_", 1)
            if len(parts) == 2:
                t_name, key = parts
            else:
                t_name, key = col_name, ""
            if t_name not in report:
                report[t_name] = {}
            report[t_name][key] = value
        return report

    def get_channels(self, file_path):
        file_path = normalize_path(file_path)
        return self.data_storage.get(file_path, {}).get("channels", [])

    def set_channels(self, file_path, channels):
        file_path = normalize_path(file_path)
        self.set_data(file_path, "channels", channels)

    def set_qc_thresholds(self, thresholds):
        self.qc_thresholds = thresholds

    def get_qc_thresholds(self):
        return self.qc_thresholds

    def set_log(self, logs):
        self.logs.update(logs)

    def get_logs(self):
        return self.logs
    
    def clear_log(self):
        self.logs.clear()
        
    def evaluate_qc_status(self, row):
        try:
            n_channels = float(self.item(row, 3).text()) or 1
            for key, threshold in self.qc_thresholds.items():
                if key in self.report_columns:
                    value = self._get_column_value(row, key, n_channels)
                    if value > threshold:
                        return "✖"
            return "✔"
        except Exception:
            return "✖"

    def update_qc_filter_status(self):
        for row in range(self.rowCount()):
            qc_status = self.item(row, 5).text()
            if qc_status != "✔":
                self.set_elem(row, 7, "None", align=Qt.AlignCenter)
                continue

            n_channels = float(self.item(row, 3).text()) or 1
            for key, threshold in self.qc_thresholds.items():
                if key in self.report_columns:
                    value = self._get_column_value(row, key, n_channels)
                    if value > threshold:
                        self.set_elem(row, 7, "✖", align=Qt.AlignCenter)
                        break
            else:
                self.set_elem(row, 7, "✔", align=Qt.AlignCenter)

    def _get_column_value(self, row, key, n_channels):
        col = self.report_columns[key]
        value = float(self.item(row, col).text()) if self.item(row, col) and self.item(row, col).text().replace('.', '', 1).isdigit() else float('inf')
        return value / n_channels * 100 if "channels" in key.lower() else value

    def update_log_entry(self, file, path_id, node, params, status):
        if file not in self.logs:
            self.logs[file] = {}
        if path_id not in self.logs[file]:
            self.logs[file][path_id] = {}
        self.logs[file][path_id][node] = {"params": params, "status": status}
        self.log_updated.emit(file, path_id, node, params, status)

    def clear_data(self):
        for file_path in self.data_storage:
            for col_name, value in self.data_storage[file_path].items():
                if isinstance(value, Figure):
                    plt.close(value)
        self.data_storage.clear()
        self.data_files.clear()
        self.qc_thresholds.clear()
        self.report_columns.clear()
        self.logs.clear()
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(len(TABLE_COLUMNS))
        self.setHorizontalHeaderLabels([col["label"] for col in TABLE_COLUMNS])


    def remove_empty_columns(self):
        keep_columns = base_columns
        columns_to_remove = []
        
        for col in reversed(range(self.columnCount())):
            header = self.horizontalHeaderItem(col).text()
            if header in keep_columns:
                continue  # базовые не трогаем

            all_empty = True
            for row in range(self.rowCount()):
                item = self.item(row, col)
                if item is not None and item.text().strip() != "":
                    all_empty = False
                    break

            if all_empty:
                columns_to_remove.append(col)

        for col in columns_to_remove:
            self.removeColumn(col)

            headers = list(self.report_columns.keys())
            for header in headers:
                if self.report_columns[header] == col:
                    del self.report_columns[header]
                    break

            # Смещаем индексы оставшихся report_columns
            for header in self.report_columns:
                if self.report_columns[header] > col:
                    self.report_columns[header] -= 1