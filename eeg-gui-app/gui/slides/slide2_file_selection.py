from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QLabel, QAbstractItemView
)
from PyQt5.QtCore import Qt
import mne
from gui.slides.slide3_qc import Slide3QC
from gui.slides.slide4_preprocessing import Slide4Preprocessing
from PyQt5.QtWidgets import QSizePolicy  # в начало файла, если ещё не импортировано
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import os

class QCFilterDialog(QDialog):
    def __init__(self, thresholds=None):
        super().__init__()
        self.setWindowTitle("Настройка фильтра QC")
        self.thresholds = thresholds or {}
        layout = QFormLayout(self)
        self.inputs = {}

        fields = [
            "Количество плохих каналов",
            "Количество мостиковых каналов",
            "Количество высокоамплитудных",
            "Количество низкоамплитудных",
            "Количество шумных каналов"
        ]

        for field in fields:
            value = self.thresholds.get(field, "")
            input_field = QLineEdit(str(value))
            layout.addRow(field, input_field)
            self.inputs[field] = input_field

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_thresholds(self):
        return {
            k: int(v.text()) if v.text().isdigit() else float('inf')
            for k, v in self.inputs.items()
        }


class Slide2FileSelection(QWidget):
    def __init__(self):
        super().__init__()
        self.files = []
        self.thresholds = {}
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
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Имя файла", "Длительность\n(сек)", "Частота дискретизации\n(Гц)", "Количество каналов",
            "QC", "Предобработка", "Количество\nмостиковых",
            "Количество\nвысокоамплитудных", "Количество\nнизкоамплитудных",
            "Количество шумных", "Итоговое количество\nплохих", "QC результат"
        ])
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.table)

        run_buttons_layout = QHBoxLayout()

        self.run_qc_button = QPushButton("Запустить QC")
        self.run_qc_button.clicked.connect(self.run_qc_for_selected)
        run_buttons_layout.addWidget(self.run_qc_button)

        self.run_preproc_button = QPushButton("Запустить Preprocessing")
        self.run_preproc_button.clicked.connect(self.run_preproc_for_selected)
        run_buttons_layout.addWidget(self.run_preproc_button)

        layout.addLayout(run_buttons_layout)

    def bids_path_exists(self, bids_path):
        try:
            _ = bids_path.fpath  # fpath вызывает ошибку, если путь невалидный
            return bids_path.fpath.exists()
        except Exception:
            return False
        
    def import_bids_dataset(self):
        bids_root = QFileDialog.getExistingDirectory(self, "Выберите BIDS-директорию", "")
        if not bids_root:
            return

        subjects = get_entity_vals(bids_root, 'subject')
        tasks = get_entity_vals(bids_root, 'task')
        all_runs = get_entity_vals(bids_root, 'run')  # без фильтров

        for subject in subjects:
            for task in tasks:
                # отберём только те run, что реально существуют для текущего subject-task
                runs = []
                for run in all_runs:
                    bids_path = BIDSPath(subject=subject, task=task, run=run, root=bids_root)
                    if self.bids_path_exists(bids_path):
                        runs.append(run)

                if not runs:
                    runs = [None]  # если run не задан

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


    def get_selected_file_paths(self):
        paths = []
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 6)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    paths.append(self.files[row])
        return paths

    def run_qc_for_selected(self):
        selected_files = self.get_selected_file_paths()
        print("🔍 Запуск QC для:", selected_files)

    def run_preproc_for_selected(self):
        selected_files = self.get_selected_file_paths()
        print("⚙ Запуск Preprocessing для:", selected_files)

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

        set_item(0, os.path.basename(str(file_path)))
        set_item(1, duration)
        set_item(2, sfreq)
        set_item(3, n_channels)
        set_item(4, "None")
        set_item(5, "None")

        # Метрики QC (заполняются позже)
        for col in range(7, 12):
            set_item(col, "None")  # пустые значения до QC

    def open_filter_dialog(self):
        dialog = QCFilterDialog(self.thresholds)
        if dialog.exec_():
            self.thresholds = dialog.get_thresholds()
            self.apply_qc_filter()

    def apply_qc_filter(self):
        for row in range(self.table.rowCount()):
            try:
                values = {
                    key: int(self.table.item(row, col).text())
                    for key, col in zip(self.thresholds.keys(), range(7, 12))
                }
                passed = all(values[k] <= self.thresholds.get(k, float('inf')) for k in values)
                result = "✔" if passed else "✖"
            except Exception:
                result = "✖"

            item = QTableWidgetItem(result)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 11, item)

    def run_qc_for_selected(self):
        files_to_qc = [f for i, f in enumerate(self.files)
                    if self.table.item(i, 4).text() == "None"]  # QC ещё не проводился
        print("🔍 Запуск QC для:", files_to_qc)

        from gui.main_window import MainWindow  # Импорт временный, можно будет передать окно через сигнал
        mw = self.window()  # Получаем главное окно
        if hasattr(mw, 'slides') and isinstance(mw.slides[2], Slide3QC):
            mw.slides[2].set_input_files(files_to_qc)

    def run_preproc_for_selected(self):
        files_to_proc = []
        for i, f in enumerate(self.files):
            result_item = self.table.item(i, 11)
            if result_item and result_item.text() == "✔":
                files_to_proc.append(f)

        print("⚙ Запуск Preprocessing для:", files_to_proc)

        from gui.main_window import MainWindow
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[3], Slide4Preprocessing):
            mw.slides[3].set_input_files(files_to_proc)
