from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QCheckBox, QLineEdit, QComboBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt

class QCFilterDialog(QDialog):
    OPERATORS = ["≤", "≥", "==", "!="]

    def __init__(self, available_keys, thresholds=None):
        super().__init__()
        self.setWindowTitle("Настройка фильтра QC")
        self.available_keys = available_keys
        self.thresholds = thresholds or {}

        self.controls = {}  # key -> dict: {enabled, value, op}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Активировать"), 0, 0)
        grid.addWidget(QLabel("Параметр"), 0, 1)
        grid.addWidget(QLabel("Порог"), 0, 2)
        grid.addWidget(QLabel("Условие"), 0, 3)

        for row, key in enumerate(self.available_keys, start=1):
            # Чекбокс
            check = QCheckBox()
            check.setChecked(key in self.thresholds)

            # Название параметра
            label = QLabel(key)

            # Значение порога
            threshold_value = self.thresholds.get(key, "")
            edit = QLineEdit(str(threshold_value))

            # Оператор
            op = QComboBox()
            op.addItems(self.OPERATORS)
            op.setCurrentIndex(0)

            grid.addWidget(check, row, 0)
            grid.addWidget(label, row, 1)
            grid.addWidget(edit, row, 2)
            grid.addWidget(op, row, 3)

            self.controls[key] = {
                "enabled": check,
                "value": edit,
                "operator": op
            }

        layout.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_thresholds(self):
        result = {}
        for key, widgets in self.controls.items():
            if not widgets["enabled"].isChecked():
                continue

            value_text = widgets["value"].text().strip()
            if not value_text:
                continue

            try:
                value = float(value_text)
            except ValueError:
                continue

            result[key] = {
                "value": value,
                "operator": widgets["operator"].currentText()
            }

        return result
