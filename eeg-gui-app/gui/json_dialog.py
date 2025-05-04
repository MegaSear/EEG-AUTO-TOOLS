import json
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

class JsonDialog(QDialog):
    def __init__(self, json_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JSON для переименования каналов")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("JSON с текущими каналами (значения пустые):"))
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setText(json.dumps(json_data, indent=2, ensure_ascii=False))
        layout.addWidget(self.text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)