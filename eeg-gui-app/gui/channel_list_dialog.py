import json
from pathlib import Path
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QDialogButtonBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from gui.json_dialog import JsonDialog

class ChannelListDialog(QDialog):
    def __init__(self, channels, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.setWindowTitle("Список каналов")
        self.setMinimumWidth(300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Имена каналов:"))
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.channels)
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.NoContextMenu)
        layout.addWidget(self.list_widget)
        button_layout = QHBoxLayout()
        json_button = QPushButton("Просмотр пустого шаблона JSON для переименования каналов")
        json_button.clicked.connect(self.create_json)
        button_layout.addWidget(json_button)
        ok_button = QDialogButtonBox(QDialogButtonBox.Ok)
        ok_button.accepted.connect(self.accept)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

    def create_json(self):
        json_data = {channel: "" for channel in self.channels}
        dialog = JsonDialog(json_data, self)
        dialog.show()