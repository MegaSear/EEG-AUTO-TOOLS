import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageViewDialog(QDialog):
    def __init__(self, figure=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Просмотр графика")
        layout = QVBoxLayout(self)
        if figure:
            try:
                canvas = FigureCanvas(figure)
                canvas.setMinimumSize(800, 600)
                layout.addWidget(canvas)
            except Exception as e:
                layout.addWidget(QLabel(f"Ошибка отображения графика: {e}"))
        else:
            layout.addWidget(QLabel("График не найден"))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)