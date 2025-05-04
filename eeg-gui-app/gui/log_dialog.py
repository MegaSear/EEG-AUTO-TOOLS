import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QTreeWidget, QTreeWidgetItem, QHeaderView
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont, QColor

class LogDialog(QDialog):
    def __init__(self, logs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Logs")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Path", "Transform", "Parameters", "Status"])
        header = self.tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)   # :contentReference[oaicite:0]{index=0}
        header.setStretchLastSection(False)
        layout.addWidget(self.tree)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.populate_tree(logs)
        QTimer.singleShot(0, self._fit_to_contents)

    def _fit_to_contents(self):
        """Подгоняем ширину диалога под сумму видимых столбцов + полоса прокрутки."""
        self.tree.doItemsLayout()                                 
        h = self.tree.header()

        total = sum(h.sectionSize(i) for i in range(self.tree.columnCount()))
        total += self.tree.verticalScrollBar().sizeHint().width()  # учесть скролл‑бар
        total += self.layout().contentsMargins().left() + self.layout().contentsMargins().right()

        self.setMinimumWidth(total)
        self.resize(total, self.height())

    def populate_tree(self, logs):
        for file, paths in logs.items():
            file_item = QTreeWidgetItem(self.tree, [os.path.basename(file), "", "", "", ""])
            for path_id, nodes in paths.items():
                path_item = QTreeWidgetItem(file_item, ["", f"Path {path_id}", "", "", ""])
                for node, info in nodes.items():
                    QTreeWidgetItem(path_item, ["", "", str(node), str(info["params"]), info["status"]])
                    
    def update_entry(self, file, path_id, node, params, status):
        for i in range(self.tree.topLevelItemCount()):
            file_item = self.tree.topLevelItem(i)
            if file_item.text(0) == os.path.basename(file):
                for j in range(file_item.childCount()):
                    path_item = file_item.child(j)
                    if path_item.text(1) == f"Path {path_id}":
                        for k in range(path_item.childCount()):
                            node_item = path_item.child(k)
                            if node_item.text(2) == node:
                                # обновить параметры и статус
                                node_item.setText(3, str(params))
                                node_item.setText(4, status)
                                self._fit_to_contents()
                                return