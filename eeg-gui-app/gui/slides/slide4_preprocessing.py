from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QDialog, QFormLayout,
    QLineEdit, QDialogButtonBox, QScrollArea, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QEvent
from PyQt5.QtGui import QBrush, QColor, QPen, QPainterPath, QKeyEvent

import inspect
import eeg_auto_tools.transforms as transforms_module
from eeg_auto_tools.transforms import Transform
from eeg_auto_tools.transforms import Sequence
import mne


def get_available_transforms():
    transforms = {}
    for name, cls in inspect.getmembers(transforms_module, inspect.isclass):
        if (issubclass(cls, Transform) and cls is not Transform and cls.__module__ == transforms_module.__name__):
            transforms[name] = cls
    return transforms


class TransformBlock(QGraphicsRectItem):
    def __init__(self, name, transform_class):
        super().__init__(0, 0, 180, 60)
        self.setBrush(QBrush(QColor("#AED6F1")))
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        self.name = name
        self.transform_class = transform_class
        self.params = {
            k: v.default if v.default is not inspect.Parameter.empty else ""
            for k, v in inspect.signature(transform_class.__init__).parameters.items()
            if k != "self"
        }

        self.text = QGraphicsTextItem(name, self)
        self.text.setDefaultTextColor(Qt.black)
        self.text.setPos(10, 20)

        if name != "InputRaw":
            self.input_port = QGraphicsEllipseItem(self.rect().width() / 2 - 5, -10, 10, 10, self)
            self.input_port.setBrush(QBrush(Qt.black))
            self.input_port.setFlag(QGraphicsItem.ItemIsSelectable)
        else:
            self.input_port = None

        self.output_port = QGraphicsEllipseItem(self.rect().width() / 2 - 5, self.rect().height(), 10, 10, self)
        self.output_port.setBrush(QBrush(Qt.black))
        self.output_port.setFlag(QGraphicsItem.ItemIsSelectable)

        if name == "InputRaw":
            self.text.setDefaultTextColor(Qt.darkGreen)
            self.text.setPlainText("Input")

    def mouseDoubleClickEvent(self, event):
        dialog = TransformParamDialog(self.name, self.params)
        if dialog.exec_():
            self.params = dialog.get_params()
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            scene = self.scene()
            if scene:
                for edge in scene.items():
                    if isinstance(edge, Edge):
                        if edge.start_item.parentItem() == self or edge.end_item.parentItem() == self:
                            edge.update_path()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.blue if self.isSelected() else Qt.black, 2)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())


class TransformParamDialog(QDialog):
    def __init__(self, name, params):
        super().__init__()
        self.setWindowTitle(f"Параметры: {name}")
        layout = QFormLayout(self)
        self.inputs = {}
        for key, value in params.items():
            line_edit = QLineEdit(str(value))
            layout.addRow(key, line_edit)
            self.inputs[key] = line_edit
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            key: self._parse_value(field.text())
            for key, field in self.inputs.items()
        }

    def _parse_value(self, text):
        try:
            return eval(text)
        except Exception:
            return text


class Edge(QGraphicsPathItem):
    def __init__(self, start_item, end_item):
        super().__init__()
        self.start_item = start_item
        self.end_item = end_item
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.update_path()

    def update_path(self):
        start_pos = self.start_item.mapToScene(self.start_item.boundingRect().center())
        end_pos = self.end_item.mapToScene(self.end_item.boundingRect().center())
        path = QPainterPath(start_pos)
        dx = (end_pos.x() - start_pos.x()) / 2
        control1 = QPointF(start_pos.x(), start_pos.y() + 40)
        control2 = QPointF(end_pos.x(), end_pos.y() - 40)
        path.cubicTo(control1, control2, end_pos)
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.blue if self.isSelected() else Qt.black, 2)
        painter.setPen(pen)
        painter.drawPath(self.path())


class Slide4Preprocessing(QWidget):
    def __init__(self):
        super().__init__()
        self.available_transforms = get_available_transforms()
        self.blocks = []
        self.edges = []
        self.connecting = False
        self.temp_line = None
        self.start_port = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.left_panel = QScrollArea()
        self.left_panel.setWidgetResizable(True)
        button_container = QWidget()
        self.left_layout = QVBoxLayout(button_container)

        title_label = QLabel("Блоки для Preprocessing")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.left_layout.addWidget(title_label)

        for name, cls in self.available_transforms.items():
            btn = QPushButton(name)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _, name=name, cls=cls: self.spawn_block(name, cls))
            self.left_layout.addWidget(btn)

        self.left_layout.addStretch()
        self.left_panel.setWidget(button_container)
        main_layout.addWidget(self.left_panel, 1)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 4)

        title = QLabel("Конфигуратор Preprocessing")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(title)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        right_layout.addWidget(self.view)

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)

        self.run_button = QPushButton("▶ Запустить Preprocessing-граф")
        self.run_button.clicked.connect(self.run_qc_pipeline)
        right_layout.addWidget(self.run_button)

        self.input_block = TransformBlock("InputRaw", Transform)
        self.input_block.setBrush(QBrush(QColor("#D5F5E3")))
        self.input_block.setPos(QPointF(300, 100))
        self.input_block.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.input_block.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.scene.addItem(self.input_block)
        self.blocks.append(self.input_block)

    def set_input_files(self, files):
        self.input_files = files

    def spawn_block(self, name, transform_class):
        block = TransformBlock(name, transform_class)
        block.setPos(QPointF(40 * len(self.blocks), 40 * len(self.blocks)))
        self.scene.addItem(block)
        self.blocks.append(block)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = self.view.mapToScene(event.pos())
            items = self.scene.items(pos)
            for item in items:
                if isinstance(item, QGraphicsEllipseItem):
                    self.start_port = item
                    self.connecting = True
                    self.temp_line = QGraphicsPathItem()
                    self.temp_line.setPen(QPen(Qt.red, 2, Qt.DashLine))
                    self.scene.addItem(self.temp_line)
                    break
        elif event.type() == QEvent.MouseMove and self.connecting:
            pos = self.view.mapToScene(event.pos())
            path = QPainterPath(self.start_port.mapToScene(self.start_port.boundingRect().center()))
            control = QPointF((path.currentPosition().x() + pos.x()) / 2, path.currentPosition().y() + 40)
            path.cubicTo(control, QPointF(pos.x(), pos.y() - 40), pos)
            self.temp_line.setPath(path)
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.connecting:
            pos = self.view.mapToScene(event.pos())
            items = self.scene.items(pos)
            for item in items:
                if isinstance(item, QGraphicsEllipseItem) and item != self.start_port:
                    edge = Edge(self.start_port, item)
                    self.scene.addItem(edge)
                    self.edges.append(edge)
                    break
            self.scene.removeItem(self.temp_line)
            self.temp_line = None
            self.connecting = False
            self.start_port = None
        return super().eventFilter(source, event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete:
            for item in self.scene.selectedItems():
                if isinstance(item, TransformBlock) and item != self.input_block:
                    for edge in self.edges[:]:
                        if edge.start_item.parentItem() == item or edge.end_item.parentItem() == item:
                            self.scene.removeItem(edge)
                            self.edges.remove(edge)
                    self.scene.removeItem(item)
                    self.blocks.remove(item)
                elif isinstance(item, Edge):
                    self.scene.removeItem(item)
                    self.edges.remove(item)

    def build_transform_sequence(self):
        graph = {block: [] for block in self.blocks}
        for edge in self.edges:
            start = edge.start_item.parentItem()
            end = edge.end_item.parentItem()
            graph[start].append(end)

        visited = set()
        sorted_blocks = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            if node != self.input_block:
                sorted_blocks.insert(0, node)

        dfs(self.input_block)
        return sorted_blocks

    def run_qc_pipeline(self):
        ordered_blocks = self.build_transform_sequence()
        transform_objects = {}

        for i, block in enumerate(ordered_blocks):
            try:
                kwargs = block.params
                transform = block.transform_class(**kwargs)
                transform_objects[f"{block.name}_{i}"] = transform
            except Exception as e:
                print(f"[!] Ошибка при создании {block.name}: {e}")
                continue

        sequence = Sequence(**transform_objects)

        if not hasattr(self, 'input_files'):
            print("⚠ Файлы для обработки не заданы.")
            return

        for file in self.input_files:
            try:
                print(f"⚙ Обработка: {file}")
                raw = mne.io.read_raw(file, preload=True, verbose=False)
                processed = sequence(raw)
                print(f"✅ Успешно: {file}")
            except Exception as e:
                print(f"[!] Ошибка при обработке {file}: {e}")
