from PyQt5.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QGraphicsItem, QPushButton,
    QFileDialog, QMessageBox, QComboBox, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QFrame
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QBrush, QColor, QPen, QPainterPath
import inspect
import os
import sys
import json
import csv
import mne
from pathlib import Path

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, lib_path)
from eeg_auto_tools.transforms import Transform
import eeg_auto_tools.transforms as transforms_module
from eeg_auto_tools.montages import read_elc, create_custom_montage

COLORS = {
    "input_block": QColor("#D5F5E3"),
    "transform_block": QColor("#AED6F1"),
    "selected": Qt.blue,
    "default": Qt.black,
    "temp_line": Qt.red
}

def get_available_transforms():
    return {
        name: cls for name, cls in inspect.getmembers(transforms_module, inspect.isclass)
        if issubclass(cls, Transform) and cls is not Transform and cls.__module__ == transforms_module.__name__
    }


class TransformBlock(QGraphicsRectItem):
    def __init__(self, name, transform_class, block_id=None):
        super().__init__(0, 0, 180, 60)
        self.block_id = block_id or id(self)
        self.name = name
        self.transform_class = transform_class
        self.params = self._get_default_params()
        self._setup_ui(name)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

    def _get_default_params(self):
        if hasattr(self.transform_class, "get_parameters_schema"):
            return {k: v["default"] for k, v in self.transform_class.get_parameters_schema().items()}
        return {}

    def _setup_ui(self, name):
        self.setBrush(QBrush(COLORS["input_block" if name == "InputRaw" else "transform_block"]))
        self.text = QGraphicsTextItem("Input" if name == "InputRaw" else name, self)
        self.text.setDefaultTextColor(Qt.darkGreen if name == "InputRaw" else Qt.black)
        self.text.setPos(10, 20)
        self.input_port = None if name == "InputRaw" else QGraphicsEllipseItem(self.rect().width() / 2 - 5, -10, 10, 10, self)
        if self.input_port:
            self.input_port.setBrush(QBrush(COLORS["default"]))
            self.input_port.setFlag(QGraphicsItem.ItemIsSelectable)
        self.output_port = QGraphicsEllipseItem(self.rect().width() / 2 - 5, self.rect().height(), 10, 10, self)
        self.output_port.setBrush(QBrush(COLORS["default"]))
        self.output_port.setFlag(QGraphicsItem.ItemIsSelectable)

    def mouseDoubleClickEvent(self, event):
        if not hasattr(self, '_param_dialog') or self._param_dialog is None:
            self._param_dialog = TransformParamDialog(self.name, self.params)
            self._param_dialog.setAttribute(Qt.WA_DeleteOnClose)
            self._param_dialog.accepted.connect(lambda: self.set_params(self._param_dialog.get_params()))
            self._param_dialog.destroyed.connect(lambda: setattr(self, '_param_dialog', None))
            self._param_dialog.show()
        else:
            self._param_dialog.raise_()
            self._param_dialog.activateWindow()
        super().mouseDoubleClickEvent(event)


    def set_params(self, params):
        self.params = params

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            for edge in self.scene().items():
                if isinstance(edge, Edge) and (edge.start_item.parentItem() == self or edge.end_item.parentItem() == self):
                    edge.update_path()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        pen = QPen(COLORS["selected"] if self.isSelected() else COLORS["default"], 2)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())

class TransformParamDialog(QDialog):
    def __init__(self, name, params):
        super().__init__()
        self.setWindowTitle(f"Параметры {name}:")
        self.inputs = {}
        self.name = name
        self.plotter = None
        self._setup_ui(params)

    def _setup_ui(self, params):
        main_layout = QHBoxLayout(self)

        transform_class = get_available_transforms().get(self.name, Transform)
        schema = transform_class.get_parameters_schema() if hasattr(transform_class, "get_parameters_schema") else {}
        description = getattr(transform_class, "DESCRIPTION", None)

        if description:
            description_frame = QFrame()
            description_frame.setStyleSheet("""
                QFrame {
                    background-color: #f8f8f8;
                    border: 1px solid #ccc;
                    border-left: 5px solid #409eff;
                    padding: 0px 0px 0px 0px;
                    border-radius: 4px;
                }
            """)

            description_layout = QVBoxLayout(description_frame)
            description_layout.setContentsMargins(0, 0, 0, 0)
            description_layout.setSpacing(0)

            # Заголовок "Описание:"
            title_label = QLabel("Описание:")
            title_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    font-size: 14px;
                    color: #409eff;
                }
            """)
            title_label.setFrameStyle(QFrame.NoFrame)
            title_label.setAttribute(Qt.WA_TranslucentBackground)
            description_layout.addWidget(title_label)

            # Текст описания
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setStyleSheet("""
                QLabel {
                    color: #222;
                    font-size: 12px;
                    line-height: 150%;
                    padding-left: 12px;
                }
            """)
            description_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            description_label.setFrameStyle(QFrame.NoFrame)
            description_label.setAttribute(Qt.WA_TranslucentBackground)
            description_layout.addWidget(description_label)

            # Сделаем контейнер немного шире
            description_frame.setMinimumWidth(380)

            main_layout.addWidget(description_frame)

        form_widget = QWidget()
        layout = QFormLayout(form_widget)
        for key, value in params.items():
            param_info = schema.get(key, {"type": type(value), "default": value})
            param_type = param_info.get("type")

            if self.name == "SetMontage" and key == "montage":
                # Специальная обработка для монтажа
                self.standard_montage = QComboBox()
                for choice in param_info.get("choices", []):
                    self.standard_montage.addItem(str(choice))
                self.standard_montage.setCurrentText(str(value))
                layout.addRow("Montage", self.standard_montage)

                # Монтажный режим
                self.montage_mode = QComboBox()
                self.montage_mode.addItems(["Standard Montage", "ELC File"])
                layout.addRow("Режим монтажа", self.montage_mode)

                self.montage_mode.currentTextChanged.connect(self._update_montage_mode_ui)

            elif self.name == "SetMontage" and key == "elc_file":
                # Поле для выбора elc файла
                elc_layout = QHBoxLayout()
                self.elc_line = QLineEdit(str(value) if value else "")
                elc_browse = QPushButton("...")
                elc_browse.setFixedWidth(30)

                def browse_elc():
                    file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать .elc файл", "", "ELC файлы (*.elc)")
                    if file_path:
                        self.elc_line.setText(file_path)
                        print(f"Выбран elc файл: {file_path}")

                elc_browse.clicked.connect(browse_elc)
                elc_layout.addWidget(self.elc_line)
                elc_layout.addWidget(elc_browse)
                elc_container = QWidget()
                elc_container.setLayout(elc_layout)
                layout.addRow("ELC File", elc_container)
                self.elc_container = elc_container  # сохраним чтобы скрывать/показывать

            else:
                # Обычные параметры
                if param_type == "choice" and "choices" in param_info:
                    combo = QComboBox()
                    for choice in param_info["choices"]:
                        combo.addItem(str(choice))
                    combo.setCurrentText(str(value))
                    layout.addRow(key, combo)
                    self.inputs[key] = combo
                elif param_type == bool:
                    combo = QComboBox()
                    combo.addItems(["True", "False"])
                    combo.setCurrentText(str(value))
                    layout.addRow(key, combo)
                    self.inputs[key] = combo
                elif param_type == "file":
                    file_layout = QHBoxLayout()
                    line_edit = QLineEdit(str(value) if value else "")
                    browse_button = QPushButton("...")
                    browse_button.setFixedWidth(30)

                    def make_browse(line_edit_widget):
                        def browse_file():
                            file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать файл", "", "Все файлы (*)")
                            if file_path:
                                line_edit_widget.setText(file_path)
                                print(f"Выбран файл: {file_path}")
                        return browse_file

                    browse_button.clicked.connect(make_browse(line_edit))
                    file_layout.addWidget(line_edit)
                    file_layout.addWidget(browse_button)
                    container = QWidget()
                    container.setLayout(file_layout)
                    layout.addRow(key, container)
                    self.inputs[key] = line_edit
                else:
                    line_edit = QLineEdit(str(value))
                    layout.addRow(key, line_edit)
                    self.inputs[key] = line_edit

        if self.name == "SetMontage":
            # 3D Preview
            check_button = QPushButton("Check Montage")
            check_button.clicked.connect(self.visualize_montage)
            layout.addRow(check_button)

            self.viz_widget = QWidget()
            self.viz_widget.setLayout(QVBoxLayout())
            layout.addRow(self.viz_widget)

            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self.viz_widget)
            self.viz_widget.layout().addWidget(self.plotter)
            self.plotter.set_background('white')

            self._update_montage_mode_ui()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        main_layout.addWidget(form_widget)

    def _update_montage_mode_ui(self):
        if hasattr(self, 'elc_container'):
            if self.montage_mode.currentText() == "ELC File":
                self.standard_montage.hide()
                self.elc_container.show()
            else:
                self.standard_montage.show()
                self.elc_container.hide()

    def visualize_montage(self):
        if self.plotter is None:
            return
        try:
            import mne
            import pyvista
            import numpy as np

            if self.montage_mode.currentText() == "ELC File":
                elc_file = self.elc_line.text()
                if not elc_file:
                    QMessageBox.critical(self, "Ошибка", "Файл ELC не выбран!")
                    return
                ch_dict, nasion, lpa, rpa, hsp = read_elc(elc_file)
                montage = mne.channels.make_dig_montage(ch_pos=ch_dict, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame='head')
            elif self.standard_montage.currentText() == "waveguard64":
                montage = create_custom_montage("waveguard64")
            else:
                montage = mne.channels.make_standard_montage(self.standard_montage.currentText())

            self.plotter.clear()
            pos = montage.get_positions()
            if pos and 'ch_pos' in pos:
                coords = np.array(list(pos['ch_pos'].values()))
                point_cloud = pyvista.PolyData(coords)
                glyph = point_cloud.glyph(scale=False, geom=pyvista.Sphere(radius=0.003), orient=False)
                self.plotter.add_mesh(glyph, color='red', smooth_shading=True)
                labels = list(pos['ch_pos'].keys())
                self.plotter.add_point_labels(coords, labels, point_size=10, font_size=14, always_visible=True, shape=None, text_color='black')

            self.plotter.reset_camera()
            self.plotter.update()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось визуализировать монтаж: {e}")

    def get_params(self):
        params = {}
        transform_class = get_available_transforms().get(self.name, Transform)
        schema = transform_class.get_parameters_schema() if hasattr(transform_class, "get_parameters_schema") else {}

        for key, field in self.inputs.items():
            param_info = schema.get(key, {})
            param_type = param_info.get("type")

            if isinstance(field, QComboBox):
                val = field.currentText()
                if param_type == bool:
                    val = val == "True"
                elif param_type == "choice":
                    choices = param_info.get("choices", [])
                    if all(isinstance(c, (int, float)) for c in choices):
                        val = float(val) if "." in val else int(val)
                params[key] = val
            else:
                text = field.text()
                try:
                    if param_type == int:
                        params[key] = int(text)
                    elif param_type == float:
                        params[key] = float(text)
                    elif param_type == bool:
                        params[key] = text.lower() == "true"
                    elif param_type == tuple:
                        params[key] = eval(text)
                    else:
                        params[key] = text
                except Exception:
                    params[key] = text

        if self.name == "SetMontage":
            if self.montage_mode.currentText() == "ELC File":
                params["montage"] = "personal"
                params["elc_file"] = self.elc_line.text()
            else:
                params["montage"] = self.standard_montage.currentText()
                params["elc_file"] = None

        return params

    
    def load_mapping(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл маппинга", "",
            "JSON файлы (*.json);;CSV файлы (*.csv);;Все файлы (*)"
        )
        if not file_path:
            return
        try:
            mapping = self._read_mapping_file(file_path)
            self.inputs["channel_mapping"].setText(str(mapping))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить маппинг: {e}")

    def _read_mapping_file(self, file_path):
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                if not isinstance(mapping, dict):
                    raise ValueError("JSON файл должен содержать словарь")
                return mapping
        elif file_path.endswith('.csv'):
            mapping = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) < 2:
                    raise ValueError("CSV файл должен содержать два столбца: old_name, new_name")
                for row in reader:
                    if len(row) >= 2:
                        mapping[row[0].strip()] = row[1].strip()
                return mapping
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте JSON или CSV.")

class Edge(QGraphicsPathItem):
    def __init__(self, start_item, end_item):
        super().__init__()
        self.start_item = start_item
        self.end_item = end_item
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.update_path()

    def update_path(self):
        start_pos = self.start_item.mapToScene(self.start_item.boundingRect().center())
        end_pos = self.end_item.mapToScene(self.end_item.boundingRect().center())
        path = QPainterPath(start_pos)
        control1 = QPointF(start_pos.x(), start_pos.y() + 40)
        control2 = QPointF(end_pos.x(), end_pos.y() - 40)
        path.cubicTo(control1, control2, end_pos)
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        pen = QPen(COLORS["selected"] if self.isSelected() else COLORS["default"], 2)
        painter.setPen(pen)
        painter.drawPath(self.path())

class GraphManager:
    def __init__(self, scene):
        self.scene = scene
        self.available_transforms = get_available_transforms()
        self.blocks = []
        self.edges = []
        self.input_block = None

    def spawn_block(self, name, transform_class):
        block = TransformBlock(name, transform_class)
        block.setPos(QPointF(40 * len(self.blocks), 40 * len(self.blocks)))
        self.scene.addItem(block)
        self.blocks.append(block)
        return block

    def create_input_block(self):
        self.input_block = TransformBlock("InputRaw", Transform)
        self.input_block.setBrush(QBrush(COLORS["input_block"]))
        self.scene.addItem(self.input_block)
        self.input_block.setPos(QPointF(300, 100))
        self.input_block.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.input_block.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.blocks.append(self.input_block)
        return self.input_block

    def add_edge(self, start_port, end_port):
        edge = Edge(start_port, end_port)
        self.scene.addItem(edge)
        self.edges.append(edge)
        return edge

    def remove_item(self, item):
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

    def _build_graph(self):
        graph = {block: [] for block in self.blocks}
        reverse_graph = {block: [] for block in self.blocks}
        for edge in self.edges:
            start = edge.start_item.parentItem()
            end = edge.end_item.parentItem()
            graph[start].append(end)
            reverse_graph[end].append(start)
        return graph, reverse_graph

    def _check_connectivity(self, graph):
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor)
        dfs(self.input_block)
        if len(visited) != len(self.blocks):
            raise ValueError(f"Граф несвязный: посещено {len(visited)} из {len(self.blocks)} узлов.")

    def _check_no_cycles(self, graph):
        visited = set()
        rec_stack = set()
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False
        if dfs(self.input_block):
            raise ValueError("Обнаружен цикл в графе!")

    def _assign_transform_names(self, graph):
        transform_names = {}
        visited = set()
        counter = 1
        def dfs(node):
            nonlocal counter
            if node in visited:
                return
            visited.add(node)
            if node != self.input_block:
                transform_names[node] = f"QC_{node.name}_{counter}"
                counter += 1
            for neighbor in graph.get(node, []):
                dfs(neighbor)
        dfs(self.input_block)
        return transform_names

    def _find_all_paths(self, graph):
        paths = []
        path = []
        def dfs(node):
            path.append(node)
            neighbors = graph.get(node, [])
            if not neighbors:
                paths.append(path[:])
            else:
                for neighbor in neighbors:
                    dfs(neighbor)
            path.pop()
        dfs(self.input_block)
        return paths

    def serialize(self):
        blocks_data = [
            {
                "block_id": block.block_id,
                "name": block.name,
                "pos": {"x": block.pos().x(), "y": block.pos().y()},
                "params": block.params
            }
            for block in self.blocks
        ]
        edges_data = [
            {
                "start_block_id": edge.start_item.parentItem().block_id,
                "start_port": "output",
                "end_block_id": edge.end_item.parentItem().block_id,
                "end_port": "input"
            }
            for edge in self.edges
        ]
        return {
            "blocks": blocks_data,
            "edges": edges_data,
        }

    def deserialize(self, data):
        self.blocks.clear()
        self.edges.clear()
        self.input_block = None
        block_id_to_block = {}
        for block_data in data.get("blocks", []):
            name = block_data["name"]
            block_id = block_data["block_id"]
            transform_class = self.available_transforms.get(name, Transform) if name != "InputRaw" else Transform
            block = TransformBlock(name, transform_class, block_id=block_id)
            self.scene.addItem(block)
            block.setPos(QPointF(block_data["pos"]["x"], block_data["pos"]["y"]))
            block.params = block_data["params"]
            self.blocks.append(block)
            block_id_to_block[block_id] = block
            if name == "InputRaw":
                self.input_block = block
                self.input_block.setBrush(QBrush(COLORS["input_block"]))
                self.input_block.setFlag(QGraphicsItem.ItemIsMovable, False)
                self.input_block.setFlag(QGraphicsItem.ItemIsSelectable, False)
        if not self.input_block:
            self.create_input_block()
        for edge_data in data.get("edges", []):
            start_block = block_id_to_block.get(edge_data["start_block_id"])
            end_block = block_id_to_block.get(edge_data["end_block_id"])
            if start_block and end_block and start_block.output_port and end_block.input_port:
                edge = Edge(start_block.output_port, end_block.input_port)
                self.scene.addItem(edge)
                self.edges.append(edge)
            else:
                print(f"⚠ Не удалось восстановить ребро: {start_block.name if start_block else 'None'} -> {end_block.name if end_block else 'None'}")
