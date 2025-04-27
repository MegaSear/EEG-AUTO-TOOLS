# gui/slides/slide3_qc.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QDialog, QFormLayout,
    QLineEdit, QDialogButtonBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QEvent, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QBrush, QColor, QPen, QPainterPath, QKeyEvent
import inspect
import sys
import os
import mne
import numpy as np
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, lib_path)
from eeg_auto_tools.transforms import Transform, Sequence
import eeg_auto_tools.transforms as transforms_module

def get_available_transforms():
    transforms = {}
    for name, cls in inspect.getmembers(transforms_module, inspect.isclass):
        if issubclass(cls, Transform) and cls is not Transform and cls.__module__ == transforms_module.__name__:
            transforms[name] = cls
    return transforms

class TransformBlock(QGraphicsRectItem):
    def __init__(self, name, transform_class, block_id=None):
        super().__init__(0, 0, 180, 60)
        self.block_id = block_id if block_id is not None else id(self)  # Уникальный идентификатор блока
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
        control1 = QPointF(start_pos.x(), start_pos.y() + 40)
        control2 = QPointF(end_pos.x(), end_pos.y() - 40)
        path.cubicTo(control1, control2, end_pos)
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.blue if self.isSelected() else Qt.black, 2)
        painter.setPen(pen)
        painter.drawPath(self.path())

class Worker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict, dict)
    error = pyqtSignal(str)

    def __init__(self, slide):
        super().__init__()
        self.slide = slide

    def run(self):
        try:
            if not hasattr(self.slide, 'input_files'):
                self.error.emit("⚠ Файлы для обработки не заданы.")
                return

            graph = {block: [] for block in self.slide.blocks}
            reverse_graph = {block: [] for block in self.slide.blocks}
            for edge in self.slide.edges:
                start = edge.start_item.parentItem()
                end = edge.end_item.parentItem()
                graph[start].append(end)
                reverse_graph[end].append(start)

            self.slide._check_connectivity(graph)
            self.slide._check_no_cycles(graph)

            transform_names = self.slide._assign_transform_names(graph)
            paths = self.slide._find_all_paths(graph)
            self.progress.emit(f"🧩 Найдено {len(paths)} путей для обработки.")

            reports_per_file = {}
            self.slide.cache = {}

            total_files = len(self.slide.input_files)
            for file_idx, file in enumerate(self.slide.input_files):
                self.progress.emit(f"⚙ Обработка файла: {file} ({file_idx + 1}/{total_files})")
                raw = mne.io.read_raw(file, preload=True, verbose=False)
                self.slide.cache[file] = {}
                file_report = {}
                processed_nodes = set()

                def process_node(node, current_raw):
                    if node in processed_nodes:
                        cache_file = self.slide.cache[file].get(transform_names.get(node, "InputRaw"))
                        if cache_file and os.path.exists(cache_file):
                            return self.slide._load_cached_raw(cache_file)
                        return current_raw
                    processed_nodes.add(node)
                    if node == self.slide.input_block:
                        cache_file = self.slide._cache_raw(current_raw, file, "InputRaw")
                        self.slide.cache[file]["InputRaw"] = cache_file
                        return current_raw
                    t_name = transform_names[node]
                    transform = node.transform_class(**node.params)
                    processed = transform(current_raw.copy())
                    repo_data, _ = transform.get_report()
                    if repo_data:
                        file_report[t_name] = repo_data
                    if len(graph[node]) > 1 or not graph[node]:
                        cache_file = self.slide._cache_raw(processed, file, t_name)
                        self.slide.cache[file][t_name] = cache_file
                    return processed

                for path in paths:
                    current_raw = raw
                    for node in path:
                        current_raw = process_node(node, current_raw)

                reports_per_file[file] = file_report
                self.progress.emit(f"✅ Файл {file} успешно обработан.")

            self.finished.emit(reports_per_file, transform_names)

        except ValueError as e:
            self.error.emit(f"[!] Ошибка в графе: {e}")
        except Exception as e:
            self.error.emit(f"[!] Ошибка при обработке: {e}")

class Slide3QC(QWidget):
    def __init__(self):
        super().__init__()
        self.available_transforms = get_available_transforms()
        self.blocks = []
        self.edges = []
        self.connecting = False
        self.temp_line = None
        self.start_port = None
        self.cache_dir = "cache_qc"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.is_processing = False
        self.cache = {}  # Для хранения путей к кэшам
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.left_panel = QScrollArea()
        self.left_panel.setWidgetResizable(True)
        button_container = QWidget()
        self.left_layout = QVBoxLayout(button_container)
        title_label = QLabel("Блоки для QC")
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
        title = QLabel("Конфигуратор Quality Check (QC)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(title)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        right_layout.addWidget(self.view)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)

        self.progress_label = QLabel("Готов к обработке")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14px; color: gray;")
        right_layout.addWidget(self.progress_label)

        self.run_button = QPushButton("▶ Запустить QC-граф")
        self.run_button.clicked.connect(self.start_processing)
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
                    print(f"Начало соединения: {self.start_port.parentItem().name} (порт: {'output' if self.start_port == self.start_port.parentItem().output_port else 'input'})")
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
                    start_block = self.start_port.parentItem()
                    end_block = item.parentItem()
                    if (self.start_port == start_block.output_port and
                            item == end_block.input_port and
                            start_block != end_block):
                        edge = Edge(self.start_port, item)
                        self.scene.addItem(edge)
                        self.edges.append(edge)
                        print(f"Создано ребро: {start_block.name} -> {end_block.name}")
                    else:
                        print(f"⚠ Некорректное соединение: {start_block.name} (порт: {'output' if self.start_port == start_block.output_port else 'input'}) -> {end_block.name} (порт: {'output' if item == end_block.output_port else 'input'})")
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
            print(f"[!] Несвязные узлы: {[block.name for block in self.blocks if block not in visited]}")
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

    def _cache_raw(self, raw, file_path, node_name):
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}_{node_name}.npz")
        np.savez_compressed(cache_file, data=raw.get_data(), info=np.array([raw.info], dtype=object))
        return cache_file

    def _load_cached_raw(self, cache_file):
        with np.load(cache_file, allow_pickle=True) as data:
            raw_data = data['data']
            info = data['info'][0]
            return mne.io.RawArray(raw_data, info, verbose=False)

    def start_processing(self):
        if self.is_processing:
            print("⚠ Обработка уже выполняется!")
            return

        print(f"🔍 Количество блоков: {len(self.blocks)}")
        print(f"🔍 Количество рёбер: {len(self.edges)}")
        print(f"🔍 Блоки: {[block.name for block in self.blocks]}")
        print(f"🔍 Рёбра: {[(edge.start_item.parentItem().name, edge.end_item.parentItem().name) for edge in self.edges]}")

        if len(self.blocks) <= 1 or not self.edges:
            self.progress_label.setText("⚠ Граф пуст или не содержит рёбер. Добавьте блоки и связи.")
            print("⚠ Граф пуст или не содержит рёбер. Добавьте блоки и связи.")
            return

        self.is_processing = True
        self.run_button.setEnabled(False)
        self.progress_label.setText("Инициализация обработки...")

        self.thread = QThread()
        self.worker = Worker(self)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.thread.started.connect(self.worker.run)

        self.thread.start()

    def update_progress(self, message):
        print(message)
        self.progress_label.setText(message)

    def on_processing_finished(self, reports_per_file, transform_names):
        self.progress_label.setText("Обработка завершена!")
        self.is_processing = False
        self.run_button.setEnabled(True)

        from gui.slides.slide2_file_selection import Slide2FileSelection
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[1], Slide2FileSelection):
            mw.slides[1].update_reports(reports_per_file, transform_names)

        self.thread.quit()
        self.thread.wait()

    def on_processing_error(self, error_message):
        print(error_message)
        self.progress_label.setText(error_message)
        self.is_processing = False
        self.run_button.setEnabled(True)

        self.thread.quit()
        self.thread.wait()

    def clear(self):
        """Очищает все данные слайда, не добавляя InputRaw."""
        self.scene.clear()
        self.blocks.clear()
        self.edges.clear()
        self.cache.clear()
        self.connecting = False
        self.temp_line = None
        self.start_port = None
        self.is_processing = False
        self.run_button.setEnabled(True)
        self.progress_label.setText("Готов к обработке")
        self.input_block = None  # Удаляем ссылку на input_block

    def serialize(self):
        """Сериализует данные слайда для сохранения в файл."""
        # Сохраняем блоки с их уникальными идентификаторами
        blocks_data = []
        for block in self.blocks:
            block_data = {
                "block_id": block.block_id,  # Сохраняем уникальный идентификатор
                "name": block.name,
                "pos": {"x": block.pos().x(), "y": block.pos().y()},
                "params": block.params
            }
            blocks_data.append(block_data)

        # Сохраняем рёбра, используя block_id для идентификации
        edges_data = []
        for edge in self.edges:
            edge_data = {
                "start_block_id": edge.start_item.parentItem().block_id,
                "start_port": "output",
                "end_block_id": edge.end_item.parentItem().block_id,
                "end_port": "input"
            }
            edges_data.append(edge_data)

        # Сохраняем кэши
        cache_data = self.cache

        return {
            "blocks": blocks_data,
            "edges": edges_data,
            "cache": cache_data,
            "cache_dir": self.cache_dir
        }

    def deserialize(self, data):
        """Загружает данные слайда из словаря."""
        self.clear()

        # Загружаем блоки
        self.available_transforms = get_available_transforms()
        block_id_to_block = {}  # Словарь для сопоставления block_id с объектами блоков
        for block_data in data.get("blocks", []):
            name = block_data["name"]
            block_id = block_data["block_id"]
            transform_class = self.available_transforms.get(name, Transform) if name != "InputRaw" else Transform
            block = TransformBlock(name, transform_class, block_id=block_id)
            block.setPos(QPointF(block_data["pos"]["x"], block_data["pos"]["y"]))
            block.params = block_data["params"]
            self.scene.addItem(block)
            self.blocks.append(block)
            block_id_to_block[block_id] = block
            if name == "InputRaw":
                self.input_block = block
                self.input_block.setBrush(QBrush(QColor("#D5F5E3")))
                self.input_block.setFlag(QGraphicsItem.ItemIsMovable, False)
                self.input_block.setFlag(QGraphicsItem.ItemIsSelectable, False)

        # Если InputRaw не был найден в сохранённых данных, создаём новый
        if not self.input_block:
            self.input_block = TransformBlock("InputRaw", Transform)
            self.input_block.setBrush(QBrush(QColor("#D5F5E3")))
            self.input_block.setPos(QPointF(300, 100))
            self.input_block.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.input_block.setFlag(QGraphicsItem.ItemIsSelectable, False)
            self.scene.addItem(self.input_block)
            self.blocks.append(self.input_block)

        # Загружаем рёбра, используя block_id для корректного восстановления
        for edge_data in data.get("edges", []):
            start_block_id = edge_data["start_block_id"]
            end_block_id = edge_data["end_block_id"]
            start_block = block_id_to_block.get(start_block_id)
            end_block = block_id_to_block.get(end_block_id)
            if start_block and end_block:
                start_port = start_block.output_port
                end_port = end_block.input_port
                if start_port and end_port:
                    edge = Edge(start_port, end_port)
                    self.scene.addItem(edge)
                    self.edges.append(edge)
                else:
                    print(f"⚠ Не удалось восстановить ребро: {start_block.name} -> {end_block.name} (порты недоступны)")

        # Загружаем кэши
        self.cache = data.get("cache", {})
        self.cache_dir = data.get("cache_dir", "cache_qc")
        os.makedirs(self.cache_dir, exist_ok=True)