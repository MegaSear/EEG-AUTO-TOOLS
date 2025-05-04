from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QGraphicsView, QGraphicsScene, QSizePolicy, QGraphicsEllipseItem, QGraphicsPathItem
)
from PyQt5.QtGui import QPainterPath, QPen
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal, QPointF
from gui.graph import GraphManager
from gui.utils import Worker

class Slide3QC(QWidget):
    def __init__(self, cache_dir):
        super().__init__()
        self.scene = QGraphicsScene()
        self.graph_manager = GraphManager(self.scene)
        self.connecting = False
        self.temp_line = None
        self.start_port = None
        self.is_processing = False
        self.cache_dir = cache_dir
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        self.left_panel = QScrollArea()
        self.left_panel.setWidgetResizable(True)
        button_container = QWidget()
        left_layout = QVBoxLayout(button_container)
        left_layout.addWidget(QLabel("Блоки для QC", alignment=Qt.AlignCenter, styleSheet="font-weight: bold; font-size: 14px;"))
        for name, cls in self.graph_manager.available_transforms.items():
            btn = QPushButton(name)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _, n=name, c=cls: self.graph_manager.spawn_block(n, c))
            left_layout.addWidget(btn)
        left_layout.addStretch()
        self.left_panel.setWidget(button_container)
        main_layout.addWidget(self.left_panel, 1)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 4)
        right_layout.addWidget(QLabel("Конфигуратор Quality Check (QC)", alignment=Qt.AlignCenter, styleSheet="font-size: 18px; font-weight: bold;"))
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        right_layout.addWidget(self.view)
        self.progress_label = QLabel("Готов к обработке", alignment=Qt.AlignCenter, styleSheet="font-size: 14px; color: gray;")
        right_layout.addWidget(self.progress_label)
        self.graph_manager.create_input_block()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = self.view.mapToScene(event.pos())
            for item in self.scene.items(pos):
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
            for item in self.scene.items(pos):
                if isinstance(item, QGraphicsEllipseItem) and item != self.start_port:
                    start_block = self.start_port.parentItem()
                    end_block = item.parentItem()
                    if (self.start_port == start_block.output_port and item == end_block.input_port and start_block != end_block):
                        self.graph_manager.add_edge(self.start_port, item)
                        print(f"Создано ребро: {start_block.name} -> {end_block.name}")
                    else:
                        print(f"⚠ Некорректное соединение: {start_block.name} -> {end_block.name}")
                    break
            self.scene.removeItem(self.temp_line)
            self.temp_line = None
            self.connecting = False
            self.start_port = None
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene.selectedItems():
                self.graph_manager.remove_item(item)

    def start_processing(self):
        if self.is_processing:
            self.progress_label.setText("⚠ Обработка уже выполняется!")
            return
        
        from gui.slides.slide2_file_selection import Slide2FileSelection
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[1], Slide2FileSelection):
            current_files = mw.slides[1].table.data_files
            mw.slides[1].table.clear_log()
        else:
            current_files = []
        self.graph_manager.input_files = current_files

        print(f"🔍 Количество блоков: {len(self.graph_manager.blocks)}")
        print(f"🔍 Количество рёбер: {len(self.graph_manager.edges)}")
        print(f"🔍 Блоки: {[block.name for block in self.graph_manager.blocks]}")
        print(f"🔍 Рёбра: {[(e.start_item.parentItem().name, e.end_item.parentItem().name) for e in self.graph_manager.edges]}")
        if len(self.graph_manager.blocks) <= 1 or not self.graph_manager.edges:
            self.progress_label.setText("⚠ Граф пуст или не содержит рёбер.")
            return
        self.is_processing = True
        self.progress_label.setText("Инициализация обработки...")
        self.thread = QThread()
        self.worker = Worker(self.graph_manager, cache_dir=self.cache_dir)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.update_log)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def update_progress(self, message):
        print(message)
        self.progress_label.setText(message)

    def update_log(self, log_entry):
        from gui.slides.slide2_file_selection import Slide2FileSelection
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[1], Slide2FileSelection):
            file = log_entry["file"]
            path_id = log_entry["path_id"]
            node = log_entry["node"]
            params = log_entry["params"]
            status = log_entry["status"]
            table = mw.slides[1].table
            table.update_log_entry(file, path_id, node, params, status)


    def on_processing_finished(self, reports_per_file, transform_names, logs):
        self.progress_label.setText("Обработка завершена!")
        self.is_processing = False
        from gui.slides.slide2_file_selection import Slide2FileSelection
        mw = self.window()
        if hasattr(mw, 'slides') and isinstance(mw.slides[1], Slide2FileSelection):
            qc_reports_per_file = {}
            for file_path, report in reports_per_file.items():
                qc_reports_per_file[file_path] = {}
                for t_name, data in report.items():
                    qc_t_name = f"QC_{t_name}"
                    if isinstance(data, dict):
                        qc_reports_per_file[file_path][qc_t_name] = data
                    else:
                        qc_reports_per_file[file_path][qc_t_name] = data
            mw.slides[1].table.set_log(logs)  # Save logs to FileTableWidget
            mw.slides[1].update_reports(qc_reports_per_file, transform_names, logs)

        # После update_reports
        for row, file_path in enumerate(mw.slides[1].table.data_files):
            if file_path in reports_per_file:
                mw.slides[1].table.set_elem(row, 5, "✔", align=Qt.AlignCenter)

        self.thread.quit()
        self.thread.wait()

    def on_processing_error(self, error_message):
        print(error_message)
        self.progress_label.setText(error_message)
        self.is_processing = False
        self.thread.quit()
        self.thread.wait()

    def clear(self):
        self.scene.clear()
        self.graph_manager = GraphManager(self.scene)
        self.connecting = False
        self.temp_line = None
        self.start_port = None
        self.is_processing = False
        self.progress_label.setText("Готов к обработке")

    def serialize(self):
        return self.graph_manager.serialize()

    def deserialize(self, data):
        self.clear()
        self.graph_manager.deserialize(data)