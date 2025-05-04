import sys
from pathlib import Path
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog

class StartupWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)
        self._chosen_path = None
        self._is_new_project = False

        btn_new = QPushButton("Создать новый проект")
        btn_load = QPushButton("Загрузить проект…")

        btn_new.clicked.connect(self._new_project)
        btn_load.clicked.connect(self._load_clicked)

        layout.addWidget(btn_new)
        layout.addWidget(btn_load)

    def _new_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Укажите имя новой папки для проекта", "")
        if path:
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True)
                print(f"LOG: Создана новая папка проекта: {path}")
                sys.stdout.flush()
            self._chosen_path = str(path)
            self._is_new_project = True
            self.accept()  # Только если реально выбрали
        else:
            print("LOG: Создание проекта отменено")
            sys.stdout.flush()

    def _load_clicked(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку проекта для загрузки")
        if path:
            self._chosen_path = path
            self._is_new_project = False
            self.accept()  # Только если реально выбрали
        else:
            print("LOG: Загрузка проекта отменена")
            sys.stdout.flush()

    @property
    def chosen(self):
        return self._chosen_path

    @property
    def project_dir(self):
        return self._chosen_path

    @property
    def is_new_project(self):
        return self._is_new_project
