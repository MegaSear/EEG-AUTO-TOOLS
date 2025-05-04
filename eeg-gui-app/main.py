import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.startup_window import StartupWindow

def main():
    app = QApplication(sys.argv)
    start_window = StartupWindow()

    if not start_window.exec_():
        # Нажал крестик или Отмена -> выход
        print("LOG: Пользователь закрыл стартовое окно. Завершение.")
        sys.exit()

    main_window = MainWindow(project_dir=start_window.project_dir)

    if start_window.chosen:
        if start_window.is_new_project:
            print(f"LOG: Создан новый проект в {start_window.chosen}")
        else:
            print(f"LOG: Загрузка проекта из {start_window.chosen}")
            main_window.load_project(start_window.chosen)

    print("LOG: Запуск главного окна приложения")
    main_window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
