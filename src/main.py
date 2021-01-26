import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QFile, QTranslator
from resources.windows.mainwindow import Ui_Form


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load("hellotr_la")
    app.installTranslator(translator)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
