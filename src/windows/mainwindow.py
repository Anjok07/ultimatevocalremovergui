# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'mainwindow.ui'
##
# Created by: Qt User Interface Compiler version 6.0.0
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(960, 550)
        MainWindow.setMinimumSize(QSize(0, 0))
        MainWindow.setStyleSheet(u"/* Universal */\n"
                                 "* {\n"
                                 "	font: 15pt \"Yu Gothic UI\";	\n"
                                 "	color: rgb(255, 255, 255);\n"
                                 "	background-color: none;\n"
                                 "	background: rgb(12, 23, 40);\n"
                                 "}\n"
                                 "/* Window */\n"
                                 "QWidget#Form, QFrame, QPushButton {\n"
                                 "	background-color: rgb(12, 23, 40);\n"
                                 "}\n"
                                 "/* Frames */\n"
                                 "QFrame {\n"
                                 "\n"
                                 "}\n"
                                 "/* Button */\n"
                                 "QPushButton {\n"
                                 "	border-width: 2px;\n"
                                 "	border-style: solid;\n"
                                 "	border-radius: 25px;\n"
                                 "	border-color: rgb(109, 213, 237);\n"
                                 "	background-color: rgba(109, 213, 237, 4);\n"
                                 "}\n"
                                 "QPushButton:hover {\n"
                                 "	background-color: rgba(109, 213, 237, 10);\n"
                                 "}\n"
                                 "QPushButton:pressed {\n"
                                 "	background-color: rgba(109, 213, 237, 30);\n"
                                 "}")
        self.gridLayout = QGridLayout(MainWindow)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(5, 5, 5, 0)
        self.frame_2 = QFrame(MainWindow)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QSize(650, 0))
        self.frame_2.setMaximumSize(QSize(700, 16777215))
        self.frame_2.setStyleSheet(u"")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_2)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(30, 30, 30, 30)
        self.frame_4 = QFrame(self.frame_2)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy1)
        self.frame_4.setStyleSheet(u"QFrame {\n"
                                   "	border-width: 3px;\n"
                                   "    border-style: dotted;\n"
                                   "    border-color: rgb(160, 160, 160);\n"
                                   "	border-radius: 5px;\n"
                                   "}")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.frame_4.setLineWidth(1)
        self.verticalLayout_2 = QVBoxLayout(self.frame_4)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_musicFiles = QLabel(self.frame_4)
        self.label_musicFiles.setObjectName(u"label_musicFiles")
        sizePolicy.setHeightForWidth(
            self.label_musicFiles.sizePolicy().hasHeightForWidth())
        self.label_musicFiles.setSizePolicy(sizePolicy)
        self.label_musicFiles.setMinimumSize(QSize(0, 100))
        self.label_musicFiles.setMaximumSize(QSize(16777215, 16777215))
        self.label_musicFiles.setAcceptDrops(True)
        self.label_musicFiles.setStyleSheet(u"QLabel {\n"
                                            "	color:  rgb(160, 160, 160);\n"
                                            "	border: none;\n"
                                            "}")
        self.label_musicFiles.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_musicFiles)

        self.verticalLayout.addWidget(self.frame_4)

        self.label_arrow = QLabel(self.frame_2)
        self.label_arrow.setObjectName(u"label_arrow")
        self.label_arrow.setMaximumSize(QSize(16777215, 60))
        self.label_arrow.setStyleSheet(u"font-size: 40px;")
        self.label_arrow.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_arrow)

        self.frame_3 = QFrame(self.frame_2)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setMinimumSize(QSize(0, 175))
        self.frame_3.setMaximumSize(QSize(16777215, 175))
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame_3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(25)
        self.gridLayout_2.setVerticalSpacing(20)
        self.gridLayout_2.setContentsMargins(-1, -1, -1, 30)
        self.label_instrumental = QLabel(self.frame_3)
        self.label_instrumental.setObjectName(u"label_instrumental")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.label_instrumental.sizePolicy().hasHeightForWidth())
        self.label_instrumental.setSizePolicy(sizePolicy2)

        self.gridLayout_2.addWidget(self.label_instrumental, 0, 0, 1, 1)

        self.label_instrumentalFiles = QLabel(self.frame_3)
        self.label_instrumentalFiles.setObjectName(u"label_instrumentalFiles")
        self.label_instrumentalFiles.setStyleSheet(u"QLabel {\n"
                                                   "	color: rgba(160, 160, 160, 80);\n"
                                                   "	border-width: 3px;\n"
                                                   "    border-style: dotted;\n"
                                                   "    border-color: rgba(160, 160, 160, 80);\n"
                                                   "	border-radius: 5px;\n"
                                                   "}")
        self.label_instrumentalFiles.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_instrumentalFiles, 0, 1, 1, 1)

        self.label_vocals = QLabel(self.frame_3)
        self.label_vocals.setObjectName(u"label_vocals")

        self.gridLayout_2.addWidget(self.label_vocals, 1, 0, 1, 1)

        self.label_vocalsFile = QLabel(self.frame_3)
        self.label_vocalsFile.setObjectName(u"label_vocalsFile")
        self.label_vocalsFile.setStyleSheet(u"QLabel {\n"
                                            "	color:rgba(160, 160, 160, 80);\n"
                                            "	border-width: 3px;\n"
                                            "    border-style: dotted;\n"
                                            "    border-color: rgba(160, 160, 160, 80);\n"
                                            "	border-radius: 5px;\n"
                                            "}")
        self.label_vocalsFile.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_vocalsFile, 1, 1, 1, 1)

        self.verticalLayout.addWidget(self.frame_3)

        self.frame = QFrame(self.frame_2)
        self.frame.setObjectName(u"frame")
        self.frame.setMaximumSize(QSize(250, 50))
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_seperate = QPushButton(self.frame)
        self.pushButton_seperate.setObjectName(u"pushButton_seperate")
        self.pushButton_seperate.setMinimumSize(QSize(160, 0))
        self.pushButton_seperate.setMaximumSize(QSize(16777215, 50))
        self.pushButton_seperate.setStyleSheet(u"color: rgb(238, 238, 238);\n"
                                               "border-top-right-radius: 0px;\n"
                                               "border-bottom-right-radius: 0px;")

        self.horizontalLayout.addWidget(self.pushButton_seperate)

        self.pushButton_settings = QPushButton(self.frame)
        self.pushButton_settings.setObjectName(u"pushButton_settings")
        self.pushButton_settings.setMinimumSize(QSize(50, 50))
        self.pushButton_settings.setMaximumSize(QSize(50, 50))
        self.pushButton_settings.setStyleSheet(u"border-left: none;\n"
                                               "border-top-left-radius: 0px;\n"
                                               "border-bottom-left-radius: 0px;")

        self.horizontalLayout.addWidget(self.pushButton_settings)

        self.verticalLayout.addWidget(self.frame, 0, Qt.AlignHCenter)

        self.gridLayout.addWidget(self.frame_2, 0, 0, 1, 1)

        self.progressBar = QProgressBar(MainWindow)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setMinimumSize(QSize(0, 10))
        self.progressBar.setMaximumSize(QSize(16777215, 6))
        self.progressBar.setStyleSheet(u"QProgressBar:horizontal {\n"
                                       "	border: 0px solid gray;\n"
                                       "}\n"
                                       "QProgressBar::chunk:horizontal {\n"
                                       "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0.0795455 rgba(33, 147, 176, 255), stop:1 rgba(109, 213, 237, 255));\n"
                                       "	border-top-right-radius: 2px;\n"
                                       "	border-bottom-right-radius: 2px;\n"
                                       "}\n"
                                       "")
        self.progressBar.setValue(80)
        self.progressBar.setTextVisible(False)
        self.progressBar.setTextDirection(QProgressBar.TopToBottom)

        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 1)

        self.textBrowser_command = QTextBrowser(MainWindow)
        self.textBrowser_command.setObjectName(u"textBrowser_command")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.textBrowser_command.sizePolicy().hasHeightForWidth())
        self.textBrowser_command.setSizePolicy(sizePolicy3)
        self.textBrowser_command.setMinimumSize(QSize(300, 0))
        self.textBrowser_command.setStyleSheet(u"QTextBrowser {\n"
                                               "	border-left: 2px;\n"
                                               "	border-style: solid;\n"
                                               "	border-color: rgb(109, 213, 237);\n"
                                               "	font: 8pt \"Courier\";\n"
                                               "}")
        self.textBrowser_command.setFrameShape(QFrame.StyledPanel)
        self.textBrowser_command.setFrameShadow(QFrame.Plain)
        self.textBrowser_command.setLineWidth(0)
        self.textBrowser_command.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.textBrowser_command.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)

        self.gridLayout.addWidget(self.textBrowser_command, 0, 1, 2, 1)

        self.gridLayout.setColumnStretch(1, 1)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate(
            "MainWindow", u"Vocal Remover", None))
        self.label_musicFiles.setText(QCoreApplication.translate(
            "MainWindow", u"Drag your music files", None))
        self.label_arrow.setText(
            QCoreApplication.translate("MainWindow", u"\u2193", None))
        self.label_instrumental.setText(
            QCoreApplication.translate("MainWindow", u"Instrumentals", None))
        self.label_instrumentalFiles.setText(
            QCoreApplication.translate("MainWindow", u"Audio File", None))
        self.label_vocals.setText(
            QCoreApplication.translate("MainWindow", u"Vocals", None))
        self.label_vocalsFile.setText(
            QCoreApplication.translate("MainWindow", u"Audio File", None))
        self.pushButton_seperate.setText(
            QCoreApplication.translate("MainWindow", u"  Seperate", None))
        self.pushButton_settings.setText("")
        self.progressBar.setFormat("")
        self.textBrowser_command.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                                    "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                                    "p, li { white-space: pre-wrap; }\n"
                                                                    "</style></head><body style=\" font-family:'Courier'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                                                    "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">COMMAND LINE [LOG LEVEL=1]</span></p></body></html>", None))
    # retranslateUi
