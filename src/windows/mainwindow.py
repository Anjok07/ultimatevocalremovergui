# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'mainwindow.ui'
##
# Created by: Qt User Interface Compiler version 5.15.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(911, 501)
        MainWindow.setMinimumSize(QSize(0, 0))
        MainWindow.setStyleSheet(u"/* Universal */\n"
                                 "* {\n"
                                 "	font: 15pt \"Yu Gothic UI\";	\n"
                                 "	color: rgb(255, 255, 255);\n"
                                 "	background-color: none;\n"
                                 "	background: rgb(12, 23, 40);\n"
                                 "}\n"
                                 "/* Labels */\n"
                                 "QLabel[audioFile=\"true\"] {\n"
                                 "	color:rgba(160, 160, 160, 80);\n"
                                 "	border-width: 3px;\n"
                                 "    border-style: dotted;\n"
                                 "    border-color: rgb(60, 60, 80);\n"
                                 "	border-radius: 5px;\n"
                                 "}\n"
                                 "/* Button */\n"
                                 "QPushButton[seperate=\"true\"] {\n"
                                 "	border-width: 2px;\n"
                                 "	border-style: solid;\n"
                                 "	border-radius: 15px;\n"
                                 "	border-color: rgb(109, 213, 237);\n"
                                 "	background-color: rgba(109, 213, 237, 4);\n"
                                 "}\n"
                                 "QPushButton[seperate=\"true\"]:hover {\n"
                                 "	background-color: rgba(109, 213, 237, 10);\n"
                                 "}\n"
                                 "QPushButton[seperate=\"true\"]:pressed {\n"
                                 "	background-color: rgba(109, 213, 237, 30);\n"
                                 "}\n"
                                 "QPushButton[dragFile=\"true\"] {\n"
                                 "	color:  rgb(160, 160, 160);\n"
                                 "	border-width: 3px;\n"
                                 "    border-style: dotted;\n"
                                 "    border-color: rgb(160, 160, 160);\n"
                                 "	border-radius: 5px;\n"
                                 "}\n"
                                 "QPushButton[dragFile=\"true\"]:hover"
                                 " {\n"
                                 "	background-color: rgb(2, 24, 53);\n"
                                 "}\n"
                                 "QPushButton[dragFile=\"true\"]:pressed {\n"
                                 "	background-color: rgb(1, 24, 61);\n"
                                 "}\n"
                                 "\n"
                                 "/* Progressbar */\n"
                                 "\n"
                                 "/* Command Line*/\n"
                                 "QTextBrowser {\n"
                                 "	border-left: 2px;\n"
                                 "	border-style: solid;\n"
                                 "	border-color: rgb(109, 213, 237);\n"
                                 "	font: 8pt \"Courier\";\n"
                                 "}")
        self.horizontalLayout_2 = QHBoxLayout(MainWindow)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 5, 0)
        self.frame_5 = QFrame(MainWindow)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setMinimumSize(QSize(650, 0))
        self.frame_5.setFrameShape(QFrame.NoFrame)
        self.frame_5.setFrameShadow(QFrame.Plain)
        self.frame_5.setLineWidth(0)
        self.verticalLayout_3 = QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_2 = QFrame(self.frame_5)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QSize(0, 0))
        self.frame_2.setMaximumSize(QSize(16777215, 16777215))
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_2)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(30, 30, 30, 30)
        self.frame_musicFiles = QFrame(self.frame_2)
        self.frame_musicFiles.setObjectName(u"frame_musicFiles")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.frame_musicFiles.sizePolicy().hasHeightForWidth())
        self.frame_musicFiles.setSizePolicy(sizePolicy1)
        self.frame_musicFiles.setAcceptDrops(True)
        self.frame_musicFiles.setStyleSheet(u"")
        self.frame_musicFiles.setFrameShape(QFrame.NoFrame)
        self.frame_musicFiles.setFrameShadow(QFrame.Plain)
        self.frame_musicFiles.setLineWidth(1)
        self.verticalLayout_2 = QVBoxLayout(self.frame_musicFiles)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.listWidget_musicFiles = QListWidget(self.frame_musicFiles)
        self.listWidget_musicFiles.setObjectName(u"listWidget_musicFiles")
        self.listWidget_musicFiles.setMaximumSize(QSize(16777215, 0))
        self.listWidget_musicFiles.setFrameShape(QFrame.NoFrame)
        self.listWidget_musicFiles.setLineWidth(0)

        self.verticalLayout_2.addWidget(self.listWidget_musicFiles)

        self.pushButton_musicFiles = QPushButton(self.frame_musicFiles)
        self.pushButton_musicFiles.setObjectName(u"pushButton_musicFiles")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.pushButton_musicFiles.sizePolicy().hasHeightForWidth())
        self.pushButton_musicFiles.setSizePolicy(sizePolicy2)
        self.pushButton_musicFiles.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_musicFiles.setProperty("dragFile", True)

        self.verticalLayout_2.addWidget(self.pushButton_musicFiles)

        self.verticalLayout.addWidget(self.frame_musicFiles)

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
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.label_instrumental.sizePolicy().hasHeightForWidth())
        self.label_instrumental.setSizePolicy(sizePolicy3)

        self.gridLayout_2.addWidget(self.label_instrumental, 0, 0, 1, 1)

        self.label_instrumentalFiles = QLabel(self.frame_3)
        self.label_instrumentalFiles.setObjectName(u"label_instrumentalFiles")
        self.label_instrumentalFiles.setAlignment(Qt.AlignCenter)
        self.label_instrumentalFiles.setProperty("audioFile", True)

        self.gridLayout_2.addWidget(self.label_instrumentalFiles, 0, 1, 1, 1)

        self.label_vocals = QLabel(self.frame_3)
        self.label_vocals.setObjectName(u"label_vocals")

        self.gridLayout_2.addWidget(self.label_vocals, 1, 0, 1, 1)

        self.label_vocalsFile = QLabel(self.frame_3)
        self.label_vocalsFile.setObjectName(u"label_vocalsFile")
        self.label_vocalsFile.setAlignment(Qt.AlignCenter)
        self.label_vocalsFile.setProperty("audioFile", True)

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
        self.pushButton_seperate.setStyleSheet(u"border-top-right-radius: 0px;\n"
                                               "border-bottom-right-radius: 0px;")
        self.pushButton_seperate.setProperty("seperate", True)

        self.horizontalLayout.addWidget(self.pushButton_seperate)

        self.pushButton_settings = QPushButton(self.frame)
        self.pushButton_settings.setObjectName(u"pushButton_settings")
        self.pushButton_settings.setMinimumSize(QSize(50, 50))
        self.pushButton_settings.setMaximumSize(QSize(50, 50))
        self.pushButton_settings.setStyleSheet(u"border-left: none;\n"
                                               "border-top-left-radius: 0px;\n"
                                               "border-bottom-left-radius: 0px;")
        self.pushButton_settings.setText(u"")
        self.pushButton_settings.setProperty("seperate", True)

        self.horizontalLayout.addWidget(self.pushButton_settings)

        self.verticalLayout.addWidget(self.frame, 0, Qt.AlignHCenter)

        self.verticalLayout_3.addWidget(self.frame_2)

        self.progressBar = QProgressBar(self.frame_5)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy4)
        self.progressBar.setMinimumSize(QSize(0, 10))
        self.progressBar.setMaximumSize(QSize(16777215, 6))
        self.progressBar.setStyleSheet(u"QProgressBar:horizontal {\n"
                                       "	border: 0px solid gray;\n"
                                       "}\n"
                                       "QProgressBar::chunk {\n"
                                       "}")
        self.progressBar.setMaximum(200)
        self.progressBar.setValue(200)
        self.progressBar.setTextVisible(False)
        self.progressBar.setTextDirection(QProgressBar.TopToBottom)
        self.progressBar.setFormat(u"")

        self.verticalLayout_3.addWidget(self.progressBar)

        self.horizontalLayout_2.addWidget(self.frame_5)

        self.textBrowser_command = QTextBrowser(MainWindow)
        self.textBrowser_command.setObjectName(u"textBrowser_command")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.textBrowser_command.sizePolicy().hasHeightForWidth())
        self.textBrowser_command.setSizePolicy(sizePolicy5)
        self.textBrowser_command.setMinimumSize(QSize(0, 0))
        self.textBrowser_command.setFrameShape(QFrame.StyledPanel)
        self.textBrowser_command.setFrameShadow(QFrame.Plain)
        self.textBrowser_command.setLineWidth(0)
        self.textBrowser_command.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.textBrowser_command.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.textBrowser_command.setHtml(u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:'Courier'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                         "<table border=\"1\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px;\" width=\"100%\" cellspacing=\"2\" cellpadding=\"0\">\n"
                                         "<tr>\n"
                                         "<td colspan=\"2\">\n"
                                         "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">HEADER</span>        </p></td></tr>\n"
                                         "<tr>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">name</p></td>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">val"
                                         "ue        </p></td></tr>\n"
                                         "<tr>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">name</p></td>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">value    </p></td></tr></table></body></html>")

        self.horizontalLayout_2.addWidget(self.textBrowser_command)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate(
            "MainWindow", u"Vocal Remover", None))
        self.pushButton_musicFiles.setText(QCoreApplication.translate(
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
    # retranslateUi
