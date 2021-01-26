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


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1000, 878)
        Form.setMinimumSize(QSize(1000, 562))
        Form.setStyleSheet(u"/* Universal */\n"
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
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(5, 5, 5, 0)
        self.frame_2 = QFrame(Form)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet(u"")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.label = QLabel(self.frame_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(40, 50, 621, 131))
        self.label.setAcceptDrops(True)
        self.label.setStyleSheet(u"QLabel {\n"
                                 "	color:  rgb(160, 160, 160);\n"
                                 "	border-width: 3px;\n"
                                 "    border-style: dotted;\n"
                                 "    border-color: rgb(160, 160, 160);\n"
                                 "	border-radius: 5px;\n"
                                 "}")
        self.label.setAlignment(Qt.AlignCenter)
        self.frame = QFrame(self.frame_2)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(240, 220, 221, 50))
        self.frame.setMaximumSize(QSize(16777215, 50))
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.frame)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMaximumSize(QSize(16777215, 50))
        self.pushButton.setStyleSheet(u"color: rgb(238, 238, 238);\n"
                                      "border-top-right-radius: 0px;\n"
                                      "border-bottom-right-radius: 0px;")

        self.horizontalLayout.addWidget(self.pushButton)

        self.pushButton_settings = QPushButton(self.frame)
        self.pushButton_settings.setObjectName(u"pushButton_settings")
        self.pushButton_settings.setMinimumSize(QSize(50, 50))
        self.pushButton_settings.setMaximumSize(QSize(50, 50))
        self.pushButton_settings.setStyleSheet(u"border-left: none;\n"
                                               "border-top-left-radius: 0px;\n"
                                               "border-bottom-left-radius: 0px;")

        self.horizontalLayout.addWidget(self.pushButton_settings)

        self.gridLayout.addWidget(self.frame_2, 0, 0, 1, 1)

        self.progressBar = QProgressBar(Form)
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
        self.progressBar.setValue(95)
        self.progressBar.setTextVisible(False)
        self.progressBar.setTextDirection(QProgressBar.TopToBottom)

        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 2)

        self.textBrowser = QTextBrowser(Form)
        self.textBrowser.setObjectName(u"textBrowser")
        sizePolicy1 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy1)
        self.textBrowser.setMinimumSize(QSize(300, 0))
        self.textBrowser.setStyleSheet(u"QTextBrowser {\n"
                                       "	border-left: 2px;\n"
                                       "	border-style: solid;\n"
                                       "	border-color: rgb(109, 213, 237);\n"
                                       "	font: 8pt \"Courier\";\n"
                                       "}")
        self.textBrowser.setFrameShape(QFrame.StyledPanel)
        self.textBrowser.setFrameShadow(QFrame.Plain)
        self.textBrowser.setLineWidth(0)
        self.textBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textBrowser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.gridLayout.addWidget(self.textBrowser, 0, 1, 1, 1)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText(QCoreApplication.translate(
            "Form", u"Drag your music files", None))
        self.pushButton.setText(
            QCoreApplication.translate("Form", u"  Seperate", None))
        self.pushButton_settings.setText("")
        self.progressBar.setFormat("")
        self.textBrowser.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                            "p, li { white-space: pre-wrap; }\n"
                                                            "</style></head><body style=\" font-family:'Courier'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">dwqef</span></p></body></html>", None))
    # retranslateUi
