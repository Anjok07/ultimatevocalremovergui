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
        MainWindow.resize(911, 559)
        MainWindow.setMinimumSize(QSize(0, 0))
        MainWindow.setStyleSheet(u"")
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
        self.stackedWidget_musicFiles = QStackedWidget(self.frame_2)
        self.stackedWidget_musicFiles.setObjectName(
            u"stackedWidget_musicFiles")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.stackedWidget_musicFiles.sizePolicy().hasHeightForWidth())
        self.stackedWidget_musicFiles.setSizePolicy(sizePolicy1)
        self.stackedWidget_musicFiles.setAcceptDrops(True)
        self.stackedWidget_musicFiles.setStyleSheet(u"")
        self.stackedWidget_musicFiles.setFrameShape(QFrame.NoFrame)
        self.stackedWidget_musicFiles.setFrameShadow(QFrame.Plain)
        self.stackedWidget_musicFiles.setLineWidth(1)
        self.page_select = QWidget()
        self.page_select.setObjectName(u"page_select")
        self.verticalLayout_2 = QVBoxLayout(self.page_select)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.pushButton_musicFiles = QPushButton(self.page_select)
        self.pushButton_musicFiles.setObjectName(u"pushButton_musicFiles")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.pushButton_musicFiles.sizePolicy().hasHeightForWidth())
        self.pushButton_musicFiles.setSizePolicy(sizePolicy2)
        self.pushButton_musicFiles.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_musicFiles.setProperty("musicSelect", True)
        self.pushButton_musicFiles.setProperty("title", True)

        self.verticalLayout_2.addWidget(self.pushButton_musicFiles)

        self.stackedWidget_musicFiles.addWidget(self.page_select)
        self.page_display = QWidget()
        self.page_display.setObjectName(u"page_display")
        self.verticalLayout_4 = QVBoxLayout(self.page_display)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.listWidget_musicFiles = QListWidget(self.page_display)
        self.listWidget_musicFiles.setObjectName(u"listWidget_musicFiles")
        sizePolicy3 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.listWidget_musicFiles.sizePolicy().hasHeightForWidth())
        self.listWidget_musicFiles.setSizePolicy(sizePolicy3)
        self.listWidget_musicFiles.setMaximumSize(QSize(16777215, 16777215))
        self.listWidget_musicFiles.setFrameShape(QFrame.NoFrame)
        self.listWidget_musicFiles.setLineWidth(0)
        self.listWidget_musicFiles.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.listWidget_musicFiles.setEditTriggers(
            QAbstractItemView.NoEditTriggers)
        self.listWidget_musicFiles.setAlternatingRowColors(True)
        self.listWidget_musicFiles.setSelectionMode(
            QAbstractItemView.NoSelection)
        self.listWidget_musicFiles.setWordWrap(True)
        self.listWidget_musicFiles.setProperty("musicSelect", True)

        self.verticalLayout_4.addWidget(self.listWidget_musicFiles)

        self.stackedWidget_musicFiles.addWidget(self.page_display)

        self.verticalLayout.addWidget(self.stackedWidget_musicFiles)

        self.label_arrow = QLabel(self.frame_2)
        self.label_arrow.setObjectName(u"label_arrow")
        self.label_arrow.setMaximumSize(QSize(16777215, 60))
        self.label_arrow.setStyleSheet(u"font-size: 40px;")
        self.label_arrow.setText(u"\u2193")
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
        self.stackedWidget_vocals = QStackedWidget(self.frame_3)
        self.stackedWidget_vocals.setObjectName(u"stackedWidget_vocals")
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.verticalLayout_6 = QVBoxLayout(self.page_3)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_vocalsFile = QLabel(self.page_3)
        self.label_vocalsFile.setObjectName(u"label_vocalsFile")
        self.label_vocalsFile.setAlignment(Qt.AlignCenter)
        self.label_vocalsFile.setProperty("audioPlayer", True)
        self.label_vocalsFile.setProperty("title", True)

        self.verticalLayout_6.addWidget(self.label_vocalsFile)

        self.stackedWidget_vocals.addWidget(self.page_3)
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.verticalLayout_7 = QVBoxLayout(self.page_4)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.frame_4 = QFrame(self.page_4)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_3.setSpacing(15)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.pushButton_play_vocals = QPushButton(self.frame_4)
        self.pushButton_play_vocals.setObjectName(u"pushButton_play_vocals")
        self.pushButton_play_vocals.setMinimumSize(QSize(0, 0))
        self.pushButton_play_vocals.setMaximumSize(QSize(30, 30))
        self.pushButton_play_vocals.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_play_vocals.setProperty("audioPlayer", True)

        self.horizontalLayout_3.addWidget(self.pushButton_play_vocals)

        self.horizontalSlider_vocals = QSlider(self.frame_4)
        self.horizontalSlider_vocals.setObjectName(u"horizontalSlider_vocals")
        self.horizontalSlider_vocals.setOrientation(Qt.Horizontal)
        self.horizontalSlider_vocals.setProperty("audioPlayer", True)

        self.horizontalLayout_3.addWidget(self.horizontalSlider_vocals)

        self.pushButton_menu_vocals = QPushButton(self.frame_4)
        self.pushButton_menu_vocals.setObjectName(u"pushButton_menu_vocals")
        self.pushButton_menu_vocals.setMaximumSize(QSize(30, 30))
        self.pushButton_menu_vocals.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_menu_vocals.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pushButton_menu_vocals.setStyleSheet(u"")
        self.pushButton_menu_vocals.setProperty("audioPlayer", True)

        self.horizontalLayout_3.addWidget(self.pushButton_menu_vocals)

        self.verticalLayout_7.addWidget(self.frame_4)

        self.stackedWidget_vocals.addWidget(self.page_4)

        self.gridLayout_2.addWidget(self.stackedWidget_vocals, 1, 2, 1, 1)

        self.stackedWidget_instrumentals = QStackedWidget(self.frame_3)
        self.stackedWidget_instrumentals.setObjectName(
            u"stackedWidget_instrumentals")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.verticalLayout_5 = QVBoxLayout(self.page)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_instrumentalFiles = QLabel(self.page)
        self.label_instrumentalFiles.setObjectName(u"label_instrumentalFiles")
        self.label_instrumentalFiles.setAlignment(Qt.AlignCenter)
        self.label_instrumentalFiles.setProperty("audioPlayer", True)
        self.label_instrumentalFiles.setProperty("title", True)

        self.verticalLayout_5.addWidget(self.label_instrumentalFiles)

        self.stackedWidget_instrumentals.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.verticalLayout_8 = QVBoxLayout(self.page_2)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.page_2)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_4.setSpacing(15)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pushButton_play_instrumentals = QPushButton(self.frame_6)
        self.pushButton_play_instrumentals.setObjectName(
            u"pushButton_play_instrumentals")
        self.pushButton_play_instrumentals.setMinimumSize(QSize(0, 0))
        self.pushButton_play_instrumentals.setMaximumSize(QSize(30, 30))
        self.pushButton_play_instrumentals.setCursor(
            QCursor(Qt.PointingHandCursor))
        self.pushButton_play_instrumentals.setProperty("audioPlayer", True)

        self.horizontalLayout_4.addWidget(self.pushButton_play_instrumentals)

        self.horizontalSlider_instrumentals = QSlider(self.frame_6)
        self.horizontalSlider_instrumentals.setObjectName(
            u"horizontalSlider_instrumentals")
        self.horizontalSlider_instrumentals.setOrientation(Qt.Horizontal)
        self.horizontalSlider_instrumentals.setProperty("audioPlayer", True)

        self.horizontalLayout_4.addWidget(self.horizontalSlider_instrumentals)

        self.pushButton_menu_instrumentals = QPushButton(self.frame_6)
        self.pushButton_menu_instrumentals.setObjectName(
            u"pushButton_menu_instrumentals")
        self.pushButton_menu_instrumentals.setMaximumSize(QSize(30, 30))
        self.pushButton_menu_instrumentals.setCursor(
            QCursor(Qt.PointingHandCursor))
        self.pushButton_menu_instrumentals.setContextMenuPolicy(
            Qt.CustomContextMenu)
        self.pushButton_menu_instrumentals.setStyleSheet(u"")
        self.pushButton_menu_instrumentals.setProperty("audioPlayer", True)

        self.horizontalLayout_4.addWidget(self.pushButton_menu_instrumentals)

        self.verticalLayout_8.addWidget(self.frame_6)

        self.stackedWidget_instrumentals.addWidget(self.page_2)

        self.gridLayout_2.addWidget(
            self.stackedWidget_instrumentals, 0, 2, 1, 1)

        self.label_instrumental = QLabel(self.frame_3)
        self.label_instrumental.setObjectName(u"label_instrumental")
        sizePolicy4 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.label_instrumental.sizePolicy().hasHeightForWidth())
        self.label_instrumental.setSizePolicy(sizePolicy4)
        self.label_instrumental.setProperty("title", True)

        self.gridLayout_2.addWidget(self.label_instrumental, 0, 0, 1, 1)

        self.label_vocals = QLabel(self.frame_3)
        self.label_vocals.setObjectName(u"label_vocals")
        self.label_vocals.setProperty("title", True)

        self.gridLayout_2.addWidget(self.label_vocals, 1, 0, 1, 1)

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
        self.pushButton_seperate.setProperty("title", True)

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
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy5)
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
        sizePolicy6 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(
            self.textBrowser_command.sizePolicy().hasHeightForWidth())
        self.textBrowser_command.setSizePolicy(sizePolicy6)
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
                                         "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
                                         "<table border=\"1\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px;\" width=\"100%\" cellspacing=\"2\" cellpadding=\"0\">\n"
                                         "<tr>\n"
                                         "<td colspan=\"2\">\n"
                                         "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier'; font-size:8pt; font-weight:600;\">HEADER</span><span style=\" font-family:'Courier'; font-size:8pt;\">        </span></p></td></tr>\n"
                                         "<tr>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier'; fon"
                                         "t-size:8pt;\">name</span></p></td>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier'; font-size:8pt;\">value        </span></p></td></tr>\n"
                                         "<tr>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier'; font-size:8pt;\">name</span></p></td>\n"
                                         "<td>\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'Courier'; font-size:8pt;\">value    </span></p></td></tr></table></body></html>")

        self.horizontalLayout_2.addWidget(self.textBrowser_command)

        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 2)

        self.retranslateUi(MainWindow)

        self.stackedWidget_musicFiles.setCurrentIndex(0)
        self.stackedWidget_vocals.setCurrentIndex(0)
        self.stackedWidget_instrumentals.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate(
            "MainWindow", u"Vocal Remover", None))
        self.pushButton_musicFiles.setText(QCoreApplication.translate(
            "MainWindow", u"Drag your music files", None))
        self.label_vocalsFile.setText(
            QCoreApplication.translate("MainWindow", u"Audio File", None))
        self.pushButton_play_vocals.setText("")
        self.pushButton_menu_vocals.setText("")
        self.label_instrumentalFiles.setText(
            QCoreApplication.translate("MainWindow", u"Audio File", None))
        self.pushButton_play_instrumentals.setText("")
        self.pushButton_menu_instrumentals.setText("")
        self.label_instrumental.setText(
            QCoreApplication.translate("MainWindow", u"Instrumentals", None))
        self.label_vocals.setText(
            QCoreApplication.translate("MainWindow", u"Vocals", None))
        self.pushButton_seperate.setText(
            QCoreApplication.translate("MainWindow", u" Separate", None))
    # retranslateUi
