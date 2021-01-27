# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'settingswindow.ui'
##
# Created by: Qt User Interface Compiler version 6.0.0
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_SettingsWindow(object):
    def setupUi(self, SettingsWindow):
        if not SettingsWindow.objectName():
            SettingsWindow.setObjectName(u"SettingsWindow")
        SettingsWindow.resize(777, 344)
        SettingsWindow.setStyleSheet(u"* {\n"
                                     "	font: 16pt \"Yu Gothic UI\";	\n"
                                     "	color: rgb(255, 255, 255);\n"
                                     "}\n"
                                     "\n"
                                     "QWidget#SettingsWindow {\n"
                                     "	background-color: rgb(2, 18, 40);\n"
                                     "}\n"
                                     "QFrame {\n"
                                     "	background-color: rgb(2, 18, 40);\n"
                                     "}\n"
                                     "QPushButton {\n"
                                     "	background-color: rgb(2, 5, 39);\n"
                                     "	border-width: 2px;\n"
                                     "	border-style: solid;\n"
                                     "	border-radius: 22px;\n"
                                     "	border-color: rgb(85, 78, 163)\n"
                                     "}\n"
                                     "QPushButton:hover {\n"
                                     "	background-color: rgba(85, 78, 163, 60);\n"
                                     "}\n"
                                     "QPushButton:pressed {\n"
                                     "	background-color: rgba(85, 78, 163, 140);\n"
                                     "	border-width: 0px;\n"
                                     "}\n"
                                     "* {\n"
                                     "	font: 10pt \"Yu Gothic UI\";	\n"
                                     "	color: rgb(255, 255, 255);\n"
                                     "}\n"
                                     "QFrame#frame_settingOptions, QFrame#frame_modelOptions, QFrame#frame_engine {\n"
                                     "	background-color: rgb(2, 5, 39);\n"
                                     "	border-width: 2px;\n"
                                     "	border-radius: 8px;\n"
                                     "	border-style: solid;\n"
                                     "	border-color: rgb(85, 78, 163);\n"
                                     "}\n"
                                     "QLineEdit, QComboBox {\n"
                                     "	color: rgb(0, 0, 0);\n"
                                     "}\n"
                                     "QLabel[title=\"true\"] {\n"
                                     "	font: 15pt \"Yu Gothic UI\";\n"
                                     "}")
        self.gridLayout_2 = QGridLayout(SettingsWindow)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.frame_modelOptions = QFrame(SettingsWindow)
        self.frame_modelOptions.setObjectName(u"frame_modelOptions")
        self.frame_modelOptions.setStyleSheet(u"* {\n"
                                              "	background-color: none;\n"
                                              "}\n"
                                              "QFrame#frame_modelOptions {\n"
                                              "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.148, y2:0.387227, stop:0.0852273 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0))\n"
                                              "}")
        self.frame_modelOptions.setFrameShape(QFrame.StyledPanel)
        self.frame_modelOptions.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_modelOptions)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label__title_models = QLabel(self.frame_modelOptions)
        self.label__title_models.setObjectName(u"label__title_models")
        self.label__title_models.setMinimumSize(QSize(0, 31))
        self.label__title_models.setIndent(8)
        self.label__title_models.setProperty("title", True)

        self.verticalLayout_5.addWidget(self.label__title_models)

        self.frame_5 = QFrame(self.frame_modelOptions)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.frame_8 = QFrame(self.frame_5)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_8)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 5, 9, 25)
        self.label_2 = QLabel(self.frame_8)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 30))
        self.label_2.setMaximumSize(QSize(16777215, 20))
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_2)

        self.comboBox_resType_3 = QComboBox(self.frame_8)
        self.comboBox_resType_3.setObjectName(u"comboBox_resType_3")
        self.comboBox_resType_3.setMinimumSize(QSize(0, 30))
        self.comboBox_resType_3.setMaximumSize(QSize(16777215, 30))
        self.comboBox_resType_3.setStyleSheet(u"color: rgb(0, 0, 0);")
        self.comboBox_resType_3.setEditable(False)
        self.comboBox_resType_3.setMaxVisibleItems(10)

        self.verticalLayout_3.addWidget(self.comboBox_resType_3)

        self.horizontalLayout_5.addWidget(self.frame_8)

        self.frame_constants = QFrame(self.frame_5)
        self.frame_constants.setObjectName(u"frame_constants")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.frame_constants.sizePolicy().hasHeightForWidth())
        self.frame_constants.setSizePolicy(sizePolicy)
        self.frame_constants.setMinimumSize(QSize(230, 0))
        self.frame_constants.setMaximumSize(QSize(230, 140))
        self.frame_constants.setFrameShape(QFrame.StyledPanel)
        self.frame_constants.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_constants)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, 5, -1, 3)
        self.label_sr = QLabel(self.frame_constants)
        self.label_sr.setObjectName(u"label_sr")
        self.label_sr.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_sr, 0, 1, 1, 1)

        self.label_nfft = QLabel(self.frame_constants)
        self.label_nfft.setObjectName(u"label_nfft")
        self.label_nfft.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_nfft, 5, 1, 1, 1)

        self.label_winSize = QLabel(self.frame_constants)
        self.label_winSize.setObjectName(u"label_winSize")
        self.label_winSize.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_winSize, 4, 1, 1, 1)

        self.label_hopLength = QLabel(self.frame_constants)
        self.label_hopLength.setObjectName(u"label_hopLength")
        self.label_hopLength.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_hopLength, 2, 1, 1, 1)

        self.lineEdit_instr_nfft = QLineEdit(self.frame_constants)
        self.lineEdit_instr_nfft.setObjectName(u"lineEdit_instr_nfft")
        self.lineEdit_instr_nfft.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.lineEdit_instr_nfft, 5, 0, 1, 1)

        self.lineEdit_instr_winSize = QLineEdit(self.frame_constants)
        self.lineEdit_instr_winSize.setObjectName(u"lineEdit_instr_winSize")
        self.lineEdit_instr_winSize.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.lineEdit_instr_winSize, 4, 0, 1, 1)

        self.lineEdit_stack_sr = QLineEdit(self.frame_constants)
        self.lineEdit_stack_sr.setObjectName(u"lineEdit_stack_sr")
        self.lineEdit_stack_sr.setAlignment(Qt.AlignCenter)
        self.lineEdit_stack_sr.setDragEnabled(False)
        self.lineEdit_stack_sr.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_stack_sr, 0, 2, 1, 1)

        self.lineEdit_stack_winSize = QLineEdit(self.frame_constants)
        self.lineEdit_stack_winSize.setObjectName(u"lineEdit_stack_winSize")
        self.lineEdit_stack_winSize.setAlignment(Qt.AlignCenter)
        self.lineEdit_stack_winSize.setDragEnabled(False)
        self.lineEdit_stack_winSize.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_stack_winSize, 4, 2, 1, 1)

        self.lineEdit_stack_hopLength = QLineEdit(self.frame_constants)
        self.lineEdit_stack_hopLength.setObjectName(
            u"lineEdit_stack_hopLength")
        self.lineEdit_stack_hopLength.setAlignment(Qt.AlignCenter)
        self.lineEdit_stack_hopLength.setDragEnabled(False)
        self.lineEdit_stack_hopLength.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_stack_hopLength, 2, 2, 1, 1)

        self.lineEdit_stack_nfft = QLineEdit(self.frame_constants)
        self.lineEdit_stack_nfft.setObjectName(u"lineEdit_stack_nfft")
        self.lineEdit_stack_nfft.setAlignment(Qt.AlignCenter)
        self.lineEdit_stack_nfft.setDragEnabled(False)
        self.lineEdit_stack_nfft.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_stack_nfft, 5, 2, 1, 1)

        self.lineEdit_instr_hopLength = QLineEdit(self.frame_constants)
        self.lineEdit_instr_hopLength.setObjectName(
            u"lineEdit_instr_hopLength")
        self.lineEdit_instr_hopLength.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.lineEdit_instr_hopLength, 2, 0, 1, 1)

        self.lineEdit_instr_sr = QLineEdit(self.frame_constants)
        self.lineEdit_instr_sr.setObjectName(u"lineEdit_instr_sr")
        self.lineEdit_instr_sr.setAlignment(Qt.AlignCenter)
        self.lineEdit_instr_sr.setDragEnabled(False)
        self.lineEdit_instr_sr.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_instr_sr, 0, 0, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)
        self.gridLayout_5.setColumnStretch(1, 2)
        self.gridLayout_5.setColumnStretch(2, 1)

        self.horizontalLayout_5.addWidget(self.frame_constants)

        self.frame_9 = QFrame(self.frame_5)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_9)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, 5, 9, 25)
        self.label_4 = QLabel(self.frame_9)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(0, 30))
        self.label_4.setMaximumSize(QSize(16777215, 20))
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_4)

        self.comboBox_resType_5 = QComboBox(self.frame_9)
        self.comboBox_resType_5.setObjectName(u"comboBox_resType_5")
        self.comboBox_resType_5.setMinimumSize(QSize(0, 30))
        self.comboBox_resType_5.setMaximumSize(QSize(16777215, 30))
        self.comboBox_resType_5.setStyleSheet(u"color: rgb(0, 0, 0);")
        self.comboBox_resType_5.setEditable(False)
        self.comboBox_resType_5.setMaxVisibleItems(10)

        self.verticalLayout_4.addWidget(self.comboBox_resType_5)

        self.horizontalLayout_5.addWidget(self.frame_9)

        self.verticalLayout_5.addWidget(self.frame_5)

        self.gridLayout_2.addWidget(self.frame_modelOptions, 1, 0, 1, 2)

        self.frame_engine = QFrame(SettingsWindow)
        self.frame_engine.setObjectName(u"frame_engine")
        self.frame_engine.setMinimumSize(QSize(230, 0))
        self.frame_engine.setStyleSheet(u"* {\n"
                                        "	background-color: none;\n"
                                        "}\n"
                                        "QFrame#frame_engine {\n"
                                        "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.193, y2:0.484273, stop:0.0909091 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0))\n"
                                        "}")
        self.frame_engine.setFrameShape(QFrame.StyledPanel)
        self.frame_engine.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame_engine)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.label_title_engine = QLabel(self.frame_engine)
        self.label_title_engine.setObjectName(u"label_title_engine")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.label_title_engine.sizePolicy().hasHeightForWidth())
        self.label_title_engine.setSizePolicy(sizePolicy1)
        self.label_title_engine.setMinimumSize(QSize(0, 31))
        self.label_title_engine.setIndent(8)
        self.label_title_engine.setProperty("title", True)

        self.verticalLayout_7.addWidget(self.label_title_engine)

        self.frame_12 = QFrame(self.frame_engine)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Raised)
        self.gridLayout_6 = QGridLayout(self.frame_12)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setHorizontalSpacing(20)
        self.gridLayout_6.setContentsMargins(16, -1, 16, -1)
        self.label_engine = QLabel(self.frame_12)
        self.label_engine.setObjectName(u"label_engine")
        sizePolicy1.setHeightForWidth(
            self.label_engine.sizePolicy().hasHeightForWidth())
        self.label_engine.setSizePolicy(sizePolicy1)
        self.label_engine.setStyleSheet(u"background-color: none;")
        self.label_engine.setAlignment(Qt.AlignCenter)
        self.label_engine.setIndent(10)

        self.gridLayout_6.addWidget(self.label_engine, 0, 0, 1, 1)

        self.label_resType = QLabel(self.frame_12)
        self.label_resType.setObjectName(u"label_resType")
        sizePolicy1.setHeightForWidth(
            self.label_resType.sizePolicy().hasHeightForWidth())
        self.label_resType.setSizePolicy(sizePolicy1)
        self.label_resType.setStyleSheet(u"background-color: none;")
        self.label_resType.setAlignment(Qt.AlignCenter)
        self.label_resType.setIndent(10)

        self.gridLayout_6.addWidget(self.label_resType, 0, 1, 1, 1)

        self.comboBox_engine = QComboBox(self.frame_12)
        self.comboBox_engine.addItem("")
        self.comboBox_engine.addItem("")
        self.comboBox_engine.setObjectName(u"comboBox_engine")
        self.comboBox_engine.setMinimumSize(QSize(0, 30))
        self.comboBox_engine.setStyleSheet(u"color: rgb(0, 0, 0);")

        self.gridLayout_6.addWidget(self.comboBox_engine, 1, 0, 1, 1)

        self.comboBox_resType = QComboBox(self.frame_12)
        self.comboBox_resType.addItem("")
        self.comboBox_resType.addItem("")
        self.comboBox_resType.addItem("")
        self.comboBox_resType.setObjectName(u"comboBox_resType")
        self.comboBox_resType.setMinimumSize(QSize(0, 30))
        self.comboBox_resType.setStyleSheet(u"color: rgb(0, 0, 0);")

        self.gridLayout_6.addWidget(self.comboBox_resType, 1, 1, 1, 1)

        self.verticalLayout_7.addWidget(self.frame_12)

        self.gridLayout_2.addWidget(self.frame_engine, 0, 1, 1, 1)

        self.frame_settingOptions = QFrame(SettingsWindow)
        self.frame_settingOptions.setObjectName(u"frame_settingOptions")
        self.frame_settingOptions.setMinimumSize(QSize(500, 0))
        self.frame_settingOptions.setStyleSheet(u"* {\n"
                                                "	background-color: none;\n"
                                                "}\n"
                                                "QFrame#frame_settingOptions {\n"
                                                "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.193, y2:0.484273, stop:0.0909091 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0))\n"
                                                "}")
        self.frame_settingOptions.setFrameShape(QFrame.StyledPanel)
        self.frame_settingOptions.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_settingOptions)
        self.verticalLayout_6.setSpacing(10)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_title_conversion = QLabel(self.frame_settingOptions)
        self.label_title_conversion.setObjectName(u"label_title_conversion")
        self.label_title_conversion.setMinimumSize(QSize(0, 31))
        self.label_title_conversion.setIndent(8)
        self.label_title_conversion.setProperty("title", True)

        self.verticalLayout_6.addWidget(self.label_title_conversion)

        self.frame_7 = QFrame(self.frame_settingOptions)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setMinimumSize(QSize(0, 100))
        self.frame_7.setFrameShape(QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.gridLayout_4 = QGridLayout(self.frame_7)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setVerticalSpacing(10)
        self.gridLayout_4.setContentsMargins(15, -1, 15, -1)
        self.checkBox_gpuConversion = QCheckBox(self.frame_7)
        self.checkBox_gpuConversion.setObjectName(u"checkBox_gpuConversion")

        self.gridLayout_4.addWidget(self.checkBox_gpuConversion, 1, 0, 1, 1)

        self.checkBox_saveAllStacked = QCheckBox(self.frame_7)
        self.checkBox_saveAllStacked.setObjectName(u"checkBox_saveAllStacked")

        self.gridLayout_4.addWidget(self.checkBox_saveAllStacked, 3, 0, 1, 1)

        self.checkBox_postProcess = QCheckBox(self.frame_7)
        self.checkBox_postProcess.setObjectName(u"checkBox_postProcess")

        self.gridLayout_4.addWidget(self.checkBox_postProcess, 1, 1, 1, 1)

        self.checkBox_stackOnly = QCheckBox(self.frame_7)
        self.checkBox_stackOnly.setObjectName(u"checkBox_stackOnly")

        self.gridLayout_4.addWidget(self.checkBox_stackOnly, 3, 2, 1, 1)

        self.checkBox_outputImage = QCheckBox(self.frame_7)
        self.checkBox_outputImage.setObjectName(u"checkBox_outputImage")

        self.gridLayout_4.addWidget(self.checkBox_outputImage, 1, 2, 1, 1)

        self.checkBox_modelTest = QCheckBox(self.frame_7)
        self.checkBox_modelTest.setObjectName(u"checkBox_modelTest")

        self.gridLayout_4.addWidget(self.checkBox_modelTest, 2, 0, 1, 1)

        self.checkBox_tta = QCheckBox(self.frame_7)
        self.checkBox_tta.setObjectName(u"checkBox_tta")

        self.gridLayout_4.addWidget(self.checkBox_tta, 2, 1, 1, 1)

        self.checkBox_customParameters = QCheckBox(self.frame_7)
        self.checkBox_customParameters.setObjectName(
            u"checkBox_customParameters")

        self.gridLayout_4.addWidget(self.checkBox_customParameters, 2, 2, 1, 1)

        self.frame_10 = QFrame(self.frame_7)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.NoFrame)
        self.frame_10.setFrameShadow(QFrame.Plain)
        self.frame_10.setLineWidth(0)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.checkBox_stackPasses = QCheckBox(self.frame_10)
        self.checkBox_stackPasses.setObjectName(u"checkBox_stackPasses")

        self.horizontalLayout_3.addWidget(self.checkBox_stackPasses)

        self.lineEdit_stackPasses = QLineEdit(self.frame_10)
        self.lineEdit_stackPasses.setObjectName(u"lineEdit_stackPasses")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.lineEdit_stackPasses.sizePolicy().hasHeightForWidth())
        self.lineEdit_stackPasses.setSizePolicy(sizePolicy2)
        self.lineEdit_stackPasses.setMaximumSize(QSize(22, 25))
        self.lineEdit_stackPasses.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_3.addWidget(
            self.lineEdit_stackPasses, 0, Qt.AlignHCenter)

        self.gridLayout_4.addWidget(self.frame_10, 3, 1, 1, 1, Qt.AlignLeft)

        self.verticalLayout_6.addWidget(self.frame_7)

        self.gridLayout_2.addWidget(self.frame_settingOptions, 0, 0, 1, 1)

        self.retranslateUi(SettingsWindow)

        QMetaObject.connectSlotsByName(SettingsWindow)
    # setupUi

    def retranslateUi(self, SettingsWindow):
        SettingsWindow.setWindowTitle(QCoreApplication.translate(
            "SettingsWindow", u"Settings", None))
        self.label__title_models.setText(
            QCoreApplication.translate("SettingsWindow", u"Models", None))
        self.label_2.setText(QCoreApplication.translate(
            "SettingsWindow", u"Instrumental Model", None))
        self.comboBox_resType_3.setCurrentText("")
        self.label_sr.setText(QCoreApplication.translate(
            "SettingsWindow", u"SR", None))
        self.label_nfft.setText(QCoreApplication.translate(
            "SettingsWindow", u"N_FFT", None))
        self.label_winSize.setText(QCoreApplication.translate(
            "SettingsWindow", u"Window Size", None))
        self.label_hopLength.setText(QCoreApplication.translate(
            "SettingsWindow", u"Hop Length", None))
        self.lineEdit_instr_nfft.setText(
            QCoreApplication.translate("SettingsWindow", u"2048", None))
        self.lineEdit_instr_winSize.setText(
            QCoreApplication.translate("SettingsWindow", u"320", None))
        self.lineEdit_stack_sr.setText(
            QCoreApplication.translate("SettingsWindow", u"44100", None))
        self.lineEdit_stack_winSize.setText(
            QCoreApplication.translate("SettingsWindow", u"320", None))
        self.lineEdit_stack_hopLength.setText(
            QCoreApplication.translate("SettingsWindow", u"1024", None))
        self.lineEdit_stack_nfft.setText(
            QCoreApplication.translate("SettingsWindow", u"2048", None))
        self.lineEdit_instr_hopLength.setText(
            QCoreApplication.translate("SettingsWindow", u"1024", None))
        self.lineEdit_instr_sr.setText(
            QCoreApplication.translate("SettingsWindow", u"44100", None))
        self.label_4.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stacked Model", None))
        self.comboBox_resType_5.setCurrentText("")
        self.label_title_engine.setText(
            QCoreApplication.translate("SettingsWindow", u"Engine", None))
        self.label_engine.setText(QCoreApplication.translate(
            "SettingsWindow", u"AI Engine", None))
        self.label_resType.setText(QCoreApplication.translate(
            "SettingsWindow", u"Resolution Type", None))
        self.comboBox_engine.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"v4", None))
        self.comboBox_engine.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"v2", None))

        self.comboBox_resType.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"Kaiser Best", None))
        self.comboBox_resType.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"Kaiser Fast", None))
        self.comboBox_resType.setItemText(
            2, QCoreApplication.translate("SettingsWindow", u"Scipy", None))

        self.label_title_conversion.setText(
            QCoreApplication.translate("SettingsWindow", u"Conversion", None))
        self.checkBox_gpuConversion.setText(QCoreApplication.translate(
            "SettingsWindow", u"GPU Conversion", None))
        self.checkBox_saveAllStacked.setText(QCoreApplication.translate(
            "SettingsWindow", u"Save All Stacked Outputs", None))
        self.checkBox_postProcess.setText(QCoreApplication.translate(
            "SettingsWindow", u"Post-Process", None))
        self.checkBox_stackOnly.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stack Conversion Only", None))
        self.checkBox_outputImage.setText(QCoreApplication.translate(
            "SettingsWindow", u"Output Image", None))
        self.checkBox_modelTest.setText(QCoreApplication.translate(
            "SettingsWindow", u"Model Test Mode", None))
        self.checkBox_tta.setText(
            QCoreApplication.translate("SettingsWindow", u"TTA", None))
        self.checkBox_customParameters.setText(QCoreApplication.translate(
            "SettingsWindow", u"Custom Parameters", None))
        self.checkBox_stackPasses.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stack Passes", None))
    # retranslateUi
