# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'settingswindow.ui'
##
# Created by: Qt User Interface Compiler version 5.15.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_SettingsWindow(object):
    def setupUi(self, SettingsWindow):
        if not SettingsWindow.objectName():
            SettingsWindow.setObjectName(u"SettingsWindow")
        SettingsWindow.setEnabled(True)
        SettingsWindow.resize(1097, 508)
        SettingsWindow.setStyleSheet(u"/* Universal */\n"
                                     "* {\n"
                                     "	font: 10pt \"Yu Gothic UI\";	\n"
                                     "	color: rgb(255, 255, 255);\n"
                                     "	background-color: rgb(2, 18, 40);\n"
                                     "}\n"
                                     "QToolTip {\n"
                                     "	color: rgb(0, 0, 0);\n"
                                     "}\n"
                                     "QScrollBar {\n"
                                     "	background-color: none;\n"
                                     "}\n"
                                     "QLineEdit, QComboBox {\n"
                                     "	color: rgb(0, 0, 0);\n"
                                     "	background-color: none;\n"
                                     "}\n"
                                     "QCheckBox {\n"
                                     "	color: #CCC;\n"
                                     "}\n"
                                     "QLineEdit:disabled {\n"
                                     "	color: #222;\n"
                                     "	border: 1px solid gray;\n"
                                     "	background-color: #999;\n"
                                     "}\n"
                                     "QComboBox QAbstractItemView {\n"
                                     "    border: 2px solid rgb(49, 96, 107);\n"
                                     "	outline: none;\n"
                                     "	background-color: rgb(2, 18, 40);\n"
                                     "	selection-background-color: rgb(49, 96, 107);\n"
                                     "}\n"
                                     "QGroupBox,\n"
                                     "QLabel[title=\"true\"] {\n"
                                     "	font: 15pt \"Yu Gothic UI\";\n"
                                     "}\n"
                                     "QLabel[path=\"true\"] {\n"
                                     "	font: 7pt \"Yu Gothic UI\";	\n"
                                     "	color: #ccc;\n"
                                     "}\n"
                                     "/* Pushbutton */\n"
                                     "QPushButton {\n"
                                     "	border-radius: 5px;\n"
                                     "	background-color: rgb(49, 96, 107);\n"
                                     "}\n"
                                     "QPushButton:hover {\n"
                                     "	background-color: rgb(25, 45, 60);\n"
                                     "}\n"
                                     "QPushB"
                                     "utton:pressed {\n"
                                     "	background-color: rgb(49, 96, 107);\n"
                                     "}\n"
                                     "QPushButton[clear=\"true\"] {\n"
                                     "	border: 2px solid rgb(109, 213, 237);\n"
                                     "	background-color: none;\n"
                                     "	border-radius: 5px;\n"
                                     "}\n"
                                     "QPushButton[clear=\"true\"]:hover {\n"
                                     "	background-color: rgb(25, 45, 60);\n"
                                     "}\n"
                                     "QPushButton[clear=\"true\"]:pressed {\n"
                                     "	background-color: rgb(49, 96, 107);\n"
                                     "}\n"
                                     "QPushButton[export=\"true\"] {\n"
                                     "	border: none;\n"
                                     "background-color: none;\n"
                                     "}\n"
                                     "/* LANGUAGE */\n"
                                     "QPushButton[language=\"true\"] {\n"
                                     "	border-radius: 10px;\n"
                                     "	background-color: rgba(255, 255, 255, 5);\n"
                                     "	border: none;\n"
                                     "}\n"
                                     "QPushButton[language=\"true\"]:checked {\n"
                                     "	border: 3px solid rgb(109, 213, 237);\n"
                                     "}\n"
                                     "/* MENU Radiobutton, Frame */\n"
                                     "QRadioButton[menu=\"true\"]::indicator {\n"
                                     "	width: 1px;\n"
                                     "	height: 1px;\n"
                                     "}\n"
                                     "QFrame[menu=\"true\"],\n"
                                     "QRadioButton[menu=\"true\"]::unchecked, \n"
                                     "QRadioButton[menu=\"true\"]::indicator::unchecked {\n"
                                     "    background-color: rgb(49, 96, 107);\n"
                                     "}\n"
                                     "QRadioButton"
                                     "[menu=\"true\"]::unchecked::hover, \n"
                                     "QRadioButton[menu=\"true\"]::indicator::hover {\n"
                                     "    background-color: rgb(25, 50, 60);\n"
                                     "}\n"
                                     "QRadioButton[menu=\"true\"]::checked,\n"
                                     "QRadioButton[menu=\"true\"]::indicator::checked {\n"
                                     "    background-color: rgb(2, 18, 40);\n"
                                     "}")
        self.horizontalLayout = QHBoxLayout(SettingsWindow)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_15 = QFrame(SettingsWindow)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setMinimumSize(QSize(150, 0))
        self.frame_15.setFrameShape(QFrame.NoFrame)
        self.frame_15.setFrameShadow(QFrame.Raised)
        self.frame_15.setProperty("menu", True)
        self.verticalLayout_11 = QVBoxLayout(self.frame_15)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.frame_settingsSelection = QFrame(self.frame_15)
        self.frame_settingsSelection.setObjectName(u"frame_settingsSelection")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.frame_settingsSelection.sizePolicy().hasHeightForWidth())
        self.frame_settingsSelection.setSizePolicy(sizePolicy)
        self.frame_settingsSelection.setMinimumSize(QSize(150, 0))
        self.frame_settingsSelection.setFrameShape(QFrame.NoFrame)
        self.frame_settingsSelection.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_settingsSelection)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.radioButton_seperationSettings = QRadioButton(
            self.frame_settingsSelection)
        self.radioButton_seperationSettings.setObjectName(
            u"radioButton_seperationSettings")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.radioButton_seperationSettings.sizePolicy().hasHeightForWidth())
        self.radioButton_seperationSettings.setSizePolicy(sizePolicy1)
        self.radioButton_seperationSettings.setMinimumSize(QSize(0, 40))
        self.radioButton_seperationSettings.setCheckable(True)
        self.radioButton_seperationSettings.setProperty("menu", True)

        self.verticalLayout.addWidget(self.radioButton_seperationSettings)

        self.radioButton_shortcuts = QRadioButton(self.frame_settingsSelection)
        self.radioButton_shortcuts.setObjectName(u"radioButton_shortcuts")
        self.radioButton_shortcuts.setMinimumSize(QSize(0, 40))
        self.radioButton_shortcuts.setProperty("menu", True)

        self.verticalLayout.addWidget(self.radioButton_shortcuts)

        self.radioButton_customization = QRadioButton(
            self.frame_settingsSelection)
        self.radioButton_customization.setObjectName(
            u"radioButton_customization")
        self.radioButton_customization.setMinimumSize(QSize(0, 40))
        self.radioButton_customization.setProperty("menu", True)

        self.verticalLayout.addWidget(self.radioButton_customization)

        self.verticalLayout_11.addWidget(
            self.frame_settingsSelection, 0, Qt.AlignTop)

        self.radioButton_preferences = QRadioButton(self.frame_15)
        self.radioButton_preferences.setObjectName(u"radioButton_preferences")
        sizePolicy1.setHeightForWidth(
            self.radioButton_preferences.sizePolicy().hasHeightForWidth())
        self.radioButton_preferences.setSizePolicy(sizePolicy1)
        self.radioButton_preferences.setMinimumSize(QSize(0, 40))
        self.radioButton_preferences.setProperty("menu", True)

        self.verticalLayout_11.addWidget(self.radioButton_preferences)

        self.horizontalLayout.addWidget(self.frame_15)

        self.frame_14 = QFrame(SettingsWindow)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setMinimumSize(QSize(100, 0))
        self.frame_14.setFrameShape(QFrame.NoFrame)
        self.frame_14.setFrameShadow(QFrame.Raised)
        self.verticalLayout_18 = QVBoxLayout(self.frame_14)
        self.verticalLayout_18.setSpacing(0)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.verticalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.frame_14)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setFrameShape(QFrame.NoFrame)
        self.stackedWidget.setFrameShadow(QFrame.Raised)
        self.stackedWidget.setLineWidth(0)
        self.page_seperationSettings = QWidget()
        self.page_seperationSettings.setObjectName(u"page_seperationSettings")
        self.page_seperationSettings.setStyleSheet(u"")
        self.page_seperationSettings.setProperty("minimumFrameWidth", 780)
        self.verticalLayout_10 = QVBoxLayout(self.page_seperationSettings)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(30, 0, 0, 0)
        self.scrollArea_2 = QScrollArea(self.page_seperationSettings)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setFrameShape(QFrame.NoFrame)
        self.scrollArea_2.setFrameShadow(QFrame.Plain)
        self.scrollArea_2.setLineWidth(0)
        self.scrollArea_2.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustIgnored)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(
            u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 735, 424))
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.scrollAreaWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents_2.setSizePolicy(sizePolicy2)
        self.scrollAreaWidgetContents_2.setMinimumSize(QSize(0, 0))
        self.verticalLayout_5 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 30, 35, 0)
        self.frame_3 = QFrame(self.scrollAreaWidgetContents_2)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy2.setHeightForWidth(
            self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy2)
        self.frame_3.setFrameShape(QFrame.NoFrame)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.frame_3.setLineWidth(0)
        self.verticalLayout_6 = QVBoxLayout(self.frame_3)
        self.verticalLayout_6.setSpacing(15)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 10)
        self.horizontalFrame_1 = QFrame(self.frame_3)
        self.horizontalFrame_1.setObjectName(u"horizontalFrame_1")
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalFrame_1)
        self.horizontalLayout_4.setSpacing(45)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.groupBox_conversion = QGroupBox(self.horizontalFrame_1)
        self.groupBox_conversion.setObjectName(u"groupBox_conversion")
        self.groupBox_conversion.setStyleSheet(u"")
        self.groupBox_conversion.setAlignment(
            Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.groupBox_conversion.setFlat(True)
        self.groupBox_conversion.setCheckable(False)
        self.groupBox_conversion.setProperty("titleText", True)
        self.gridLayout = QGridLayout(self.groupBox_conversion)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setHorizontalSpacing(15)
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setContentsMargins(35, 10, 30, 10)
        self.checkBox_postProcess = QCheckBox(self.groupBox_conversion)
        self.checkBox_postProcess.setObjectName(u"checkBox_postProcess")

        self.gridLayout.addWidget(self.checkBox_postProcess, 0, 1, 1, 1)

        self.checkBox_gpuConversion = QCheckBox(self.groupBox_conversion)
        self.checkBox_gpuConversion.setObjectName(u"checkBox_gpuConversion")

        self.gridLayout.addWidget(self.checkBox_gpuConversion, 0, 0, 1, 1)

        self.frame_10 = QFrame(self.groupBox_conversion)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setMinimumSize(QSize(135, 21))
        self.frame_10.setFrameShape(QFrame.NoFrame)
        self.frame_10.setFrameShadow(QFrame.Plain)
        self.frame_10.setLineWidth(0)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.checkBox_stackPasses = QCheckBox(self.frame_10)
        self.checkBox_stackPasses.setObjectName(u"checkBox_stackPasses")
        self.checkBox_stackPasses.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_3.addWidget(
            self.checkBox_stackPasses, 0, Qt.AlignLeft)

        self.comboBox_stackPasses = QComboBox(self.frame_10)
        self.comboBox_stackPasses.addItem(u"1")
        self.comboBox_stackPasses.addItem(u"2")
        self.comboBox_stackPasses.addItem(u"3")
        self.comboBox_stackPasses.addItem(u"4")
        self.comboBox_stackPasses.addItem(u"5")
        self.comboBox_stackPasses.addItem(u"6")
        self.comboBox_stackPasses.addItem(u"7")
        self.comboBox_stackPasses.addItem(u"8")
        self.comboBox_stackPasses.addItem(u"9")
        self.comboBox_stackPasses.setObjectName(u"comboBox_stackPasses")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.comboBox_stackPasses.sizePolicy().hasHeightForWidth())
        self.comboBox_stackPasses.setSizePolicy(sizePolicy3)
        self.comboBox_stackPasses.setMinimumSize(QSize(0, 0))
        self.comboBox_stackPasses.setMaximumSize(QSize(45, 21))
        self.comboBox_stackPasses.setCurrentText(u"1")
        self.comboBox_stackPasses.setMaxVisibleItems(4)

        self.horizontalLayout_3.addWidget(self.comboBox_stackPasses)

        self.gridLayout.addWidget(self.frame_10, 1, 1, 1, 1, Qt.AlignLeft)

        self.checkBox_saveAllStacked = QCheckBox(self.groupBox_conversion)
        self.checkBox_saveAllStacked.setObjectName(u"checkBox_saveAllStacked")

        self.gridLayout.addWidget(self.checkBox_saveAllStacked, 2, 1, 1, 1)

        self.checkBox_stackOnly = QCheckBox(self.groupBox_conversion)
        self.checkBox_stackOnly.setObjectName(u"checkBox_stackOnly")
        self.checkBox_stackOnly.setEnabled(True)

        self.gridLayout.addWidget(self.checkBox_stackOnly, 3, 1, 1, 1)

        self.checkBox_outputImage = QCheckBox(self.groupBox_conversion)
        self.checkBox_outputImage.setObjectName(u"checkBox_outputImage")

        self.gridLayout.addWidget(self.checkBox_outputImage, 4, 0, 1, 1)

        self.checkBox_tta = QCheckBox(self.groupBox_conversion)
        self.checkBox_tta.setObjectName(u"checkBox_tta")

        self.gridLayout.addWidget(self.checkBox_tta, 1, 0, 1, 1)

        self.checkBox_modelFolder = QCheckBox(self.groupBox_conversion)
        self.checkBox_modelFolder.setObjectName(u"checkBox_modelFolder")

        self.gridLayout.addWidget(self.checkBox_modelFolder, 2, 0, 1, 1)

        self.checkBox_customParameters = QCheckBox(self.groupBox_conversion)
        self.checkBox_customParameters.setObjectName(
            u"checkBox_customParameters")

        self.gridLayout.addWidget(self.checkBox_customParameters, 3, 0, 1, 1)

        self.gridLayout.setColumnMinimumWidth(1, 167)

        self.horizontalLayout_4.addWidget(self.groupBox_conversion)

        self.groupBox_2 = QGroupBox(self.horizontalFrame_1)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFlat(True)
        self.groupBox_2.setProperty("titleText", True)
        self.verticalLayout_8 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_8.setSpacing(10)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(35, 10, 30, 10)
        self.frame_12 = QFrame(self.groupBox_2)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setFrameShape(QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Raised)
        self.gridLayout_6 = QGridLayout(self.frame_12)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setHorizontalSpacing(20)
        self.gridLayout_6.setVerticalSpacing(15)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.label_engine = QLabel(self.frame_12)
        self.label_engine.setObjectName(u"label_engine")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.label_engine.sizePolicy().hasHeightForWidth())
        self.label_engine.setSizePolicy(sizePolicy4)
        self.label_engine.setMinimumSize(QSize(0, 0))
        self.label_engine.setAlignment(Qt.AlignCenter)
        self.label_engine.setIndent(10)

        self.gridLayout_6.addWidget(self.label_engine, 0, 0, 1, 1)

        self.comboBox_engine = QComboBox(self.frame_12)
        self.comboBox_engine.addItem(u"v2")
        self.comboBox_engine.addItem(u"v4")
        self.comboBox_engine.setObjectName(u"comboBox_engine")
        self.comboBox_engine.setEnabled(True)
        self.comboBox_engine.setMinimumSize(QSize(0, 25))

        self.gridLayout_6.addWidget(self.comboBox_engine, 1, 0, 1, 1)

        self.comboBox_resType = QComboBox(self.frame_12)
        self.comboBox_resType.addItem(u"Kaiser Fast")
        self.comboBox_resType.addItem(u"Kaiser Best")
        self.comboBox_resType.addItem(u"Scipy")
        self.comboBox_resType.setObjectName(u"comboBox_resType")
        self.comboBox_resType.setMinimumSize(QSize(0, 25))

        self.gridLayout_6.addWidget(self.comboBox_resType, 1, 1, 1, 1)

        self.label_resType = QLabel(self.frame_12)
        self.label_resType.setObjectName(u"label_resType")
        sizePolicy4.setHeightForWidth(
            self.label_resType.sizePolicy().hasHeightForWidth())
        self.label_resType.setSizePolicy(sizePolicy4)
        self.label_resType.setAlignment(Qt.AlignCenter)
        self.label_resType.setIndent(10)

        self.gridLayout_6.addWidget(self.label_resType, 0, 1, 1, 1)

        self.gridLayout_6.setColumnMinimumWidth(0, 92)
        self.gridLayout_6.setColumnMinimumWidth(1, 92)

        self.verticalLayout_8.addWidget(self.frame_12, 0, Qt.AlignTop)

        self.horizontalLayout_4.addWidget(self.groupBox_2)

        self.horizontalLayout_4.setStretch(0, 4)
        self.horizontalLayout_4.setStretch(1, 3)

        self.verticalLayout_6.addWidget(self.horizontalFrame_1)

        self.groupBox_3 = QGroupBox(self.frame_3)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setFlat(True)
        self.groupBox_3.setProperty("titleText", True)
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(35, 10, 30, 10)
        self.frame_instrumentalComboBox = QFrame(self.groupBox_3)
        self.frame_instrumentalComboBox.setObjectName(
            u"frame_instrumentalComboBox")
        self.frame_instrumentalComboBox.setFrameShape(QFrame.StyledPanel)
        self.frame_instrumentalComboBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_instrumentalComboBox)
        self.verticalLayout_3.setSpacing(15)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 40)
        self.label_2 = QLabel(self.frame_instrumentalComboBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 30))
        self.label_2.setMaximumSize(QSize(16777215, 20))
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_2)

        self.comboBox_instrumental = QComboBox(self.frame_instrumentalComboBox)
        self.comboBox_instrumental.setObjectName(u"comboBox_instrumental")
        self.comboBox_instrumental.setMinimumSize(QSize(0, 25))
        self.comboBox_instrumental.setMaximumSize(QSize(16777215, 30))
        self.comboBox_instrumental.setEditable(False)
        self.comboBox_instrumental.setCurrentText(u"")
        self.comboBox_instrumental.setMaxVisibleItems(5)
        self.comboBox_instrumental.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.verticalLayout_3.addWidget(self.comboBox_instrumental)

        self.horizontalLayout_2.addWidget(self.frame_instrumentalComboBox)

        self.frame_constants = QFrame(self.groupBox_3)
        self.frame_constants.setObjectName(u"frame_constants")
        sizePolicy5 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.frame_constants.sizePolicy().hasHeightForWidth())
        self.frame_constants.setSizePolicy(sizePolicy5)
        self.frame_constants.setMinimumSize(QSize(0, 0))
        self.frame_constants.setMaximumSize(QSize(230, 140))
        self.frame_constants.setFrameShape(QFrame.StyledPanel)
        self.frame_constants.setFrameShadow(QFrame.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_constants)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, 5, -1, 3)
        self.comboBox_winSize_stacked = QComboBox(self.frame_constants)
        self.comboBox_winSize_stacked.addItem("")
        self.comboBox_winSize_stacked.addItem("")
        self.comboBox_winSize_stacked.addItem("")
        self.comboBox_winSize_stacked.addItem("")
        self.comboBox_winSize_stacked.setObjectName(
            u"comboBox_winSize_stacked")
        self.comboBox_winSize_stacked.setMinimumSize(QSize(0, 25))
        self.comboBox_winSize_stacked.setMaximumSize(QSize(60, 25))
        self.comboBox_winSize_stacked.setEditable(True)
        self.comboBox_winSize_stacked.setCurrentText(u"512")
        self.comboBox_winSize_stacked.setProperty("canEdit", True)

        self.gridLayout_5.addWidget(self.comboBox_winSize_stacked, 2, 2, 1, 1)

        self.label_hopLength = QLabel(self.frame_constants)
        self.label_hopLength.setObjectName(u"label_hopLength")
        self.label_hopLength.setMinimumSize(QSize(80, 0))
        self.label_hopLength.setMaximumSize(QSize(80, 16777215))
        self.label_hopLength.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_hopLength, 1, 1, 1, 1)

        self.lineEdit_hopLength_stacked = QLineEdit(self.frame_constants)
        self.lineEdit_hopLength_stacked.setObjectName(
            u"lineEdit_hopLength_stacked")
        self.lineEdit_hopLength_stacked.setMinimumSize(QSize(60, 25))
        self.lineEdit_hopLength_stacked.setMaximumSize(QSize(60, 25))
        self.lineEdit_hopLength_stacked.setText(u"1024")
        self.lineEdit_hopLength_stacked.setAlignment(Qt.AlignCenter)
        self.lineEdit_hopLength_stacked.setDragEnabled(False)
        self.lineEdit_hopLength_stacked.setReadOnly(False)

        self.gridLayout_5.addWidget(
            self.lineEdit_hopLength_stacked, 1, 2, 1, 1)

        self.lineEdit_nfft_stacked = QLineEdit(self.frame_constants)
        self.lineEdit_nfft_stacked.setObjectName(u"lineEdit_nfft_stacked")
        self.lineEdit_nfft_stacked.setMinimumSize(QSize(60, 25))
        self.lineEdit_nfft_stacked.setMaximumSize(QSize(60, 25))
        self.lineEdit_nfft_stacked.setText(u"2048")
        self.lineEdit_nfft_stacked.setAlignment(Qt.AlignCenter)
        self.lineEdit_nfft_stacked.setDragEnabled(False)
        self.lineEdit_nfft_stacked.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_nfft_stacked, 4, 2, 1, 1)

        self.label_sr = QLabel(self.frame_constants)
        self.label_sr.setObjectName(u"label_sr")
        self.label_sr.setMinimumSize(QSize(80, 0))
        self.label_sr.setMaximumSize(QSize(80, 16777215))
        self.label_sr.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_sr, 0, 1, 1, 1)

        self.label_winSize = QLabel(self.frame_constants)
        self.label_winSize.setObjectName(u"label_winSize")
        self.label_winSize.setMinimumSize(QSize(80, 0))
        self.label_winSize.setMaximumSize(QSize(80, 16777215))
        self.label_winSize.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_winSize, 2, 1, 1, 1)

        self.label_nfft = QLabel(self.frame_constants)
        self.label_nfft.setObjectName(u"label_nfft")
        self.label_nfft.setMinimumSize(QSize(80, 0))
        self.label_nfft.setMaximumSize(QSize(80, 16777215))
        self.label_nfft.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_nfft, 4, 1, 1, 1)

        self.lineEdit_sr_stacked = QLineEdit(self.frame_constants)
        self.lineEdit_sr_stacked.setObjectName(u"lineEdit_sr_stacked")
        self.lineEdit_sr_stacked.setMinimumSize(QSize(60, 25))
        self.lineEdit_sr_stacked.setMaximumSize(QSize(60, 25))
        self.lineEdit_sr_stacked.setText(u"44100")
        self.lineEdit_sr_stacked.setAlignment(Qt.AlignCenter)
        self.lineEdit_sr_stacked.setDragEnabled(False)
        self.lineEdit_sr_stacked.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_sr_stacked, 0, 2, 1, 1)

        self.lineEdit_sr = QLineEdit(self.frame_constants)
        self.lineEdit_sr.setObjectName(u"lineEdit_sr")
        self.lineEdit_sr.setEnabled(False)
        self.lineEdit_sr.setMinimumSize(QSize(60, 25))
        self.lineEdit_sr.setMaximumSize(QSize(60, 25))
        self.lineEdit_sr.setText(u"44100")
        self.lineEdit_sr.setAlignment(Qt.AlignCenter)
        self.lineEdit_sr.setDragEnabled(False)
        self.lineEdit_sr.setReadOnly(False)

        self.gridLayout_5.addWidget(self.lineEdit_sr, 0, 0, 1, 1)

        self.lineEdit_hopLength = QLineEdit(self.frame_constants)
        self.lineEdit_hopLength.setObjectName(u"lineEdit_hopLength")
        self.lineEdit_hopLength.setMinimumSize(QSize(60, 25))
        self.lineEdit_hopLength.setMaximumSize(QSize(60, 25))
        self.lineEdit_hopLength.setText(u"1024")
        self.lineEdit_hopLength.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.lineEdit_hopLength, 1, 0, 1, 1)

        self.comboBox_winSize = QComboBox(self.frame_constants)
        self.comboBox_winSize.addItem("")
        self.comboBox_winSize.addItem("")
        self.comboBox_winSize.addItem("")
        self.comboBox_winSize.addItem("")
        self.comboBox_winSize.setObjectName(u"comboBox_winSize")
        self.comboBox_winSize.setMinimumSize(QSize(0, 25))
        self.comboBox_winSize.setMaximumSize(QSize(60, 25))
        self.comboBox_winSize.setEditable(True)
        self.comboBox_winSize.setCurrentText(u"512")
        self.comboBox_winSize.setProperty("canEdit", True)

        self.gridLayout_5.addWidget(self.comboBox_winSize, 2, 0, 1, 1)

        self.lineEdit_nfft = QLineEdit(self.frame_constants)
        self.lineEdit_nfft.setObjectName(u"lineEdit_nfft")
        self.lineEdit_nfft.setMinimumSize(QSize(60, 25))
        self.lineEdit_nfft.setMaximumSize(QSize(60, 25))
        self.lineEdit_nfft.setText(u"2048")
        self.lineEdit_nfft.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.lineEdit_nfft, 4, 0, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.horizontalLayout_2.addWidget(
            self.frame_constants, 0, Qt.AlignLeft)

        self.frame_stackComboBox = QFrame(self.groupBox_3)
        self.frame_stackComboBox.setObjectName(u"frame_stackComboBox")
        self.frame_stackComboBox.setFrameShape(QFrame.StyledPanel)
        self.frame_stackComboBox.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_stackComboBox)
        self.verticalLayout_4.setSpacing(15)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 40)
        self.label_4 = QLabel(self.frame_stackComboBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(0, 30))
        self.label_4.setMaximumSize(QSize(16777215, 20))
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_4)

        self.comboBox_stacked = QComboBox(self.frame_stackComboBox)
        self.comboBox_stacked.setObjectName(u"comboBox_stacked")
        self.comboBox_stacked.setEnabled(True)
        self.comboBox_stacked.setMinimumSize(QSize(0, 25))
        self.comboBox_stacked.setMaximumSize(QSize(16777215, 30))
        self.comboBox_stacked.setEditable(False)
        self.comboBox_stacked.setCurrentText(u"")
        self.comboBox_stacked.setMaxVisibleItems(5)
        self.comboBox_stacked.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.verticalLayout_4.addWidget(self.comboBox_stacked)

        self.horizontalLayout_2.addWidget(self.frame_stackComboBox)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(2, 1)

        self.verticalLayout_6.addWidget(self.groupBox_3)

        self.verticalLayout_5.addWidget(self.frame_3, 0, Qt.AlignTop)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_10.addWidget(self.scrollArea_2)

        self.stackedWidget.addWidget(self.page_seperationSettings)
        self.page_shortcuts = QWidget()
        self.page_shortcuts.setObjectName(u"page_shortcuts")
        self.page_shortcuts.setProperty("minimumFrameWidth", 0)
        self.verticalLayout_2 = QVBoxLayout(self.page_shortcuts)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(30, 0, 0, 0)
        self.scrollArea = QScrollArea(self.page_shortcuts)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(
            u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 53, 40))
        self.scrollAreaWidgetContents.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                    "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                    "}")
        self.verticalLayout_14 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 30, 35, 0)
        self.frame = QFrame(self.scrollAreaWidgetContents)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)

        self.verticalLayout_14.addWidget(self.frame)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_2.addWidget(self.scrollArea)

        self.stackedWidget.addWidget(self.page_shortcuts)
        self.page_customization = QWidget()
        self.page_customization.setObjectName(u"page_customization")
        self.page_customization.setProperty("minimumFrameWidth", 0)
        self.verticalLayout_12 = QVBoxLayout(self.page_customization)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.verticalLayout_12.setContentsMargins(30, 0, 0, 0)
        self.scrollArea_3 = QScrollArea(self.page_customization)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setFrameShape(QFrame.NoFrame)
        self.scrollArea_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(
            u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 53, 40))
        self.scrollAreaWidgetContents_4.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                      "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                      "}")
        self.verticalLayout_15 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(0, 30, 35, 0)
        self.frame_2 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)

        self.verticalLayout_15.addWidget(self.frame_2)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_4)

        self.verticalLayout_12.addWidget(self.scrollArea_3)

        self.stackedWidget.addWidget(self.page_customization)
        self.page_preferences = QWidget()
        self.page_preferences.setObjectName(u"page_preferences")
        self.page_preferences.setStyleSheet(u"")
        self.page_preferences.setProperty("minimumFrameWidth", 770)
        self.verticalLayout_7 = QVBoxLayout(self.page_preferences)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(30, 0, 0, 0)
        self.scrollArea_4 = QScrollArea(self.page_preferences)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setFrameShape(QFrame.NoFrame)
        self.scrollArea_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_5 = QWidget()
        self.scrollAreaWidgetContents_5.setObjectName(
            u"scrollAreaWidgetContents_5")
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 917, 447))
        self.scrollAreaWidgetContents_5.setMinimumSize(QSize(600, 0))
        self.scrollAreaWidgetContents_5.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                      "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                      "}")
        self.verticalLayout_16 = QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(0, 30, 35, 0)
        self.frame_4 = QFrame(self.scrollAreaWidgetContents_5)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.frame_4.setLineWidth(0)
        self.verticalLayout_17 = QVBoxLayout(self.frame_4)
        self.verticalLayout_17.setSpacing(15)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 10)
        self.frame_5 = QFrame(self.frame_4)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.NoFrame)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_6.setSpacing(45)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.frame_5)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setMinimumSize(QSize(40, 0))
        self.frame_6.setFrameShape(QFrame.NoFrame)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_20 = QVBoxLayout(self.frame_6)
        self.verticalLayout_20.setSpacing(15)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.groupBox_6 = QGroupBox(self.frame_6)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setMinimumSize(QSize(0, 0))
        self.groupBox_6.setFlat(True)
        self.verticalLayout_13 = QVBoxLayout(self.groupBox_6)
        self.verticalLayout_13.setSpacing(10)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(35, 10, 30, 10)
        self.horizontalFrame = QFrame(self.groupBox_6)
        self.horizontalFrame.setObjectName(u"horizontalFrame")
        self.horizontalFrame.setLineWidth(0)
        self.formLayout_4 = QFormLayout(self.horizontalFrame)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.horizontalFrame)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(100, 0))

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.label)

        self.frame_11 = QFrame(self.horizontalFrame)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setFrameShape(QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.comboBox_command = QComboBox(self.frame_11)
        self.comboBox_command.addItem("")
        self.comboBox_command.addItem("")
        self.comboBox_command.setObjectName(u"comboBox_command")
        self.comboBox_command.setMinimumSize(QSize(0, 25))

        self.horizontalLayout_5.addWidget(self.comboBox_command)

        self.pushButton_clearCommand = QPushButton(self.frame_11)
        self.pushButton_clearCommand.setObjectName(u"pushButton_clearCommand")
        sizePolicy6 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(
            self.pushButton_clearCommand.sizePolicy().hasHeightForWidth())
        self.pushButton_clearCommand.setSizePolicy(sizePolicy6)
        self.pushButton_clearCommand.setMinimumSize(QSize(50, 0))
        self.pushButton_clearCommand.setFlat(True)
        self.pushButton_clearCommand.setProperty("clear", True)

        self.horizontalLayout_5.addWidget(self.pushButton_clearCommand)

        self.formLayout_4.setWidget(0, QFormLayout.FieldRole, self.frame_11)

        self.verticalLayout_13.addWidget(self.horizontalFrame)

        self.frame_8 = QFrame(self.groupBox_6)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setFrameShape(QFrame.NoFrame)
        self.frame_8.setFrameShadow(QFrame.Raised)
        self.frame_8.setLineWidth(0)
        self.gridLayout_4 = QGridLayout(self.frame_8)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setHorizontalSpacing(0)
        self.gridLayout_4.setVerticalSpacing(10)
        self.gridLayout_4.setContentsMargins(0, 5, 0, 0)
        self.checkBox_disableAnimations = QCheckBox(self.frame_8)
        self.checkBox_disableAnimations.setObjectName(
            u"checkBox_disableAnimations")

        self.gridLayout_4.addWidget(
            self.checkBox_disableAnimations, 4, 0, 1, 1)

        self.checkBox_disableShortcuts = QCheckBox(self.frame_8)
        self.checkBox_disableShortcuts.setObjectName(
            u"checkBox_disableShortcuts")

        self.gridLayout_4.addWidget(self.checkBox_disableShortcuts, 4, 1, 1, 1)

        self.checkBox_notifyUpdates = QCheckBox(self.frame_8)
        self.checkBox_notifyUpdates.setObjectName(u"checkBox_notifyUpdates")

        self.gridLayout_4.addWidget(self.checkBox_notifyUpdates, 1, 0, 1, 2)

        self.checkBox_settingsStartup = QCheckBox(self.frame_8)
        self.checkBox_settingsStartup.setObjectName(
            u"checkBox_settingsStartup")

        self.gridLayout_4.addWidget(self.checkBox_settingsStartup, 2, 0, 1, 1)

        self.checkBox_notifiyOnFinish = QCheckBox(self.frame_8)
        self.checkBox_notifiyOnFinish.setObjectName(
            u"checkBox_notifiyOnFinish")

        self.gridLayout_4.addWidget(self.checkBox_notifiyOnFinish, 0, 0, 1, 2)

        self.checkBox_multiThreading = QCheckBox(self.frame_8)
        self.checkBox_multiThreading.setObjectName(u"checkBox_multiThreading")
        self.checkBox_multiThreading.setEnabled(False)

        self.gridLayout_4.addWidget(self.checkBox_multiThreading, 5, 0, 1, 1)

        self.verticalLayout_13.addWidget(self.frame_8)

        self.verticalLayout_20.addWidget(self.groupBox_6)

        self.groupBox_export = QGroupBox(self.frame_6)
        self.groupBox_export.setObjectName(u"groupBox_export")
        self.groupBox_export.setMinimumSize(QSize(0, 104))
        self.groupBox_export.setAcceptDrops(True)
        self.groupBox_export.setFlat(True)
        self.formLayout = QFormLayout(self.groupBox_export)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.formLayout.setHorizontalSpacing(20)
        self.formLayout.setVerticalSpacing(10)
        self.formLayout.setContentsMargins(35, 10, 30, 10)
        self.frame_13 = QFrame(self.groupBox_export)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setMinimumSize(QSize(100, 0))
        self.frame_13.setFrameShape(QFrame.NoFrame)
        self.frame_13.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_13)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.frame_13)
        self.label_3.setObjectName(u"label_3")
        sizePolicy5.setHeightForWidth(
            self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy5)
        self.label_3.setMinimumSize(QSize(110, 0))

        self.horizontalLayout_8.addWidget(self.label_3)

        self.pushButton_exportDirectory = QPushButton(self.frame_13)
        self.pushButton_exportDirectory.setObjectName(
            u"pushButton_exportDirectory")
        self.pushButton_exportDirectory.setMinimumSize(QSize(18, 18))
        self.pushButton_exportDirectory.setMaximumSize(QSize(18, 18))
        self.pushButton_exportDirectory.setCursor(
            QCursor(Qt.PointingHandCursor))
        self.pushButton_exportDirectory.setFlat(True)
        self.pushButton_exportDirectory.setProperty("export", True)

        self.horizontalLayout_8.addWidget(self.pushButton_exportDirectory)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.frame_13)

        self.label_autoSave = QLabel(self.groupBox_export)
        self.label_autoSave.setObjectName(u"label_autoSave")

        self.formLayout.setWidget(
            1, QFormLayout.LabelRole, self.label_autoSave)

        self.frame_7 = QFrame(self.groupBox_export)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setMinimumSize(QSize(0, 30))
        self.frame_7.setFrameShape(QFrame.NoFrame)
        self.frame_7.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(5, 0, 0, 0)
        self.checkBox_autoSaveInstrumentals = QCheckBox(self.frame_7)
        self.checkBox_autoSaveInstrumentals.setObjectName(
            u"checkBox_autoSaveInstrumentals")

        self.horizontalLayout_7.addWidget(
            self.checkBox_autoSaveInstrumentals, 0, Qt.AlignHCenter)

        self.checkBox_autoSaveVocals = QCheckBox(self.frame_7)
        self.checkBox_autoSaveVocals.setObjectName(u"checkBox_autoSaveVocals")

        self.horizontalLayout_7.addWidget(
            self.checkBox_autoSaveVocals, 0, Qt.AlignHCenter)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(1, 1)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.frame_7)

        self.label_exportDirectory = QLabel(self.groupBox_export)
        self.label_exportDirectory.setObjectName(u"label_exportDirectory")
        sizePolicy2.setHeightForWidth(
            self.label_exportDirectory.sizePolicy().hasHeightForWidth())
        self.label_exportDirectory.setSizePolicy(sizePolicy2)
        self.label_exportDirectory.setText(u"B:/Downloads")
        self.label_exportDirectory.setTextFormat(Qt.AutoText)
        self.label_exportDirectory.setScaledContents(True)
        self.label_exportDirectory.setAlignment(
            Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.label_exportDirectory.setWordWrap(True)
        self.label_exportDirectory.setIndent(5)
        self.label_exportDirectory.setProperty("path", True)

        self.formLayout.setWidget(
            0, QFormLayout.FieldRole, self.label_exportDirectory)

        self.verticalLayout_20.addWidget(self.groupBox_export)

        self.horizontalLayout_6.addWidget(self.frame_6)

        self.groupBox_4 = QGroupBox(self.frame_5)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy5.setHeightForWidth(
            self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy5)
        self.groupBox_4.setMinimumSize(QSize(0, 0))
        self.groupBox_4.setFlat(True)
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(35, 10, 30, 10)
        self.frame_languages = QFrame(self.groupBox_4)
        self.frame_languages.setObjectName(u"frame_languages")
        self.frame_languages.setFrameShape(QFrame.StyledPanel)
        self.frame_languages.setFrameShadow(QFrame.Raised)
        self.gridLayout_3 = QGridLayout(self.frame_languages)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.pushButton_english = QPushButton(self.frame_languages)
        self.pushButton_english.setObjectName(u"pushButton_english")
        self.pushButton_english.setMinimumSize(QSize(80, 49))
        self.pushButton_english.setMaximumSize(QSize(80, 49))
        self.pushButton_english.setText(u"")
        self.pushButton_english.setCheckable(True)
        self.pushButton_english.setChecked(False)
        self.pushButton_english.setFlat(False)
        self.pushButton_english.setProperty("language", True)

        self.gridLayout_3.addWidget(self.pushButton_english, 0, 0, 1, 1)

        self.pushButton_filipino = QPushButton(self.frame_languages)
        self.pushButton_filipino.setObjectName(u"pushButton_filipino")
        self.pushButton_filipino.setMinimumSize(QSize(80, 49))
        self.pushButton_filipino.setMaximumSize(QSize(80, 49))
        self.pushButton_filipino.setCheckable(True)
        self.pushButton_filipino.setProperty("language", True)

        self.gridLayout_3.addWidget(self.pushButton_filipino, 1, 0, 1, 1)

        self.pushButton_german = QPushButton(self.frame_languages)
        self.pushButton_german.setObjectName(u"pushButton_german")
        self.pushButton_german.setMinimumSize(QSize(80, 49))
        self.pushButton_german.setMaximumSize(QSize(80, 49))
        self.pushButton_german.setText(u"")
        self.pushButton_german.setCheckable(True)
        self.pushButton_german.setChecked(False)
        self.pushButton_german.setFlat(False)
        self.pushButton_german.setProperty("language", True)

        self.gridLayout_3.addWidget(self.pushButton_german, 0, 1, 1, 1)

        self.pushButton_japanese = QPushButton(self.frame_languages)
        self.pushButton_japanese.setObjectName(u"pushButton_japanese")
        self.pushButton_japanese.setMinimumSize(QSize(80, 49))
        self.pushButton_japanese.setMaximumSize(QSize(80, 49))
        self.pushButton_japanese.setToolTipDuration(-1)
        self.pushButton_japanese.setCheckable(True)
        self.pushButton_japanese.setProperty("language", True)

        self.gridLayout_3.addWidget(self.pushButton_japanese, 1, 1, 1, 1)

        self.pushButton_turkish = QPushButton(self.frame_languages)
        self.pushButton_turkish.setObjectName(u"pushButton_turkish")
        self.pushButton_turkish.setMinimumSize(QSize(80, 49))
        self.pushButton_turkish.setMaximumSize(QSize(80, 49))
        self.pushButton_turkish.setCheckable(True)
        self.pushButton_turkish.setFlat(True)
        self.pushButton_turkish.setProperty("language", True)

        self.gridLayout_3.addWidget(self.pushButton_turkish, 2, 0, 1, 1)

        self.gridLayout_2.addWidget(
            self.frame_languages, 0, 1, 1, 1, Qt.AlignTop)

        self.horizontalLayout_6.addWidget(self.groupBox_4)

        self.horizontalLayout_6.setStretch(1, 2)

        self.verticalLayout_17.addWidget(self.frame_5, 0, Qt.AlignTop)

        self.verticalLayout_16.addWidget(self.frame_4, 0, Qt.AlignTop)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_5)

        self.verticalLayout_7.addWidget(self.scrollArea_4)

        self.stackedWidget.addWidget(self.page_preferences)

        self.verticalLayout_18.addWidget(self.stackedWidget)

        self.frame_16 = QFrame(self.frame_14)
        self.frame_16.setObjectName(u"frame_16")
        sizePolicy.setHeightForWidth(
            self.frame_16.sizePolicy().hasHeightForWidth())
        self.frame_16.setSizePolicy(sizePolicy)
        self.frame_16.setMinimumSize(QSize(0, 60))
        self.frame_16.setFrameShape(QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_16)
        self.horizontalLayout_9.setSpacing(10)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 15, 15, 15)
        self.pushButton = QPushButton(self.frame_16)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumSize(QSize(100, 29))
        self.pushButton.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_9.addWidget(self.pushButton)

        self.pushButton_resetDefault = QPushButton(self.frame_16)
        self.pushButton_resetDefault.setObjectName(u"pushButton_resetDefault")
        sizePolicy7 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(
            self.pushButton_resetDefault.sizePolicy().hasHeightForWidth())
        self.pushButton_resetDefault.setSizePolicy(sizePolicy7)
        self.pushButton_resetDefault.setMinimumSize(QSize(110, 29))
        self.pushButton_resetDefault.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_9.addWidget(self.pushButton_resetDefault)

        self.verticalLayout_18.addWidget(self.frame_16, 0, Qt.AlignRight)

        self.horizontalLayout.addWidget(self.frame_14)

        self.retranslateUi(SettingsWindow)

        self.stackedWidget.setCurrentIndex(3)
        self.comboBox_winSize_stacked.setCurrentIndex(2)
        self.comboBox_winSize.setCurrentIndex(2)
        self.pushButton_english.setDefault(False)
        self.pushButton_german.setDefault(False)

        QMetaObject.connectSlotsByName(SettingsWindow)
    # setupUi

    def retranslateUi(self, SettingsWindow):
        SettingsWindow.setWindowTitle(QCoreApplication.translate(
            "SettingsWindow", u"Settings", None))
        self.radioButton_seperationSettings.setText(
            QCoreApplication.translate("SettingsWindow", u"Seperation Settings", None))
        self.radioButton_shortcuts.setText(
            QCoreApplication.translate("SettingsWindow", u"Shortcuts", None))
        self.radioButton_customization.setText(
            QCoreApplication.translate("SettingsWindow", u"Customization", None))
        self.radioButton_preferences.setText(
            QCoreApplication.translate("SettingsWindow", u"Preferences", None))
        self.groupBox_conversion.setTitle(
            QCoreApplication.translate("SettingsWindow", u"Conversion ", None))
        self.checkBox_postProcess.setText(QCoreApplication.translate(
            "SettingsWindow", u"Post-Process", None))
        self.checkBox_gpuConversion.setText(QCoreApplication.translate(
            "SettingsWindow", u"GPU Conversion", None))
        self.checkBox_stackPasses.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stack Passes", None))

        self.checkBox_saveAllStacked.setText(QCoreApplication.translate(
            "SettingsWindow", u"Save All Stacked Outputs", None))
        self.checkBox_stackOnly.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stack Conversion Only", None))
# if QT_CONFIG(tooltip)
        self.checkBox_outputImage.setToolTip(QCoreApplication.translate(
            "SettingsWindow", u"Save spectogram of seperated music files", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_outputImage.setText(QCoreApplication.translate(
            "SettingsWindow", u"Output Image", None))
        self.checkBox_tta.setText(
            QCoreApplication.translate("SettingsWindow", u"TTA", None))
        self.checkBox_modelFolder.setText(QCoreApplication.translate(
            "SettingsWindow", u"Model Test Mode", None))
        self.checkBox_customParameters.setText(QCoreApplication.translate(
            "SettingsWindow", u"Custom Parameters", None))
        self.groupBox_2.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Engine ", None))
        self.label_engine.setText(QCoreApplication.translate(
            "SettingsWindow", u"AI Engine", None))

        self.label_resType.setText(QCoreApplication.translate(
            "SettingsWindow", u"Resolution Type", None))
        self.groupBox_3.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Models ", None))
        self.label_2.setText(QCoreApplication.translate(
            "SettingsWindow", u"Instrumental Model", None))
        self.comboBox_winSize_stacked.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"272", None))
        self.comboBox_winSize_stacked.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"320", None))
        self.comboBox_winSize_stacked.setItemText(
            2, QCoreApplication.translate("SettingsWindow", u"512", None))
        self.comboBox_winSize_stacked.setItemText(
            3, QCoreApplication.translate("SettingsWindow", u"1024", None))

        self.label_hopLength.setText(QCoreApplication.translate(
            "SettingsWindow", u"Hop Length", None))
# if QT_CONFIG(tooltip)
        self.label_sr.setToolTip(QCoreApplication.translate(
            "SettingsWindow", u"Sampling Rate", None))
#endif // QT_CONFIG(tooltip)
        self.label_sr.setText(QCoreApplication.translate(
            "SettingsWindow", u"SR", None))
        self.label_winSize.setText(QCoreApplication.translate(
            "SettingsWindow", u"Window Size", None))
        self.label_nfft.setText(QCoreApplication.translate(
            "SettingsWindow", u"N_FFT", None))
        self.comboBox_winSize.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"272", None))
        self.comboBox_winSize.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"320", None))
        self.comboBox_winSize.setItemText(
            2, QCoreApplication.translate("SettingsWindow", u"512", None))
        self.comboBox_winSize.setItemText(
            3, QCoreApplication.translate("SettingsWindow", u"1024", None))

        self.label_4.setText(QCoreApplication.translate(
            "SettingsWindow", u"Stacked Model", None))
        self.groupBox_6.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Settings ", None))
        self.label.setText(QCoreApplication.translate(
            "SettingsWindow", u"Command Line", None))
        self.comboBox_command.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"Off", None))
        self.comboBox_command.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"On", None))

        self.pushButton_clearCommand.setText(
            QCoreApplication.translate("SettingsWindow", u"Clear", None))
        self.checkBox_disableAnimations.setText(QCoreApplication.translate(
            "SettingsWindow", u"Disable Animations", None))
        self.checkBox_disableShortcuts.setText(QCoreApplication.translate(
            "SettingsWindow", u"Disable Shortcuts", None))
        self.checkBox_notifyUpdates.setText(QCoreApplication.translate(
            "SettingsWindow", u"Notify me of application updates", None))
        self.checkBox_settingsStartup.setText(QCoreApplication.translate(
            "SettingsWindow", u"Open Settings on Startup", None))
        self.checkBox_notifiyOnFinish.setText(QCoreApplication.translate(
            "SettingsWindow", u"Notify me on finish of seperation", None))
# if QT_CONFIG(tooltip)
        self.checkBox_multiThreading.setToolTip(QCoreApplication.translate(
            "SettingsWindow", u"Process multiple files simultaneously", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_multiThreading.setText(QCoreApplication.translate(
            "SettingsWindow", u"Multithreading (Experimental)", None))
        self.groupBox_export.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Export Settings ", None))
        self.label_3.setText(QCoreApplication.translate(
            "SettingsWindow", u"Export Directory", None))
        self.pushButton_exportDirectory.setText("")
        self.label_autoSave.setText(QCoreApplication.translate(
            "SettingsWindow", u"Automatically Save:", None))
        self.checkBox_autoSaveInstrumentals.setText(
            QCoreApplication.translate("SettingsWindow", u"Instrumentals", None))
        self.checkBox_autoSaveVocals.setText(
            QCoreApplication.translate("SettingsWindow", u"Vocals", None))
        self.groupBox_4.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Language ", None))
# if QT_CONFIG(tooltip)
        self.pushButton_english.setToolTip(
            QCoreApplication.translate("SettingsWindow", u"English", None))
#endif // QT_CONFIG(tooltip)
# if QT_CONFIG(tooltip)
        self.pushButton_filipino.setToolTip(
            QCoreApplication.translate("SettingsWindow", u"Filipino", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_filipino.setText("")
# if QT_CONFIG(tooltip)
        self.pushButton_german.setToolTip(
            QCoreApplication.translate("SettingsWindow", u"German", None))
#endif // QT_CONFIG(tooltip)
# if QT_CONFIG(tooltip)
        self.pushButton_japanese.setToolTip(
            QCoreApplication.translate("SettingsWindow", u"Japanese", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton_japanese.setText("")
        self.pushButton_turkish.setText("")
        self.pushButton.setText(QCoreApplication.translate(
            "SettingsWindow", u"Get Log File", None))
        self.pushButton_resetDefault.setText(QCoreApplication.translate(
            "SettingsWindow", u"Reset to default", None))
    # retranslateUi
