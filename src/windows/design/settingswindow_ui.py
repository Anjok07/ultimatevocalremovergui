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
        SettingsWindow.resize(830, 510)
        SettingsWindow.setStyleSheet(u"")
        self.horizontalLayout = QHBoxLayout(SettingsWindow)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_15 = QFrame(SettingsWindow)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setMinimumSize(QSize(170, 0))
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
        self.radioButton_separationSettings = QRadioButton(
            self.frame_settingsSelection)
        self.radioButton_separationSettings.setObjectName(
            u"radioButton_separationSettings")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.radioButton_separationSettings.sizePolicy().hasHeightForWidth())
        self.radioButton_separationSettings.setSizePolicy(sizePolicy1)
        self.radioButton_separationSettings.setMinimumSize(QSize(0, 40))
        self.radioButton_separationSettings.setCheckable(True)
        self.radioButton_separationSettings.setChecked(True)
        self.radioButton_separationSettings.setProperty("menu", True)

        self.verticalLayout.addWidget(self.radioButton_separationSettings)

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
        self.page_seperationSettings.setProperty("minimumFrameWidth", 770)
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
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 630, 510))
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
        self.verticalLayout_5.setContentsMargins(0, 25, 35, 0)
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
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
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
        self.checkBox_outputImage = QCheckBox(self.groupBox_conversion)
        self.checkBox_outputImage.setObjectName(u"checkBox_outputImage")

        self.gridLayout.addWidget(self.checkBox_outputImage, 3, 0, 1, 1)

        self.checkBox_modelFolder = QCheckBox(self.groupBox_conversion)
        self.checkBox_modelFolder.setObjectName(u"checkBox_modelFolder")

        self.gridLayout.addWidget(self.checkBox_modelFolder, 2, 0, 1, 1)

        self.checkBox_tta = QCheckBox(self.groupBox_conversion)
        self.checkBox_tta.setObjectName(u"checkBox_tta")

        self.gridLayout.addWidget(self.checkBox_tta, 1, 0, 1, 1)

        self.checkBox_postProcess = QCheckBox(self.groupBox_conversion)
        self.checkBox_postProcess.setObjectName(u"checkBox_postProcess")

        self.gridLayout.addWidget(self.checkBox_postProcess, 0, 1, 1, 1)

        self.checkBox_gpuConversion = QCheckBox(self.groupBox_conversion)
        self.checkBox_gpuConversion.setObjectName(u"checkBox_gpuConversion")

        self.gridLayout.addWidget(self.checkBox_gpuConversion, 0, 0, 1, 1)

        self.checkBox_deepExtraction = QCheckBox(self.groupBox_conversion)
        self.checkBox_deepExtraction.setObjectName(u"checkBox_deepExtraction")
        self.checkBox_deepExtraction.setEnabled(True)

        self.gridLayout.addWidget(self.checkBox_deepExtraction, 1, 1, 1, 1)

        self.frame_9 = QFrame(self.groupBox_conversion)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setFrameShape(QFrame.NoFrame)
        self.frame_9.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.doubleSpinBox_aggressiveness = QDoubleSpinBox(self.frame_9)
        self.doubleSpinBox_aggressiveness.setObjectName(
            u"doubleSpinBox_aggressiveness")
        self.doubleSpinBox_aggressiveness.setEnabled(True)
        self.doubleSpinBox_aggressiveness.setMinimumSize(QSize(50, 0))
        self.doubleSpinBox_aggressiveness.setMaximumSize(QSize(50, 16777215))
        self.doubleSpinBox_aggressiveness.setMinimum(-0.100000000000000)
        self.doubleSpinBox_aggressiveness.setMaximum(0.100000000000000)
        self.doubleSpinBox_aggressiveness.setSingleStep(0.010000000000000)
        self.doubleSpinBox_aggressiveness.setValue(0.020000000000000)

        self.horizontalLayout_3.addWidget(self.doubleSpinBox_aggressiveness)

        self.label_7 = QLabel(self.frame_9)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setEnabled(True)

        self.horizontalLayout_3.addWidget(self.label_7)

        self.gridLayout.addWidget(self.frame_9, 2, 1, 1, 1)

        self.frame_10 = QFrame(self.groupBox_conversion)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setFrameShape(QFrame.NoFrame)
        self.frame_10.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.comboBox_highEndProcess = QComboBox(self.frame_10)
        self.comboBox_highEndProcess.addItem("")
        self.comboBox_highEndProcess.addItem("")
        self.comboBox_highEndProcess.addItem("")
        self.comboBox_highEndProcess.addItem("")
        self.comboBox_highEndProcess.addItem("")
        self.comboBox_highEndProcess.setObjectName(u"comboBox_highEndProcess")

        self.horizontalLayout_9.addWidget(self.comboBox_highEndProcess)

        self.label_8 = QLabel(self.frame_10)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setEnabled(True)

        self.horizontalLayout_9.addWidget(self.label_8)

        self.gridLayout.addWidget(self.frame_10, 3, 1, 1, 1)

        self.horizontalLayout_4.addWidget(self.groupBox_conversion)

        self.frame_17 = QFrame(self.horizontalFrame_1)
        self.frame_17.setObjectName(u"frame_17")
        self.frame_17.setFrameShape(QFrame.NoFrame)
        self.frame_17.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_17)
        self.verticalLayout_9.setSpacing(10)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.groupBox = QGroupBox(self.frame_17)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFlat(True)
        self.horizontalLayout_10 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_10.setSpacing(10)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(20, 15, 15, 15)
        self.comboBox_presets = QComboBox(self.groupBox)
        self.comboBox_presets.setObjectName(u"comboBox_presets")
        self.comboBox_presets.setMinimumSize(QSize(0, 25))
        self.comboBox_presets.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLength)
        self.comboBox_presets.setModelColumn(0)

        self.horizontalLayout_10.addWidget(self.comboBox_presets)

        self.pushButton_presetsEdit = QPushButton(self.groupBox)
        self.pushButton_presetsEdit.setObjectName(u"pushButton_presetsEdit")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.pushButton_presetsEdit.sizePolicy().hasHeightForWidth())
        self.pushButton_presetsEdit.setSizePolicy(sizePolicy3)
        self.pushButton_presetsEdit.setMinimumSize(QSize(65, 27))
        self.pushButton_presetsEdit.setMaximumSize(QSize(16777215, 16777215))
        self.pushButton_presetsEdit.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_10.addWidget(self.pushButton_presetsEdit)

        self.horizontalLayout_10.setStretch(0, 1)

        self.verticalLayout_9.addWidget(self.groupBox)

        self.horizontalLayout_4.addWidget(self.frame_17)

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
        self.frame_instrumentalComboBox.setFrameShape(QFrame.NoFrame)
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
        self.comboBox_instrumental.setMaximumSize(QSize(16777215, 16777215))
        self.comboBox_instrumental.setEditable(False)
        self.comboBox_instrumental.setCurrentText(u"")
        self.comboBox_instrumental.setMaxVisibleItems(5)
        self.comboBox_instrumental.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.verticalLayout_3.addWidget(self.comboBox_instrumental)

        self.horizontalLayout_2.addWidget(self.frame_instrumentalComboBox)

        self.frame_constants = QFrame(self.groupBox_3)
        self.frame_constants.setObjectName(u"frame_constants")
        sizePolicy4 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.frame_constants.sizePolicy().hasHeightForWidth())
        self.frame_constants.setSizePolicy(sizePolicy4)
        self.frame_constants.setMinimumSize(QSize(0, 0))
        self.frame_constants.setMaximumSize(QSize(230, 140))
        self.frame_constants.setFrameShape(QFrame.NoFrame)
        self.frame_constants.setFrameShadow(QFrame.Raised)
        self.frame_constants.setLineWidth(0)
        self.gridLayout_5 = QGridLayout(self.frame_constants)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, 5, -1, 3)
        self.label_winSize = QLabel(self.frame_constants)
        self.label_winSize.setObjectName(u"label_winSize")
        self.label_winSize.setMinimumSize(QSize(80, 0))
        self.label_winSize.setMaximumSize(QSize(80, 16777215))
        self.label_winSize.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_winSize, 0, 1, 1, 1)

        self.comboBox_winSize = QComboBox(self.frame_constants)
        self.comboBox_winSize.addItem(u"352")
        self.comboBox_winSize.addItem(u"512")
        self.comboBox_winSize.addItem(u"1024")
        self.comboBox_winSize.setObjectName(u"comboBox_winSize")
        self.comboBox_winSize.setMinimumSize(QSize(0, 25))
        self.comboBox_winSize.setMaximumSize(QSize(60, 25))
        self.comboBox_winSize.setEditable(True)
        self.comboBox_winSize.setCurrentText(u"352")
        self.comboBox_winSize.setProperty("canEdit", True)

        self.gridLayout_5.addWidget(self.comboBox_winSize, 0, 0, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.horizontalLayout_2.addWidget(
            self.frame_constants, 0, Qt.AlignLeft)

        self.frame_stackComboBox = QFrame(self.groupBox_3)
        self.frame_stackComboBox.setObjectName(u"frame_stackComboBox")
        self.frame_stackComboBox.setFrameShape(QFrame.NoFrame)
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

        self.comboBox_vocal = QComboBox(self.frame_stackComboBox)
        self.comboBox_vocal.setObjectName(u"comboBox_vocal")
        self.comboBox_vocal.setEnabled(True)
        self.comboBox_vocal.setMinimumSize(QSize(0, 25))
        self.comboBox_vocal.setMaximumSize(QSize(16777215, 16777215))
        self.comboBox_vocal.setEditable(False)
        self.comboBox_vocal.setCurrentText(u"")
        self.comboBox_vocal.setMaxVisibleItems(5)
        self.comboBox_vocal.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.verticalLayout_4.addWidget(self.comboBox_vocal)

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
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 53, 35))
        self.scrollAreaWidgetContents.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                    "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                    "}")
        self.verticalLayout_14 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 25, 35, 0)
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
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 53, 35))
        self.scrollAreaWidgetContents_4.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                      "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                      "}")
        self.verticalLayout_15 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(0, 25, 35, 0)
        self.frame_2 = QFrame(self.scrollAreaWidgetContents_4)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.label_6 = QLabel(self.frame_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(300, 0, 201, 71))
        self.label_6.setProperty("title", True)
        self.horizontalLayoutWidget = QWidget(self.frame_2)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(260, 170, 160, 80))
        self.horizontalLayout_8 = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.radioButton_darkTheme = QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_darkTheme.setObjectName(u"radioButton_darkTheme")
        self.radioButton_darkTheme.setChecked(True)

        self.horizontalLayout_8.addWidget(self.radioButton_darkTheme)

        self.radioButton_lightTheme = QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_lightTheme.setObjectName(u"radioButton_lightTheme")

        self.horizontalLayout_8.addWidget(self.radioButton_lightTheme)

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
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 600, 342))
        self.scrollAreaWidgetContents_5.setMinimumSize(QSize(600, 0))
        self.scrollAreaWidgetContents_5.setStyleSheet(u"QFrame#frame_engine, QFrame#frame_modelOptions {\n"
                                                      "	background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.221409, y2:0.587, stop:0.119318 rgba(85, 78, 163, 255), stop:0.683616 rgba(0, 0, 0, 0));\n"
                                                      "}")
        self.verticalLayout_16 = QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(0, 25, 35, 0)
        self.frame_4 = QFrame(self.scrollAreaWidgetContents_5)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Raised)
        self.frame_4.setLineWidth(0)
        self.gridLayout_7 = QGridLayout(self.frame_4)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
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
        self.frame_11.setFrameShape(QFrame.NoFrame)
        self.frame_11.setFrameShadow(QFrame.Raised)
        self.frame_11.setLineWidth(0)
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
        sizePolicy5 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.pushButton_clearCommand.sizePolicy().hasHeightForWidth())
        self.pushButton_clearCommand.setSizePolicy(sizePolicy5)
        self.pushButton_clearCommand.setMinimumSize(QSize(50, 0))
        self.pushButton_clearCommand.setCursor(QCursor(Qt.PointingHandCursor))
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

        self.checkBox_notifiyOnFinish = QCheckBox(self.frame_8)
        self.checkBox_notifiyOnFinish.setObjectName(
            u"checkBox_notifiyOnFinish")

        self.gridLayout_4.addWidget(self.checkBox_notifiyOnFinish, 0, 0, 1, 2)

        self.checkBox_settingsStartup = QCheckBox(self.frame_8)
        self.checkBox_settingsStartup.setObjectName(
            u"checkBox_settingsStartup")

        self.gridLayout_4.addWidget(self.checkBox_settingsStartup, 2, 0, 1, 2)

        self.checkBox_multithreading = QCheckBox(self.frame_8)
        self.checkBox_multithreading.setObjectName(u"checkBox_multithreading")
        self.checkBox_multithreading.setEnabled(False)

        self.gridLayout_4.addWidget(self.checkBox_multithreading, 5, 0, 1, 2)

        self.verticalLayout_13.addWidget(self.frame_8)

        self.verticalLayout_20.addWidget(self.groupBox_6)

        self.frame_export = QFrame(self.frame_6)
        self.frame_export.setObjectName(u"frame_export")
        self.frame_export.setMinimumSize(QSize(0, 104))
        self.frame_export.setAcceptDrops(True)
        self.verticalLayout_17 = QVBoxLayout(self.frame_export)
        self.verticalLayout_17.setSpacing(10)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, -1, 10, -1)
        self.label_5 = QLabel(self.frame_export)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setProperty("title", True)

        self.horizontalLayout_7.addWidget(self.label_5)

        self.frame_7 = QFrame(self.frame_export)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setFrameShape(QFrame.HLine)
        self.frame_7.setFrameShadow(QFrame.Plain)

        self.horizontalLayout_7.addWidget(self.frame_7)

        self.pushButton = QPushButton(self.frame_export)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMaximumSize(QSize(30, 30))
        self.pushButton.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_7.addWidget(self.pushButton)

        self.horizontalLayout_7.setStretch(1, 1)

        self.verticalLayout_17.addLayout(self.horizontalLayout_7)

        self.gridLayout_10 = QGridLayout()
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.gridLayout_10.setHorizontalSpacing(10)
        self.gridLayout_10.setContentsMargins(35, 10, 30, 10)
        self.pushButton_exportDirectory = QPushButton(self.frame_export)
        self.pushButton_exportDirectory.setObjectName(
            u"pushButton_exportDirectory")
        self.pushButton_exportDirectory.setMinimumSize(QSize(18, 18))
        self.pushButton_exportDirectory.setMaximumSize(QSize(18, 18))
        self.pushButton_exportDirectory.setCursor(
            QCursor(Qt.PointingHandCursor))
        self.pushButton_exportDirectory.setFlat(True)
        self.pushButton_exportDirectory.setProperty("export", True)

        self.gridLayout_10.addWidget(
            self.pushButton_exportDirectory, 0, 1, 1, 1)

        self.label_3 = QLabel(self.frame_export)
        self.label_3.setObjectName(u"label_3")
        sizePolicy4.setHeightForWidth(
            self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy4)
        self.label_3.setMinimumSize(QSize(110, 0))
        self.label_3.setAlignment(
            Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)

        self.gridLayout_10.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_autoSave = QLabel(self.frame_export)
        self.label_autoSave.setObjectName(u"label_autoSave")
        self.label_autoSave.setAlignment(
            Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)

        self.gridLayout_10.addWidget(self.label_autoSave, 1, 0, 1, 2)

        self.checkBox_autoSaveInstrumentals = QCheckBox(self.frame_export)
        self.checkBox_autoSaveInstrumentals.setObjectName(
            u"checkBox_autoSaveInstrumentals")

        self.gridLayout_10.addWidget(
            self.checkBox_autoSaveInstrumentals, 1, 2, 1, 1)

        self.checkBox_autoSaveVocals = QCheckBox(self.frame_export)
        self.checkBox_autoSaveVocals.setObjectName(u"checkBox_autoSaveVocals")

        self.gridLayout_10.addWidget(self.checkBox_autoSaveVocals, 1, 3, 1, 1)

        self.label_exportDirectory = QLabel(self.frame_export)
        self.label_exportDirectory.setObjectName(u"label_exportDirectory")
        sizePolicy2.setHeightForWidth(
            self.label_exportDirectory.sizePolicy().hasHeightForWidth())
        self.label_exportDirectory.setSizePolicy(sizePolicy2)
        self.label_exportDirectory.setLineWidth(0)
        self.label_exportDirectory.setText(u"B:/Downloads")
        self.label_exportDirectory.setTextFormat(Qt.AutoText)
        self.label_exportDirectory.setScaledContents(True)
        self.label_exportDirectory.setAlignment(
            Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.label_exportDirectory.setWordWrap(True)
        self.label_exportDirectory.setIndent(1)
        self.label_exportDirectory.setProperty("path", True)

        self.gridLayout_10.addWidget(self.label_exportDirectory, 0, 2, 1, 2)

        self.gridLayout_10.setRowStretch(0, 1)
        self.gridLayout_10.setRowStretch(1, 1)

        self.verticalLayout_17.addLayout(self.gridLayout_10)

        self.verticalLayout_20.addWidget(self.frame_export)

        self.horizontalLayout_6.addWidget(self.frame_6)

        self.groupBox_4 = QGroupBox(self.frame_5)
        self.groupBox_4.setObjectName(u"groupBox_4")
        sizePolicy4.setHeightForWidth(
            self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy4)
        self.groupBox_4.setMinimumSize(QSize(0, 0))
        self.groupBox_4.setFlat(True)
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(35, 10, 30, 10)
        self.frame_languages = QFrame(self.groupBox_4)
        self.frame_languages.setObjectName(u"frame_languages")
        self.frame_languages.setFrameShape(QFrame.NoFrame)
        self.frame_languages.setFrameShadow(QFrame.Raised)
        self.frame_languages.setLineWidth(0)
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

        self.gridLayout_7.addWidget(self.frame_5, 0, 0, 1, 1)

        self.verticalLayout_16.addWidget(self.frame_4, 0, Qt.AlignTop)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_5)

        self.verticalLayout_7.addWidget(self.scrollArea_4)

        self.stackedWidget.addWidget(self.page_preferences)

        self.verticalLayout_18.addWidget(self.stackedWidget)

        self.horizontalLayout.addWidget(self.frame_14)

        self.retranslateUi(SettingsWindow)

        self.stackedWidget.setCurrentIndex(0)
        self.comboBox_highEndProcess.setCurrentIndex(4)
        self.comboBox_winSize.setCurrentIndex(0)
        self.pushButton_english.setDefault(False)
        self.pushButton_german.setDefault(False)

        QMetaObject.connectSlotsByName(SettingsWindow)
    # setupUi

    def retranslateUi(self, SettingsWindow):
        SettingsWindow.setWindowTitle(QCoreApplication.translate(
            "SettingsWindow", u"Settings", None))
        self.radioButton_separationSettings.setText(
            QCoreApplication.translate("SettingsWindow", u"Separation Settings", None))
        self.radioButton_shortcuts.setText(
            QCoreApplication.translate("SettingsWindow", u"Shortcuts", None))
        self.radioButton_customization.setText(
            QCoreApplication.translate("SettingsWindow", u"Customization", None))
        self.radioButton_preferences.setText(
            QCoreApplication.translate("SettingsWindow", u"Preferences", None))
        self.groupBox_conversion.setTitle(
            QCoreApplication.translate("SettingsWindow", u"Conversion ", None))
# if QT_CONFIG(tooltip)
        self.checkBox_outputImage.setToolTip(QCoreApplication.translate(
            "SettingsWindow", u"Save spectogram of seperated music files", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_outputImage.setText(QCoreApplication.translate(
            "SettingsWindow", u"Output Image", None))
        self.checkBox_modelFolder.setText(QCoreApplication.translate(
            "SettingsWindow", u"Model Test Mode", None))
        self.checkBox_tta.setText(
            QCoreApplication.translate("SettingsWindow", u"TTA", None))
        self.checkBox_postProcess.setText(QCoreApplication.translate(
            "SettingsWindow", u"Post-Process", None))
        self.checkBox_gpuConversion.setText(QCoreApplication.translate(
            "SettingsWindow", u"GPU Conversion", None))
        self.checkBox_deepExtraction.setText(QCoreApplication.translate(
            "SettingsWindow", u"Deep extraction", None))
        self.label_7.setText(QCoreApplication.translate(
            "SettingsWindow", u"Aggressiveness", None))
        self.comboBox_highEndProcess.setItemText(
            0, QCoreApplication.translate("SettingsWindow", u"None", None))
        self.comboBox_highEndProcess.setItemText(
            1, QCoreApplication.translate("SettingsWindow", u"Bypass", None))
        self.comboBox_highEndProcess.setItemText(
            2, QCoreApplication.translate("SettingsWindow", u"Correlation", None))
        self.comboBox_highEndProcess.setItemText(
            3, QCoreApplication.translate("SettingsWindow", u"Mirroring", None))
        self.comboBox_highEndProcess.setItemText(
            4, QCoreApplication.translate("SettingsWindow", u"Mirroring 2", None))

        self.label_8.setText(QCoreApplication.translate(
            "SettingsWindow", u"High End Processing", None))
        self.groupBox.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Presets ", None))
        self.pushButton_presetsEdit.setText(
            QCoreApplication.translate("SettingsWindow", u"Edit", None))
        self.groupBox_3.setTitle(QCoreApplication.translate(
            "SettingsWindow", u"Models ", None))
        self.label_2.setText(QCoreApplication.translate(
            "SettingsWindow", u"Instrumental Model", None))
        self.label_winSize.setText(QCoreApplication.translate(
            "SettingsWindow", u"Window Size", None))

        self.label_4.setText(QCoreApplication.translate(
            "SettingsWindow", u"Vocal Model", None))
        self.label_6.setText(QCoreApplication.translate(
            "SettingsWindow", u"Themes", None))
        self.radioButton_darkTheme.setText(
            QCoreApplication.translate("SettingsWindow", u"Dark", None))
        self.radioButton_lightTheme.setText(
            QCoreApplication.translate("SettingsWindow", u"Light", None))
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
        self.checkBox_notifiyOnFinish.setText(QCoreApplication.translate(
            "SettingsWindow", u"Notify me on finish of separation", None))
        self.checkBox_settingsStartup.setText(QCoreApplication.translate(
            "SettingsWindow", u"Open Settings on application startup", None))
# if QT_CONFIG(tooltip)
        self.checkBox_multithreading.setToolTip(QCoreApplication.translate(
            "SettingsWindow", u"Process multiple files simultaneously", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_multithreading.setText(QCoreApplication.translate(
            "SettingsWindow", u"Multithreading (experimental)", None))
        self.label_5.setText(QCoreApplication.translate(
            "SettingsWindow", u"Export Settings", None))
        self.pushButton.setText(
            QCoreApplication.translate("SettingsWindow", u"i", None))
        self.pushButton_exportDirectory.setText("")
        self.label_3.setText(QCoreApplication.translate(
            "SettingsWindow", u"Export Directory:", None))
        self.label_autoSave.setText(QCoreApplication.translate(
            "SettingsWindow", u"Automatically Save:", None))
        self.checkBox_autoSaveInstrumentals.setText(
            QCoreApplication.translate("SettingsWindow", u"Instrumental", None))
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
    # retranslateUi
