# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'presetseditorwindow.ui'
##
# Created by: Qt User Interface Compiler version 5.15.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_PresetsEditor(object):
    def setupUi(self, PresetsEditor):
        if not PresetsEditor.objectName():
            PresetsEditor.setObjectName(u"PresetsEditor")
        PresetsEditor.resize(342, 400)
        PresetsEditor.setStyleSheet(u"/* Universal */\n"
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
                                    "\n"
                                    "QGroupBox,\n"
                                    "QLabel[titleText=\"true\"] {\n"
                                    "	font: 15pt \"Yu Gothic UI\";\n"
                                    "}\n"
                                    "/* Pushbutton */\n"
                                    "QPushButton {\n"
                                    "	background: none;\n"
                                    "	color: #000;\n"
                                    "}")
        self.verticalLayout = QVBoxLayout(PresetsEditor)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(30, 15, 30, 25)
        self.label = QLabel(PresetsEditor)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setProperty("titleText", True)

        self.verticalLayout.addWidget(self.label)

        self.listWidget_presets = QListWidget(PresetsEditor)
        self.listWidget_presets.setObjectName(u"listWidget_presets")
        self.listWidget_presets.setFrameShape(QFrame.StyledPanel)
        self.listWidget_presets.setEditTriggers(
            QAbstractItemView.DoubleClicked)
        self.listWidget_presets.setTabKeyNavigation(False)
        self.listWidget_presets.setProperty("showDropIndicator", True)
        self.listWidget_presets.setDefaultDropAction(Qt.TargetMoveAction)
        self.listWidget_presets.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.listWidget_presets.setMovement(QListView.Free)

        self.verticalLayout.addWidget(self.listWidget_presets)

        self.frame = QFrame(PresetsEditor)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.NoFrame)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.pushButton_add = QPushButton(self.frame)
        self.pushButton_add.setObjectName(u"pushButton_add")
        self.pushButton_add.setMinimumSize(QSize(35, 35))
        self.pushButton_add.setMaximumSize(QSize(35, 35))

        self.horizontalLayout.addWidget(self.pushButton_add)

        self.pushButton_delete = QPushButton(self.frame)
        self.pushButton_delete.setObjectName(u"pushButton_delete")
        self.pushButton_delete.setMinimumSize(QSize(35, 35))
        self.pushButton_delete.setMaximumSize(QSize(35, 35))

        self.horizontalLayout.addWidget(self.pushButton_delete)

        self.pushButton_import = QPushButton(self.frame)
        self.pushButton_import.setObjectName(u"pushButton_import")
        self.pushButton_import.setMinimumSize(QSize(35, 35))
        self.pushButton_import.setMaximumSize(QSize(35, 35))

        self.horizontalLayout.addWidget(self.pushButton_import)

        self.pushButton_export = QPushButton(self.frame)
        self.pushButton_export.setObjectName(u"pushButton_export")
        self.pushButton_export.setMinimumSize(QSize(35, 35))
        self.pushButton_export.setMaximumSize(QSize(35, 35))

        self.horizontalLayout.addWidget(self.pushButton_export)

        self.verticalLayout.addWidget(self.frame, 0, Qt.AlignHCenter)

        self.retranslateUi(PresetsEditor)

        QMetaObject.connectSlotsByName(PresetsEditor)
    # setupUi

    def retranslateUi(self, PresetsEditor):
        PresetsEditor.setWindowTitle(QCoreApplication.translate(
            "PresetsEditor", u"Presets Editor", None))
        self.label.setText(QCoreApplication.translate(
            "PresetsEditor", u"Seperation Presets", None))
        self.pushButton_add.setText(
            QCoreApplication.translate("PresetsEditor", u"+", None))
        self.pushButton_delete.setText(
            QCoreApplication.translate("PresetsEditor", u"-", None))
        self.pushButton_import.setText("")
        self.pushButton_export.setText("")
    # retranslateUi
