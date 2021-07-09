# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'infowindow.ui'
##
# Created by: Qt User Interface Compiler version 5.15.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_InfoWindow(object):
    def setupUi(self, InfoWindow):
        if not InfoWindow.objectName():
            InfoWindow.setObjectName(u"InfoWindow")
        InfoWindow.resize(450, 357)
        self.verticalLayout = QVBoxLayout(InfoWindow)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 0, 0, 0)
        self.textEdit = QTextBrowser(InfoWindow)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setUndoRedoEnabled(False)
        self.textEdit.setReadOnly(True)
        self.textEdit.setHtml(u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                              "p, li { white-space: pre-wrap; }\n"
                              "</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
                              "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>")
        self.textEdit.setTextInteractionFlags(
            Qt.LinksAccessibleByMouse | Qt.TextSelectableByMouse)
        self.textEdit.setOpenExternalLinks(True)

        self.verticalLayout.addWidget(self.textEdit)

        self.retranslateUi(InfoWindow)

        QMetaObject.connectSlotsByName(InfoWindow)
    # setupUi

    def retranslateUi(self, InfoWindow):
        InfoWindow.setWindowTitle(
            QCoreApplication.translate("InfoWindow", u"Form", None))
    # retranslateUi
