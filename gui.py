# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(686, 442)
        self.train_btn = QtGui.QPushButton(Dialog)
        self.train_btn.setGeometry(QtCore.QRect(110, 40, 99, 27))
        self.train_btn.setObjectName(_fromUtf8("train_btn"))
        self.predict_btn = QtGui.QPushButton(Dialog)
        self.predict_btn.setGeometry(QtCore.QRect(450, 40, 131, 27))
        self.predict_btn.setObjectName(_fromUtf8("predict_btn"))
        self.result = QtGui.QLabel(Dialog)
        self.result.setGeometry(QtCore.QRect(90, 140, 531, 221))
        self.result.setObjectName(_fromUtf8("result"))
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.train_btn.setText(_translate("Dialog", "Train", None))
        self.predict_btn.setText(_translate("Dialog", "See Prediction", None))
        self.result.setText(_translate("Dialog", "TextLabel", None))

