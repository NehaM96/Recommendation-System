import os
from os import walk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score, precision_score,  recall_score, f1_score
from sklearn.externals import joblib
import pickle
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
from gui import Ui_Dialog
import train

users = ['Neha','Harsha','vikas','sushant','shridhar','kamath','chir','bm','vaishali']


features2 = []
x_metatest = pd.DataFrame()
x_cftest = pd.DataFrame()
metamodel={}
cfmodel={}
meta_prediction=[]
cf_prediction=[]

model=[]
mmodel=[]
class MyDialog(QDialog):
  
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.train_btn.clicked.connect(self.Train)
        #self.ui.BrowseFile_btn.clicked.connect(self.selectFile)
        self.ui.predict_btn.clicked.connect(self.Test)
    
    def Train(self):
        train.runscript()
        
        
    
    def Test(self):
        result_string = ""
        mresult_string=""
        with open("test_files.txt","r") as fp3:
            test_files = joblib.load(fp3)
        print "metadata model testing"
        for x in users:
            filename = x+"model.pkl"
            metamodel[x] = joblib.load(filename)
        test_data = pd.read_csv('test_data.csv')
        with open("features1.txt","r") as fp:
            features1 = pickle.load(fp)
        x_metatest = test_data[list(features1)].values
        
        for x in users:
            print x
            meta_prediction.append(metamodel[x].predict(x_metatest))
            print metamodel[x].predict(x_metatest)
            mmodel.append(metamodel[x].predict(x_metatest))
        for y in range(len(test_files)):
            mresult="File "+ test_files[y]+" is recommended to "
            for z in range(len(users)):
                if mmodel[z][y]==1:
                    mresult+=users[z]
                    mresult+=", "
            mresult_string+=mresult
            mresult_string+="\n" 
        #cf_prediction.append(cfmodel[x].predict(x_cftest))
        #print cf_prediction
        #self.ui.result.setText(mresult_string)
        print mresult_string

        print "collaboration model testing"
        with open("features2.txt","r") as fp:
            features2 = pickle.load(fp)
        
        for x in users:
            
            filename = x+"cfmodel.pkl"
            cfmodel[x] = joblib.load(filename)
        test_data2 = pd.read_csv('test_data2.csv')
        
        
        x_cftest = test_data2[list(features2)].values
        for x in users:
            
            print x
            print cfmodel[x].predict(x_cftest)
            model.append(cfmodel[x].predict(x_cftest))
           
        for y in range(len(test_files)):
            result="File "+ test_files[y]+" is recommended to "
            for z in range(len(users)):
                if model[z][y]==1:
                    result+=users[z]
                    result+=", "
            result_string+=result
            result_string+="\n" 
        #cf_prediction.append(cfmodel[x].predict(x_cftest))
        #print cf_prediction
        self.ui.result.setText(result_string)
        print result_string
        print "finished testing"
 
if __name__ == "__main__":
        app = QApplication(sys.argv)
        myapp = MyDialog()
        myapp.show()
        
        
        sys.exit(app.exec_())


     



