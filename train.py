import os
from os import walk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn
import csv
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, precision_score,  recall_score, f1_score
from sklearn.externals import joblib
import pickle

features2 = []
testing_files = []

def runscript():
    print "started training"
    users = ['Neha','Harsha','vikas','sushant','shridhar','kamath','chir','bm','vaishali']

    path = "All"
    files = []
    folders =[]
    paths=[]

#getting full paths of directories, sub directories and files 
    for (dirpath, dirname, filenames) in walk(path):
        files.extend(filenames)
        folders.append(dirpath)
        for f in filenames:
            paths.append(str(dirpath+"/"+f))
        

#token features : bag of words representation
    vectorizer = CountVectorizer()
   

#folder feature
    m=len(folders)
    n=len(paths)
    folder_feature = [[0 for x in range(m)] for y in range(n)] 

    for x in range(0,m):
        for y in range(0,n):
            if folders[x] in paths[y]:
                folder_feature[y][x]=1
            else:
                folder_feature[y][x]=0

    



#user access
    m=len(users)

    access_feature  = [[0 for x in range(m)] for y in range(n)] 
    for u in range(0,m):
        for f in range(0,n):
            if ("CERAMICS ENGINEERING" in paths[f]) and (users[u]=="kamath" or users[u]=="shridhar"):
                access_feature[f][u]=1
            if ("DCS" in paths[f] or "CD" in paths[f] or "CG" in paths[f]) and (users[u] == "Neha" or users[u] == "chir"):
                access_feature[f][u]=1
            if ("ECS" in paths[f]) and (users[u] == "Harsha" or users[u] == "bm"):
                access_feature[f][u]=1
    n = len(paths)       
    for (dirpath, dirname, filenames) in walk("OTC_FIles"): 
        for name in users:
            if name in dirpath:
                for f in filenames:
                    for u in range(0,m):
                        for v in range(0,n):
                            if f in paths[v] and users[u] == name:
			        access_feature[v][u]=1 


    
    folder_feature= np.array(folder_feature)
    access_feature = np.array(access_feature)
    token_feature = vectorizer.fit(paths)
    access_feature_df = pd.DataFrame(data=access_feature,index=paths,columns=users)
#print access_feature_df.iloc[6]

    folder_feature_df = pd.DataFrame(data=folder_feature, index=paths, columns=folders)
#print folder_feature_df.iloc[1]

    token_feature_df = pd.DataFrame(data=token_feature, index=paths, columns=paths)

 
    final_df = pd.merge(access_feature_df,folder_feature_df,left_index=True,right_index=True,how='right')
    final_df = pd.merge(final_df, token_feature_df,left_index=True,right_index=True,how='right')




    train_df = pd.DataFrame(columns=['paths','type'])
    validation_df = pd.DataFrame(columns=['paths','type'])
    test_df = pd.DataFrame(columns=['paths','type'])
    i=0
    validi=0
    traini=0
    testi=0
    test_files = []
    validation_files = [] 
    while(i<len(paths)):
        if i%6==0:
            validation_df.set_value(validi,'paths',paths[i])
            validation_df.set_value(validi,'type',"validation")
            validation_files.append(paths[i])
            i+=1
            validi += 1
        elif i%400==0:
            test_df.set_value(testi,'paths',paths[i])
            test_df.set_value(testi,'type',"test")
            test_files.append(paths[i])
            i+=1
            testi += 1
        else:
            train_df.set_value(traini,'paths',paths[i])
            train_df.set_value(traini,'type',"train")
            i+=1
            traini += 1


    with open("test_files.txt","wb") as fp3:
        pickle.dump(test_files,fp3)


    df = pd.concat([train_df, validation_df,test_df])
#print df.iloc[0]
    final_df = pd.merge(final_df,df,left_index=True,right_on='paths',how='inner')
#print final_df.index.values


    unwanted=['type']
#print final_df.columns.values
    features=list(final_df.columns.values)
#print features
    features.remove('type')
    for x in users:
        features.remove(x)



    cat_cols= features
    for var in cat_cols:
        number = LabelEncoder()
        final_df[var] = number.fit_transform(final_df[var])



    train=final_df[final_df['type']=='train']
    validation=final_df[final_df['type']=='validation']
    test=final_df[final_df['type']=='test']
    
    test.to_csv("test_data.csv")


#print test_files
    metadata_feature=[]
    test_feature=[]
    precision=[]
    recall=[]
    f_score=[]
    i=-1
    features1 = features
    with open("features1.txt","wb") as fp:
        pickle.dump(features1,fp)

    for x in users:
        i+=1
        x_train = train[list(features)].values
        y_train = train[x].values
        x_validation=validation[list(features)].values
        y_validation=validation[x].values
        x_test = test[list(features)].values
        metamodel_clf = LinearSVC(random_state=0)
        metamodel_clf.fit(x_train, y_train)
        filename = x+"model.pkl"
        joblib.dump(metamodel_clf,filename) 
        testmodel = metamodel_clf.predict(x_test)
        #print testmodel
        metamodel = metamodel_clf.predict(x_validation)
        metadata_feature.append(metamodel)
        test_feature.append(testmodel)
        #f_score.append(f1_score(y_validation,metamodel,average='weighted'))
        #precision.append(precision_score(y_validation,metamodel,average='weighted'))
        #recall.append(recall_score(y_validation,metamodel,average='weighted'))
        #print 'Precision:{0:0.2f}'.format(precision[i])
        #print 'Recall:{0:0.2f}'.format(recall[i])
        #print 'F-score:{0:0.2f}'.format(f_score[i])


    predicted_users = ['P_Neha','P_Harsha','P_vikas','P_sushant','P_shridhar','P_kamath','P_chir','P_bm','P_vaishali']

    metadata_feature=np.array(metadata_feature)
    metadata_feature = np.ndarray.transpose(metadata_feature)
    temp=validation.index.values
    metadata_feature_df = pd.DataFrame(data=metadata_feature,index=temp,columns=predicted_users)
    final_df = pd.merge(final_df,metadata_feature_df,left_index=True, right_index=True,how='inner')
    validation=pd.merge(validation,metadata_feature_df,left_index=True, right_index=True,how='inner')
    test_feature=np.array(test_feature)
    test_feature = np.ndarray.transpose(test_feature)
    temp=test.index.values
    test_feature_df = pd.DataFrame(data=test_feature,index=temp,columns=predicted_users)
    test=pd.merge(test,test_feature_df,left_index=True, right_index=True,how='inner')
    test.to_csv("test_data2.csv")
    for x in predicted_users:
        features.append(x)
    features2 = features
    with open("features2.txt","wb") as fp2:
        pickle.dump(features2,fp2)
    for x in users:
   
        x_validation = validation[list(features)].values
        y_validation=validation[x].values
        clf = LinearSVC(random_state=0)
        clf.fit(x_validation, y_validation) 
        filename = x+"cfmodel.pkl"
        joblib.dump(clf,filename) 

    

    
   

    print "finished training"

def get_files():
    return testing_files 
    





