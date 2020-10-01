# -*- coding: utf-8 -*-
"""Asg_3_InsuranceFraud_AR.ipynb
"""

from vecstack import stacking
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score #works
from sklearn.model_selection import train_test_split # we are not using this in this program
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.svm import LinearSVC
from collections import Counter #for Smote, 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
import warnings
warnings.filterwarnings("ignore")
import datetime

print(datetime.datetime.now())

trainfile = r'C:\Users\asus\OneDrive - Arizona State University\S_Datamining1\S_Assignment3\RevisedHomesiteTrain.csv'
train_data = pd.read_csv(trainfile)

testfile = r'C:\Users\asus\OneDrive - Arizona State University\S_Datamining1\S_Assignment3\RevisedHomesiteTest.csv'
test_data = pd.read_csv(testfile)

print(train_data.shape)
print(train_data.head()) 

print(test_data.shape)
print(test_data.head())

#Copy Train data excluding target
trainData_Copy = train_data.iloc[:, :-1].copy()
testData_Copy = test_data.iloc[:, :-1].copy()

#Combine Train and test for one Hot Encoding
combined_Data = pd.concat([trainData_Copy,testData_Copy], keys=[0,1])

#Separate Train data and test data
X_train = combined_Data.xs(0)
X_test = combined_Data.xs(1)
y_train=train_data["QuoteConversion_Flag"]

#Select just Target Column
y_train = train_data.iloc[:, -1]

print(X_train.shape)
print(X_test.head()) 
print(y_train.shape)

#Default DecisionTree
print("CONSTRUCT DEFAULT DECISION TREE AND OBTAIN RESPECTIVE ACCURACY ==================")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred_DTDefault=clf.predict(X_test)
clf_pred_DTDefault=pd.DataFrame(pred_DTDefault,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_DTDefault],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_DTDefault.csv", index = None)


#Hyperparameter tuning done for decision tree classifier
print("Hyperparameter tuning done for decision tree classifier")
print("Using subsets of data build using the training data")
parameters={'min_samples_split' : range(10,100,10),'max_depth': range(1,12,1)}
clf_random = RandomizedSearchCV(clf,parameters,n_iter=25)
clf_random.fit(X_train, y_train)
grid_parm=clf_random.best_params_
print(grid_parm)

print("Using the parameters obtained from HyperParameterTuning in the DecisionTreeClassifier ")
clf = DecisionTreeClassifier(**grid_parm)
clf.fit(X_train,y_train)
#clf_predict = clf.predict(X_test)
#print("HyperParameterTuned DT: ",clf_predict)
pred_DTHyper=clf.predict(X_test)
clf_pred_DTHyper=pd.DataFrame(pred_DTHyper,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_DTHyper],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_DTHyper.csv", index = None)

#run cross-validation on best hyperparameters, get auc score
print("#run cross-validation on best hyperparameters, get auc score")
clf_cv_score = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc")
print("=== All AUC Scores ===")
print(clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ",clf_cv_score.mean())

#Random Forest =============================================================
#Default mode 
print(""" #Random Forest =============================================================
      Default mode""")
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_predict_Default=rfc.predict(X_test)
clf_pred_rfcDefault=pd.DataFrame(rfc_predict_Default,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_rfcDefault],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_RFC_Default.csv", index = None)

#Hyperparameter tuning for random forest
print("#Hyperparameter tuning for random forest")
#parameters={ 'n_estimators': range(50,150,20),'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
parameters={ 'n_estimators': range(50,150,20),'min_samples_split' : range(10,100,10),'max_depth': range(1,40,2)}
rfc_random = RandomizedSearchCV(rfc,parameters,n_iter=15)
rfc_random.fit(X_train, y_train)
grid_parm_rfc=rfc_random.best_params_
print(grid_parm_rfc)

#contruct random forest using the best parameters
rfc= RandomForestClassifier(**grid_parm_rfc)
rfc.fit(X_train,y_train)
rfc_predict_HyperBest = rfc.predict(X_test)
clf_pred_rfcHyperBest=pd.DataFrame(rfc_predict_HyperBest,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_rfcHyperBest],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_RFC_HyperBest.csv", index = None)
#run cross-validation on best parameters, get auc score
print("#run cross-validation on best parameters, get auc score")
rfc_cv_score = cross_val_score(rfc, X_train, y_train, cv=5, scoring="roc_auc")
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ",rfc_cv_score.mean())



#Gradient Boosting ============================================================
print("#Gradient Boosting ============================================================")
search_grid={'n_estimators':[5,10,20, 30, 50],'learning_rate':[0.01,.1]} # N_estimators is the number of trees. Learning_rate = ??
abc =GradientBoostingClassifier()
abc.fit(X_train, y_train)
abc_predict=abc.predict(X_test)
print("Gradient Boosting Predictions :",abc_predict)
gradboost_pred_Default = abc.predict(X_test)
clf_pred_GradBoostDefault=pd.DataFrame(gradboost_pred_Default,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_GradBoostDefault],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_GradBoost_Default.csv", index = None)


#Randomized Search for hyperparameter tuning
print("#Randomized Search for hyperparameter tuning")
abc_random = RandomizedSearchCV(abc,search_grid,n_iter=10)
abc_random.fit(X_train, y_train)
grid_parm_abc=abc_random.best_params_
print(grid_parm_abc)

#Construct Gradient Boosting Trees using the best parameters
print("#Construct Gradient Boosting Trees using the best parameters")
abc= GradientBoostingClassifier(**grid_parm_abc)
abc.fit(X_train,y_train)
gradboost_pred_Hyper = abc.predict(X_test)
#print("Gradient Boosting predictions with Best Parameters :",abc_predict)
clf_pred_GradBoostHyper=pd.DataFrame(gradboost_pred_Hyper,columns=['QuoteConversion_Flag'])
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_GradBoostHyper],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_GradBoost_Hyper.csv", index = None)

#run cross-validation on best parameters, get auc score
print("#run cross-validation on best parameters, get auc score")
abc_cv_score = cross_val_score(abc, X_train, y_train, cv=5, scoring="roc_auc")
print("=== All AUC Scores ===")
print(abc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Boosting: ",abc_cv_score.mean())



#MLP DEFAULT===========================================================
print("#MLP============================================================")
ptc = Perceptron()
ptc.fit(X_train,y_train)
pred_MLP = ptc.predict(X_test)
ID = X_test['QuoteNumber']
clf_pred_MLP=pd.DataFrame(pred_MLP,columns=['QuoteConversion_Flag'])
print("MLP Predictions:",clf_pred_MLP)
#Writing the predictions to a file
pd.concat([ID,clf_pred_MLP],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_MLP_DEFAULT.csv", index = None)

#MLP HYPER==========================================================
print("#MLP============================================================")
ptc = Perceptron(penalty='elasticnet',n_jobs =-1)
ptc.fit(X_train,y_train)
pred_MLP = ptc.predict(X_test)
ID = X_test['QuoteNumber']
clf_pred_MLP=pd.DataFrame(pred_MLP,columns=['QuoteConversion_Flag'])
print("MLP Predictions:",clf_pred_MLP)
#Writing the predictions to a file
pd.concat([ID,clf_pred_MLP],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_MLP_HYPER.csv", index = None)


#LinearSVM DEFAULT============================================================
print("#SVM============================================================")
svm = LinearSVC()
svm.fit(X_train,y_train)
pred_SVM = svm.predict(X_test)
ID = X_test['QuoteNumber']
clf_pred_SVM=pd.DataFrame(pred_SVM,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([X_test['QuoteNumber'],clf_pred_SVM],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_SVM_DEFAULT.csv", index = None)


#LinearSVM============================================================
print("#SVM============================================================")
svm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
svm.fit(X_train,y_train)
pred_SVM = svm.predict(X_test)
ID = X_test['QuoteNumber']
clf_pred_SVM=pd.DataFrame(pred_SVM,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([X_test['QuoteNumber'],clf_pred_SVM],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_SVM_Hyper.csv", index = None)

#KNN DEFAULT============================================================
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_KNN = knn.predict(X_test)
print(pred_KNN)
ID = X_test['QuoteNumber']
clf_pred_KNN=pd.DataFrame(pred_KNN,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([ID,clf_pred_KNN],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_KNN_DEFAULT.csv", index = None)

#KNN HYPER============================================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred_KNN = knn.predict(X_test)
print(pred_KNN)
ID = X_test['QuoteNumber']
clf_pred_KNN=pd.DataFrame(pred_KNN,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([ID,clf_pred_KNN],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\WithoutStackingresults_KNN_HYPER.csv", index = None)




print("SMOTE==============================================================================")
print("___________________________________________________________________\nSMOTE\n")
print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(sampling_strategy='float', ratio=0.35) # default is 0.5
X_res, y_res = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

print("#STACKING MODELS =====================================================================")
#STACKING MODELS =====================================================================
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using GradientBoosting, RandomForest and Decision Tree Classifier\n")

models = [ GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier() ] # we can specify parameters as well in this.
      
S_Train, S_Test = stacking(models,                          # S_Train is the new training set as part of the stacking. But the target value will be the same as original
                           X_res, y_res, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=4,               # number of levels/repetitions we want in the stacking
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)

#STACKING - CONTRUCT A DEFAULT GRADIENT BOOSTING MODEL==============================
model = GradientBoostingClassifier()
model = model.fit(S_Train, y_res)
y_pred_gradient = model.predict(S_Test)
clf_pred_gradient=pd.DataFrame(y_pred_gradient,columns=['QuoteConversion_Flag'])
print("Stacking predictions based on Gradient Boosting:",y_pred_gradient)
#Get Prediction Probability for the predicted class as a dataframe
pred_Probability =pd.DataFrame(model.predict_proba(S_Test))
print("GradientBoosting Predictions:",pred_Probability)
#Writing the predictions to a file
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_gradient],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_GradientBoosting.csv", index = None)

#LinearSVM DEFAULT============================================================
print("Linear SVM on Stacking ===========================================================")
svm = LinearSVC()
svm.fit(S_Train,y_res)
pred_SVM = svm.predict(S_Test)
print("Stacking predictions Based on Linear SVM :",pred_SVM)
ID = X_test['QuoteNumber']
clf_pred=pd.DataFrame(pred_SVM,columns=['QuoteConversion_Flag'])

#Writing the predictions to a file
pd.concat([ID,clf_pred],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_LinearSVM.csv", index = None)


#MLP DEFAULT============================================================
print("#MLP============================================================")
ptc = Perceptron()
ptc.fit(S_Train,y_res)
pred_MLP_stack = ptc.predict(S_Test)

#==============USING MLP TO PREDICT AND TO WRITE RESULTS==============
ID = X_test['QuoteNumber']
clf_pred_Stack_MLP=pd.DataFrame(pred_MLP_stack,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([ID,clf_pred_Stack_MLP],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_MLP.csv", index = None)
#for checking the execution time
#print(datetime.datetime.now())


#KNN DEFAULT============================================================
knn = KNeighborsClassifier()
knn.fit(S_Train,y_res)
pred_KNN_Stack = knn.predict(S_Test)
#------USING SVM TO PREDICT AND WRITE A DOCUMENT-------------
ID = X_test['QuoteNumber']
clf_pred_KNN_Stack=pd.DataFrame(pred_KNN_Stack,columns=['QuoteConversion_Flag'])
pd.concat([ID,clf_pred_KNN_Stack],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_KNN.csv", index = None)

#==============================================================================================================================
#=================================Stacking classifiers with Hyper Parameter Tuning=============================================
#==============================================================================================================================


#STACKING - CONTRUCT A HYPER PARAMETER TUNED GRADIENT BOOSTING MODEL==============================
model = GradientBoostingClassifier(min_samples_split=5,min_samples_leaf=4,min_weight_fraction_leaf=0.0,max_depth=4)
model = model.fit(S_Train, y_res)
y_pred_gradient = model.predict(S_Test)
clf_pred_gradient_hyper=pd.DataFrame(y_pred_gradient,columns=['QuoteConversion_Flag'])
print("Stacking predictions based on Gradient Boosting:",y_pred_gradient)
#Get Prediction Probability for the predicted class as a dataframe
pred_Probability =pd.DataFrame(model.predict_proba(S_Test))
print("GradientBoosting Predictions:",pred_Probability)
#Writing the predictions to a file
ID = X_test['QuoteNumber']
pd.concat([ID,clf_pred_gradient_hyper],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_GradientBoosting_Hyper.csv", index = None)

#LinearSVM HYPER============================================================
print("Linear SVM on Stacking ===========================================================")
svm = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
svm.fit(S_Train,y_res)
pred_SVM = svm.predict(S_Test)
print("Stacking predictions Based on Linear SVM :",pred_SVM)
ID = X_test['QuoteNumber']
clf_pred_svm_hyper=pd.DataFrame(pred_SVM,columns=['QuoteConversion_Flag'])

#Writing the predictions to a file
pd.concat([ID,clf_pred_svm_hyper],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_LinearSVM_Hyper.csv", index = None)


#MLP HYPER============================================================
print("#MLP============================================================")
ptc = Perceptron(penalty='elasticnet',n_jobs =-1)
ptc.fit(S_Train,y_res)
pred_MLP_stack = ptc.predict(S_Test)

#==============USING MLP TO PREDICT AND TO WRITE RESULTS==============
ID = X_test['QuoteNumber']
clf_pred_Stack_MLP=pd.DataFrame(pred_MLP_stack,columns=['QuoteConversion_Flag'])
#Writing the predictions to a file
pd.concat([ID,clf_pred_Stack_MLP],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_MLP_HYPER.csv", index = None)


#KNN HYPER============================================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(S_Train,y_res)
pred_KNN_Stack = knn.predict(S_Test)
#------USING SVM TO PREDICT AND WRITE A DOCUMENT-------------
ID = X_test['QuoteNumber']
clf_pred_KNN_Stack=pd.DataFrame(pred_KNN_Stack,columns=['QuoteConversion_Flag'])
pd.concat([ID,clf_pred_KNN_Stack],axis=1).to_csv("C:\\Users\\asus\\OneDrive - Arizona State University\\S_Datamining1\\S_Assignment3\\Stackingresults_KNN_HYPER.csv", index = None)