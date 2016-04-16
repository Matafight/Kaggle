import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

print("Fetching the training and test datasets")
train = pd.read_csv("train.csv",dtype = {"Age":np.float64},)
test = pd.read_csv("test.csv",dtype = {"Age":np.float64},)

print ("cleaning the dataset ...")

def cleaningData(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    #repalce nonnumeric item
    titanic.loc[titanic["Sex"] == "male","Sex"] = 1
    titanic.loc[titanic["Sex"] == "female","Sex"] =0
    titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    return titanic

print("Defining submission file ...")

def create_submission(rfc,train,test,predictors,filename):
    rfc.fit(train[predictors],train["Survived"])
    prediction = rfc.predict(test[predictors])
    submission_file = pd.DataFrame({
        "PassengerId" : test["PassengerId"],
        "Survived" : prediction
        })
    submission_file.to_csv(filename,index =False)#what does this index means?

print("Defining the clean dataset")
train_data = cleaningData(train)
test_data = cleaningData(test)

#with feature engineering
train_data["PSA"] = train_data["Pclass"]*train_data["Age"]*train_data["Sex"]
train_data["SP"] = train_data["SibSp"]+train_data["Parch"]

test_data["PSA"] = test_data["Pclass"]*test_data["Age"]*test_data["Sex"]
test_data["SP"] = test_data["SibSp"] + test_data["Parch"]

predictors = ["Pclass","Sex","Age","PSA","Fare","Embarked","SP","SibSp","Parch"]

print("Finding best n_esitmators for RandomForestClassifier ...")

#max_score = 0
#best_n = 0
#
#for n in range(1,300):
#    rfc_scr = 0
#    rfc = RandomForestClassifier(n_estimators = n) #n_estimator ??
#    for train,test in KFold(len(train_data),n_folds=10,shuffle = True):
#        rfc.fit(train_data[predictors].T[train].T,train_data["Survived"].T[train].T) #how to select columns in dataframe ?
#        rfc_scr += rfc.score(train_data[predictors].T[test].T,train_data["Survived"].T[test].T)/10#score function ?
#    if(rfc_scr > max_score):
#        print(n,rfc_scr)
#        max_score=rfc_scr
#        best_n=n
#
#print(best_n,max_score)
#
#print("Finding the best max_depth for RandomForestClassifier ...")
#
#    
#max_score = 0
#best_depth = 0
#
#for depth in range(1,100):
#    rfc_scr = 0
#    rfc = RandomForestClassifier(max_depth=depth)
#    for train,test in KFold(len(train_data),n_folds = 10,shuffle = True):
#        rfc.fit(train_data[predictors].T[train].T,train_data["Survived"].T[train].T)
#        rfc_scr += rfc.score(train_data[predictors].T[test].T,train_data["Survived"].T[test].T)/10
#    if(rfc_scr > max_score):
#        print(depth,rfc_scr)
#        max_score = rfc_scr
#        best_depth = depth
#
##print(best_depth,max_score)

max_score=0
best_n=0;
best_depth=0;
for n in range(1,100):
    for depth in range(1,10):
        rfc_scr=0
        
        rfc=RandomForestClassifier(n_estimators = n,max_depth = depth)
  
        for train,test in KFold(len(train_data),n_folds = 5,shuffle =True):
            rfc.fit(train_data[predictors].T[train].T,train_data["Survived"].T[train].T)
            rfc_scr += rfc.score(train_data[predictors].T[test].T,train_data["Survived"].T[test].T)/5
        if(rfc_scr > max_score):
            max_score = rfc_scr
            best_n=n
            best_depth=depth
            print(best_n,best_depth,max_score)
            
print(best_depth,best_n)
            

print("apply the model ...")

rfc = RandomForestClassifier(n_estimators = best_n,max_depth = best_depth)
create_submission(rfc,train_data,test_data,predictors,"rftitanic.csv")

print("submitted")
