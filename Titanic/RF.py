import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv",dtype = {"Age":np.float64},)
test = pd.read_csv("test.csv",dtype = {"Age":np.float64},)

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


def create_submission(rfc,train,test,predictors,filename):
    rfc.fit(train[predictors],train["Survived"])
    prediction = rfc.predict(test[predictors])
    submission_file = pd.DataFrame({
        "PassengerId" : test["PassengerId"],
        "Survived" : prediction
        })
    submission_file.to_csv(filename,index =False)#what does this index means?


train_data = cleaningData(train)
test_data = cleaningData(test)

#with feature engineering
train_data["PSA"] = train_data["Pclass"]*train_data["Age"]*train_data["Sex"]
train_data["SP"] = train_data["SibSp"]+train_data["Parch"]

test_data["PSA"] = test_data["Pclass"]*test_data["Age"]*test_data["Sex"]
test_data["SP"] = test_data["SibSp"] + test_data["Parch"]

predictors = ["Pclass","Sex","Age","PSA","Fare","Embarked","SP","SibSp","Parch"]

print("Finding best n_esitmators for RandomForestClassifier ...")


max_score=0
best_n=0;
best_depth=0;
results=[]
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
            print("n_estimators",best_n,"max_depth",best_depth,"acc",max_score)
            results.append(max_score)
            
print(best_depth,best_n)
          
plt.plot(results,'r--')
plt.ylabel("accuracy")

print("apply the model ...")

rfc = RandomForestClassifier(n_estimators = best_n,max_depth = best_depth)
create_submission(rfc,train_data,test_data,predictors,"rftitanic.csv")

print("submitted")
