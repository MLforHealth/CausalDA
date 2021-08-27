import numpy as np
import numpy.random as rand 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo

from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 

def simple_models(m_type,para,train,test):

    classifiers = {"svm":SVC(),
    "lda": LDA(),
    "rf": RandomForestClassifier(),
    "gradient_descent": AdaBoostClassifier(),
    "knn": KNeighborsClassifier(),
    "ada_boost": AdaBoostClassifier(),
    "gnb": GaussianNB(),
    "log": LogisticRegression()}

    parameters_search = {
        "svm": {'kernel':['rbf'], 'C':[10e-3,1,5,10], "gamma": [10e-4, 10e-3, 10e-2]},
        "lda": {'solver':['svd']},
        "rf": {'n_estimators': [10,20,10,100] , 'criterion': ['gini', 'entropy'], 'max_features':["auto", "log2"]},
        "knn" : {'n_neighors':[5,10,20]},
        "log1": {'penalty':['l1'], 'solver':['liblinear']},
        "log2": {'penalty':['l2'], 'solver':['liblinear']}, 
        "log_ele": {'penalty':['elasticnet'], 'solver':['saga'], 'l1_ratio':[0.2,0.4,0.6,0.8]}
        }

    X_train, Y_train = train 
    X_test, Y_test = test 

    N = X_train.shape[0]
    N1 = X_test.shape[0]

    model = classifiers[m_type]
    parameters = parameters_search[para]
    
    validation_fold = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):    
        validation_fold[i] = rand.randint(0,6)
    ps = PredefinedSplit(validation_fold)

    def unision_shuffles(a,b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def Acc_fun(predictions, labels):
        acc = (predictions == labels).sum()/float(len(labels))
        return acc 
    
    # shuffleing the inputs 
    X_train, Y_train = unision_shuffles(X_train, Y_train)
    X_test, Y_test = unision_shuffles(X_test, Y_test)

    clf = GridSearchCV(model, parameters, cv=PredefinedSplit(validation_fold), return_train_score = True)
    clf.fit(X_train, Y_train)
    train_predictions = clf.predict(X_train).reshape(-1,1)
    test_predictions = clf.predict(X_test).reshape(-1,1)
    Acc_train = Acc_fun(np.ravel(train_predictions),Y_train)
    Acc_test = Acc_fun(np.ravel(test_predictions), Y_test)

    return Acc_test



