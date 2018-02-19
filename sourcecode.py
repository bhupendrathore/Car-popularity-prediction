#gs_sprint
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt
#read train &test files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv',header =None)
test.columns = ['buying_price', 'maintainence_cost', 'number_of_doors',
       'number_of_seats', 'luggage_boot_size', 'safety_rating']
#data preprocessing
#for training set
buying_onehot = pd.get_dummies(train.buying_price)
buying_onehot.columns = ['f01','f02','f03','f04']
train = pd.concat([train,buying_onehot],axis  = 1)
maintainence_onehot = pd.get_dummies(train.maintainence_cost)
maintainence_onehot.columns =['f05','f06','f07','f08']
train = pd.concat([train,maintainence_onehot],axis = 1)
luggage_onehot = pd.get_dummies(train.luggage_boot_size)
luggage_onehot.columns =['f09','f10','f11']
train = pd.concat([train,luggage_onehot],axis = 1)
safety_onehot = pd.get_dummies(train.safety_rating)
safety_onehot.columns =['f12','f13','f14']
train = pd.concat([train,safety_onehot],axis = 1)

#for test set
buying_onehot = pd.get_dummies(test.buying_price)
buying_onehot.columns = ['f01','f02','f03','f04']
test = pd.concat([test,buying_onehot],axis  = 1)
maintainence_onehot = pd.get_dummies(test.maintainence_cost)
maintainence_onehot.columns =['f05','f06','f07','f08']
test = pd.concat([test,maintainence_onehot],axis = 1)
luggage_onehot = pd.get_dummies(test.luggage_boot_size)
luggage_onehot.columns =['f09','f10','f11']
test = pd.concat([test,luggage_onehot],axis = 1)
safety_onehot = pd.get_dummies(test.safety_rating)
safety_onehot.columns =['f12','f13','f14']
test = pd.concat([test,safety_onehot],axis = 1)

#remove_extra
train =train.drop(['buying_price','maintainence_cost','luggage_boot_size','safety_rating'],axis=1)
test =test.drop(['buying_price','maintainence_cost','luggage_boot_size','safety_rating'],axis=1)
del buying_onehot , maintainence_onehot,luggage_onehot,safety_onehot

# split for cv
X_train,X_test,y_train,y_test = train_test_split(train.drop(['popularity'],axis=1),train.popularity,
                                                 test_size=0.33, random_state=42)

#fit xgboost
xg_train = xgb.DMatrix(X_train,label =y_train)
xg_test = xgb.DMatrix(X_test,label=y_test)
param ={}

param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = 0
param['nthread'] = 4


param['num_class'] = 5
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 1720
#bst = xgb.train(param, xg_train, num_round, watchlist)
bst2 = xgb.train(param, xgb.DMatrix(train.drop(['popularity'],axis=1),label = train.popularity), num_round)
# get prediction
pred = bst2.predict(xg_test)
error_rate = np.sum(pred != y_test) / y_test.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# train MLPClassifier

model = MLPClassifier(activation='relu', alpha=0.0001, batch_size=32, beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(20000,), learning_rate='invscaling',
       learning_rate_init=0.001, max_iter=20000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
model.fit(train.drop(['popularity'],axis=1),train.popularity)

predprobaNN = model.predict_proba(test)
predproba = bst2.predict(xgb.DMatrix(test))
predprobaNN = pd.DataFrame(predprobaNN)
yt = pd.DataFrame(yt)
yt.columns = [1,2,3,4]
predprobaNN.columns = [1,2,3,4]
predproba = pd.DataFrame(predproba).drop([0],axis = 1)

tt = predprobaNN.add(predproba,axis ='columns')
y = tt.idxmax(axis=1)
y.to_csv('ensemble.csv',index=False)



