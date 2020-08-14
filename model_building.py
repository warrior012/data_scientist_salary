# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:16:37 2020

@author: Saurabh
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
df = pd.read_csv('eda.csv')
# choose the relevant columns 
df_models = df[["avg_salary","Rating","Size","Sector","Type of ownership","Industry","Revenue","num_comp","hourly","employer_provided","job_state","same_state","age","python_yn","spark","aws","excel","job_simp","seniority","desc_len"]]  

# create the dummy data
df_dum = pd.get_dummies(df_models) 

# train test split of the data
from sklearn.model_selection import train_test_split
X = df_dum.drop("avg_salary",axis = 1)
y = df_dum.avg_salary.values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state =42) 

# multiple linear regression
import statsmodels.api as sm

X_sm= sm.add_constant(X)
model = sm.OLS(y,X_sm)
summ = model.fit().summary()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
ln = LinearRegression()
ln.fit(X_train,y_train)
np.mean(cross_val_score(ln,X_train,y_train,scoring = "neg_mean_absolute_error",cv = 3))
## lasso regression
from sklearn.linear_model import Lasso
ln_l  = Lasso(0.15)
ln_l.fit(X_train,y_train)
np.mean(cross_val_score(ln_l,X_train,y_train,scoring = "neg_mean_absolute_error",cv = 3))
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train,scoring = "neg_mean_absolute_error",cv = 3)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err,columns = ["alpha","error"])
df_err[df_err.error == max(df_err.error)]
# random  forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(criterion='mae', n_estimators=120)
rf.fit(X_train,y_train)
np.mean(cross_val_score(rf,X_train,y_train,scoring = "neg_mean_absolute_error",cv = 3))

# tune models GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters ={
    "n_estimators": range(10,300,10),
    "criterion":('mse','mae'),
    "max_features":('auto',"sqrt",'log2')}
    
gs = GridSearchCV(rf,parameters,scoring ="neg_mean_absolute_error",cv = 3)
gs.fit(X_train,y_train)

ln_pred = ln.predict(X_test)
ln_l_pred = ln_l.predict(X_test)
rf_pred = rf.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,ln_pred)
mean_absolute_error(y_test,ln_l_pred)
mean_absolute_error(y_test,rf_pred)










