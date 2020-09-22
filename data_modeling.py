#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:33:43 2020

@author: vikashchavda
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('car_price_eda_data.csv')


df_model = df[['price', 'mileage',
       'age', 'diesel_yn', 'petrol_yn',
       'hybrid_yn', 'manual_yn', 'semi_automatic_yn', 'automatic_yn',
       'A_Series_yn', 'T_Series_yn', 'S_Series_yn', 'R_Series_yn',
       'Q_Series_yn','model_type']]


# get dummy data : for each type of categorical variable (job type)
"""
# make a column for each one, has a 1 if that contains a job type, or 0, 
# increase the number of columns
"""

df_dum = pd.get_dummies(df_model)

# create a train test split  : train set, test set, validation set.
# create the X and y variables:

X = df_dum.drop('price', axis = 1)
y = df_dum.price.values
# using the .values creates an array instead of a series : recommended for models
"""
obtained code from
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
to get the know how on creating the train_test_split
This wil
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
# 0.3 = 70% in train set, 30% in test set.


# multiple linear regression
# creating in statmodels, and sklearn
"""
https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
for information on statsmodels

"""

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


"""
importing the sklearn liner model
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
use it for cross validation
takes/pulls samples from the model, and a validation set
run model on sample, and validate on the validation set
its like a mini train_test_split
gives us a good indication on how the model is performing,
baseline evaluate the other tests off of
"""
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV


lm = LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm, X_train,y_train, scoring= 'neg_mean_absolute_error',cv=3)

np.mean(cross_val_score(lm, X_train,y_train, scoring= 'neg_mean_absolute_error',cv=3))
# I used 3
# on average around -4508.13907315756 off

# lasso regression :  help with spare data set (with all these dummy variables)

"""
Lasso regression is a type of linear regression that uses shrinkage. Shrinkage 
is where data values are shrunk towards a central point, like the mean. The lasso 
procedure encourages simple, sparse models (i.e. models with fewer parameters).

parameter Alpha:normalisation term
    Alpha is zero, same as the ols multiple linear regression
    as alpha increases, increases the amount the data is smoothed

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
"""

lm_l = Lasso()
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l, X_train,y_train, scoring= 'neg_mean_absolute_error',cv=3))

# values after lasso regression is -4507.249748630279

# try different values of Alpha

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/0.5)
    lml = Lasso(alpha=(i/0.5))
    error.append(np.mean(cross_val_score(lml, X_train,y_train, scoring= 'neg_mean_absolute_error',cv=3)))

plt.plot(alpha,error)


# based on these values, distribution looks weird, but peaks around 88, which gives us the best error term:
# -4474.49, improvement on linear regression

err = tuple(zip(alpha,error)) # ties them together, turns it into a list
df_err = pd.DataFrame(err,columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


# random forest : expect to perform well, tree based decision process, a lot of zero/1 values
# don't have to worry about multi-colinearaity
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring= 'neg_mean_absolute_error',cv=3))

# the values are further improved : from -4474.49 to -4248.83
# now time to tune this!

# (other types that can be used: gradient boosted tree,support vector regression)

# tune models : GridSearchCV, imported above
"""
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

GridSearchCV is a library function that is a member of sklearn's model_selection package. 
It helps to loop through predefined hyperparameters and fit your estimator (model) on your training set. 
So, in the end, you can select the best parameters from the listed hyperparameters.

# add the import of GridsearchCV above (with the cross_val_score)

parameters to be tuned: n_estimators
                        criterion : Mse = mean squared error, mae : mean absolute error
                        max_features : auto - max_features = n_features
                                       sqrt - max_features = sqrt(n_features)
                                       log2 - max_features = log2(n_features)
can do a normal gridsearch or randomised gridsearch, randmised is a sample, and its faster
"""


parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2') }

gs = GridSearchCV(rf,parameters, scoring='neg_mean_absolute_error',cv=3)

# then fit it, uncomment this to refit, else it will take forever
#gs.fit(X_train,y_train)

gs.best_score_ # -4154.04
gs.best_estimator_ #n_estimators=100, min_samples_split=2

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tpred_lm) # 4672
mean_absolute_error(y_test,tpred_lml) # 4671
mean_absolute_error(y_test,tpred_rf) # 4162 knocking this out of the park!

# combine a couple of models, it might help

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2) # 4172.6
mean_absolute_error(y_test,(tpred_lml+tpred_rf)/2) # 4171.7

"""
can run this through a regression model and obtain weights for it
could take 90% of the random forest and 10% of the other model

better to combine the models, better for classifcation approach
"""

mean_absolute_error(y_test,((tpred_lm*0.1)+(tpred_rf*0.9))) # 4110.5, so better, RF gives the best values



import pickle
pickl = {'model' : gs.best_estimator_}
pickle.dump(pickl,open('model_file'+".p","wb")) # creates model_file.p

#load back in

file_name = "model_file.p"
with open(file_name,'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
 
model.predict(X_test.iloc[1,:].values.reshape(1,-1))

#first value returned is 19K thousand

#make a dummy file with X_test.iloc[1:]

list(X_test.iloc[1,:])