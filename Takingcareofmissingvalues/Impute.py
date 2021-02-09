# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 01:05:50 2021

@author: senjupaul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load the Data
data=pd.read_csv('melb_data.csv')
Y=data.Price
X_data=data.drop(['Price'],axis=1)
#Remove the categorical features from the data
X=X_data.select_dtypes(exclude=['object'])

#Function to check the precision of the method
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def modelaccuracy(x_train,x_valid,y_train,y_valid):
    model=RandomForestRegressor(n_estimators=10,random_state=0)
    model.fit(x_train,y_train)
    pred=model.predict(x_valid)
    return mean_absolute_error(y_valid, pred)

#Split the data in to train and validation data
from sklearn.model_selection import train_test_split
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,train_size=0.8,random_state=0)
columnnames=X_train.columns

#First method drop the coloumns with null values
reduced_x_train=X_train.dropna(axis=1)
reduced_x_valid=X_valid.dropna(axis=1)
print(modelaccuracy(reduced_x_train,reduced_x_valid,Y_train,Y_valid))

#Second method to use imputer class to get the median of the  missing value in the column
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
imputed_x_train=pd.DataFrame(imputer.fit_transform(X_train))
imputed_x_valid=pd.DataFrame(imputer.transform(X_valid))
imputed_x_valid.columns=imputed_x_train.columns=columnnames
print(modelaccuracy(imputed_x_train,imputed_x_valid, Y_train, Y_valid))




    
    