#2 multiple linear regression
#import liraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

#import dataset
data= pd.read_csv('cars.csv')

#seperating independent and independet variables
x= data.iloc[:,[2,3]].values
y =data.iloc[:,-1].values

#splittig dataset ito training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_test)

#plot a graph
ax=mtp.axes(projection= "3d")
ax.scatter3D(x_train[:,0], x_train[:,1], y_train, color="red",label="Training")
ax.scatter3D(x_test[:,0], x_test[:,1],y_test, color="blue",label="Testing")
ax.set_xlabel("Volume")
ax.set_ylabel("Weight")
ax.set_zlabel("CO2")
ax.legend(loc="upper right")
mtp.show() 
mtp.scatter(y_test,y_pred)
mtp.show()

