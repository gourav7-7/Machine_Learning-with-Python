#1 linear regression
#import package
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

#importing dataset
data_set= pd.read_csv('Salary_Data.csv')

#seperating dependent from independent variables
x= data_set.iloc[:,:-1].values
y= data_set.iloc[:,-1].values



#splitting data into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=1/3, random_state=0)



#fitting model in linear regression
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#preditction of test and training set result
y_pred= regressor.predict(x_test)
x_pred= regressor.predict(x_train)

#data visualization
mtp.scatter(x_train, y_train, color="green")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Salary vs Experience")
mtp.xlabel("Years of Experience")
mtp.ylabel("Salary(In Rupees)")
mtp.show()  

#plot for the TEST
 
mtp.scatter(x_test, y_test, color='red') 
mtp.plot(x_train, regressor.predict(x_train), color='blue') # plotting the regression line
mtp.title("Salary vs Experience (Testing set)")
mtp.xlabel("Years of experience") 
mtp.ylabel("Salaries") 
mtp.show() 

# Import metrics library
from sklearn import metrics

# Print result of MAE
print(metrics.mean_absolute_error(y_test, y_pred))

# Print result of MSE
print(metrics.mean_squared_error(y_test, y_pred))
