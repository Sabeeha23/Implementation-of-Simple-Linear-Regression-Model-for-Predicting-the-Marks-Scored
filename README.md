# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import required libraries (numpy, pandas, matplotlib, sklearn).

2. Load the dataset using pandas.

3. Display dataset records and check for missing values.

4. Plot a scatter graph to view the relationship between study hours and marks.

5. Define independent (X) and dependent (Y) variables.

6. Split the dataset into training and test sets.

7. Import LinearRegression from sklearn and create a model.

8. Train the model using training data.

9. Predict marks for test data and compare with actual values.

10. Evaluate model performance using MAE, MSE, RMSE, and R² score, and visualise results.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sabeeha Shaik
RegisterNumber: 212223230176
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()
```
```
#segregating data to variables
X = df.iloc[:,:-1].values
print(X)
```
```
Y=df.iloc[:,1].values
print(Y)
```
```
#splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
```
#displaying predicted values
print(Y_pred)
```
```
#display actual values
print(Y_test)
```
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
```
#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
#Graph plot for test data
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_test,Y_pred,color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name : Sabeeha Shaik")
print("RegNo : 212223230176")
```
## Output:
## Displaying the content in datafield
## head:
![image](https://github.com/user-attachments/assets/2a32dc6b-fa60-4210-9b5e-659725660cbc)
## tail:
![image](https://github.com/user-attachments/assets/8910e2d5-e06b-44d3-a6ff-50cdbbfbfc46)
## Segregating data to variables
![image](https://github.com/user-attachments/assets/c45ea171-e6be-47fb-8ae8-5e5da3df4fb8)
## Displaying predicted values
![image](https://github.com/user-attachments/assets/66b953b0-81e4-4fdd-9774-9a820870777c)
## Displaying actual values
![image](https://github.com/user-attachments/assets/026e7e54-20ec-4277-ad51-242288fec962)
## MSE MAE RMSE
![image](https://github.com/user-attachments/assets/6691e217-2780-4c13-9853-582ae75cc320)
## Graph plot for training data
![image](https://github.com/user-attachments/assets/00ff5ff9-c644-4b5a-8fd9-1aa631d51234)
## Graph plot for test data
![image](https://github.com/user-attachments/assets/c5e70129-7ed9-4246-8c6f-fb418b67b363)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
