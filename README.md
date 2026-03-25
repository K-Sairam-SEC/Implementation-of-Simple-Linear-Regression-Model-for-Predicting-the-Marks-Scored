# EX-02 Implementation of Simple Linear Regression Model for Predicting the Marks Scored

### AIM:
To predict the marks scored by a student using the simple linear regression model.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
   
### Developed By: SAIRAM K
### Register No: 212225270133

### Algorithm
1. Import pandas, numpy and sklearn.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
2. Calculate the values for the training data set.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
3. Calculate the values for the test data set.
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE.


### Program:
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/student_scores.csv')
print(df.head())
print(df.tail())

X = df.iloc[:,:-1].values
print(X)
Y = df.iloc[:,-1].values
print(Y)

print(X.shape)
print(Y.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE = np.sqrt(MSE)
print('RMSE = ',RMSE)

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='green')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
### Output:
## Output:
![image](https://github.com/user-attachments/assets/309346dc-9bd3-41a1-96ed-839305d70fb8)
![image](https://github.com/user-attachments/assets/8da9797b-7196-4018-80c6-fca5acaf1b4f)
![image](https://github.com/user-attachments/assets/7cde7ac9-6431-4831-b8ac-c90531863258)
![image](https://github.com/user-attachments/assets/0a4b57d1-a337-48ca-8e9b-a1b51e7dfe63)

### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
