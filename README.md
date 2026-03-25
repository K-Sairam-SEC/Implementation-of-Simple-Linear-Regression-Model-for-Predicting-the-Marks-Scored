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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("CSVs/student_scores.csv")
df.head()
df.tail()
X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)
plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("MSE : ",mean_squared_error(Ytest,Ypred))
print("MAE : ",mean_absolute_error(Ytest,Ypred))
print("RMSE : ",np.sqrt(mse))
```
### Output:
<table>
<tr>
<td width=45%>

**X and Y values split** <br>
 ![Screenshot 2024-02-26 212209](https://github.com/ROHITJAIND/EX-02-SimpleLinearRegressionModel-for-Predicting-the-Marks-Scored/assets/118707073/5b488f39-2123-4a5c-9708-9bcc33d2d82e)
</td>
</tr> 
</table>
<br>
<br>

**Predicted Values of Y**<br>
![Screenshot 2024-02-26 212638](https://github.com/ROHITJAIND/EX-02-SimpleLinearRegressionModel-for-Predicting-the-Marks-Scored/assets/118707073/e2be81a2-00bc-4298-bfbc-0de77e45b642)
<br>
<br>
**Training and Testing set**<br>
![download](https://github.com/ROHITJAIND/EX-02-SimpleLinearRegressionModel-for-Predicting-the-Marks-Scored/assets/118707073/bc3fc02b-c6b0-40b6-b185-fdfca4659cef)
![download](https://github.com/ROHITJAIND/EX-02-SimpleLinearRegressionModel-for-Predicting-the-Marks-Scored/assets/118707073/ae6dc783-4089-4850-a959-eb8d4bf14dc6)
<br>
**Values of MSE,MAE and RMSE**<br>
```
MSE :  25.463280738222547
MAE :  4.691397441397438
RMSE :  5.046115410711743
```


### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
