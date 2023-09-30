# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.


## Program:
```python
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sai Praneeth K
RegisterNumber:  212222230067


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset

dataset.head(10)

dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset=dataset.drop(['sl_no','ssc_p','gender'],axis=1)
dataset

dataset.shape

dataset.info()

dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] =dataset["hsc_b"].astype('category')
dataset["hsc_s"] =dataset["hsc_s"].astype('category')
dataset["degree_t"] =dataset["degree_t"].astype('category')
dataset["workex"] =dataset["workex"].astype('category')
dataset["specialisation"] =dataset["specialisation"].astype('category')
dataset["status"] =dataset["status"].astype('category')
dataset

dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] =dataset["hsc_b"].cat.codes
dataset["hsc_s"] =dataset["hsc_s"].cat.codes
dataset["degree_t"] =dataset["degree_t"].cat.codes
dataset["workex"] =dataset["workex"].cat.codes
dataset["specialisation"] =dataset["specialisation"].cat.codes
dataset["status"] =dataset["status"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0, solve='lbfgs',max_iter=1000).fit(x_train,y_train)
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0,87,0,95,0,2,78,2,0,0]])
```

## Output:
![Alt text](<Screenshot (135).png>) ![Alt text](<Screenshot (136).png>) ![Alt text](<Screenshot (137).png>) ![Alt text](<Screenshot (139).png>) ![Alt text](<Screenshot (141).png>) ![Alt text](<Screenshot (142).png>) ![Alt text](<Screenshot (143).png>) ![Alt text](<Screenshot (144).png>) ![Alt text](<Screenshot (145).png>)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
