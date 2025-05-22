
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.  
 
2.Upload and read the dataset. 
 
3.Check for any null values using the isnull() function. 
 
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy  
 
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARIPRASHAAAD RA 
RegisterNumber:  212223040060
*/
```

```
import pandas as pd


data = pd.read_csv("Employee.csv")
data.head()
data.info()




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])


x = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
y = data['left']
x.head()


y.head()




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train, y_train)



y_pred = dt.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)




dt.predict([[0.5, 0.8, 3, 260,6,0,1,2]])





```
## Output:
![image](https://github.com/user-attachments/assets/bd381779-47e0-49e5-8bff-084341d44596)
![image](https://github.com/user-attachments/assets/35bbec02-60db-42f5-87f9-f7baf6bc00ce)
![image](https://github.com/user-attachments/assets/51abef32-c410-481c-836b-a7cf07767dcb)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
