# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('food_items (1).csv')
print('Name:K Sadhana')
print('Reg.No: 212225240128')
print('Dataset Overview')
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000
l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)
print('Name: K SADHANA')
print('Reg. No: 212225240128')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print("Name: K SADHANA")
print("Reg. No: 212225240128")
```

## Output:
<img width="831" height="660" alt="image" src="https://github.com/user-attachments/assets/7414a1c0-85ca-4f8f-9cc0-e093f34f8484" />

<img width="716" height="580" alt="image" src="https://github.com/user-attachments/assets/34e14c28-ac75-4ea9-818b-7f1407ef0971" />

<img width="653" height="358" alt="image" src="https://github.com/user-attachments/assets/02ccea4a-3ac7-4218-91af-656b07f6380f" />

<img width="364" height="110" alt="image" src="https://github.com/user-attachments/assets/e05456fd-6141-4dd9-a2bf-0b4287e13de7" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
