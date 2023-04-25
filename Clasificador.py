# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 00:16:58 2023

@author: Bee
"""

#1. Importando datos desde una ubicación de drive.
import pandas as pd
import numpy as np

#Librerías para la evaluación de desempeño
from sklearn.metrics import confusion_matrix as CM,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score as PR
from sklearn.metrics import recall_score as RC
from sklearn.metrics import f1_score as F1

#Cargando datos pima indians dataset
dataset = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx',  sheet_name='Data')

#2. split into input (X) and output (Y) variables
dataset = np.array(dataset)

X1 = dataset[:, 1:9]
X2 = dataset[:, 10:14]

X = np.concatenate((X1, X2), axis=1)
Y = dataset[:,9]

#3. Split input and output into training (Reference) and testing sets
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=1)

#4. Data stardardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

#5. Utilizando el clasificador KNN
from sklearn.neighbors import KNeighborsClassifier as KNN
Model_1 = KNN(5)
Model_1.fit(X_train, Y_train)
Y_pred_1 =Model_1.predict (X_test)
#6. Evaluando casos mediante Naive Bayes
from sklearn.naive_bayes import GaussianNB
Modelo_2 = GaussianNB()
Modelo_2.fit(X_train, Y_train)
Y_pred_2 =Modelo_2.predict (X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
Modelo_3 = LDA()
Modelo_3.fit(X_train, Y_train)
Y_pred_3 =Modelo_3.predict (X_test)

#8. Evaluando casos mediante QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
Modelo_4 = QDA()
Modelo_4.fit(X_train, Y_train)
Y_pred_4 =Modelo_4.predict (X_test)


from sklearn.linear_model import LogisticRegression as LR
Modelo_5 = LR()
Modelo_5.fit(X_train, Y_train)
Y_pred_5 =Modelo_5.predict (X_test)

#10. Evaluando casos mediante árboles de decisión
from sklearn.tree import DecisionTreeClassifier as DT
Modelo_6 = DT()
Modelo_6.fit(X_train, Y_train)
Y_pred_6 =Modelo_6.predict (X_test)

#11. Evaluando casos mediante Boosting Adaptativo
from sklearn.ensemble import AdaBoostClassifier

Modelo_7 = AdaBoostClassifier()
Modelo_7.fit(X_train, Y_train)
Y_pred_7 = Modelo_7.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

Modelo_8 = RandomForestClassifier()
Modelo_8.fit(X_train, Y_train)
Y_pred_8 = Modelo_8.predict(X_test)


print("Accuracy KNN",ACC(Y_test, Y_pred_1))
print("Accuracy Bayes",ACC(Y_test, Y_pred_2))
print("Accuracy LDA",ACC(Y_test, Y_pred_3))
print("Accuracy QDA",ACC(Y_test, Y_pred_4))
print("Accuracy LR",ACC(Y_test, Y_pred_5))
print("Accuracy Decision Tree",ACC(Y_test, Y_pred_6))
print("Accuracy AdaBoost", ACC(Y_test, Y_pred_7))
print("Accuracy Random Forest", ACC(Y_test, Y_pred_8))