# Importing all the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Importing from the sklearn lib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Loading the Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")


# Dividing the dataset in independent varaiable and target value
X = dataset[['Age', 'EstimatedSalary' ] ] 
y = dataset['Purchased']


# Splitting the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


model = KNeighborsClassifier(n_neighbors=os.environ['START'])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error_rate = accuracy_score(y_test, y_pred)
error_rate = int(round(error_rate,4) * 10000)
os.environ['ACCURACY'] = str(error_rate)
print(error_rate)


os.system("python3 counter.py")

# error_rate = []
# for i in range(int(os.environ['START']), int(os.environ['END'])):
#     model = KNeighborsClassifier(n_neighbors=i)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     error_rate.append(accuracy_score(y_test, y_pred))


# print(error_rate.index(max(error_rate))+int(os.environ['START']), max(error_rate))

