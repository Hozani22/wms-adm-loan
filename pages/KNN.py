import pandas as pd 
import streamlit as st 
import plotly.express as px
from openpyxl import Workbook
from PIL import Image
#import pandas as pd, gspread_dataframe  # to read the file as dataframe(table)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm #contain all the clasifications that we will used, Only accept numbers
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB, GaussianNB  # to deal with the catigory values
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statistics import mean, median, variance, stdev
from openpyxl import load_workbook # to modify the data in excel
from app import loan_data_conv, X, Y
from sklearn.naive_bayes import CategoricalNB, GaussianNB  # to deal with the catigory values
from imblearn.over_sampling import SMOTE   #مكتبة موازنة الصفوف
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = loan_data_conv
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, -1].values

# Splitting the data into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

st.write('confusion_matrix')
#رسم مصفوفة الارتباك
fig = plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True)
st.pyplot(fig)

model = KNeighborsClassifier()

# ملائمة النموذج مع بيانات التدريب
model.fit(X_train, Y_train)

#حساب تموذج التصنيف مع بيانات الاختبار
Y_pred = model.predict(X_test)

# معايير التقييم
st.subheader(' KNN  معايير التقييم')
st.write('accuracy_score: ',100 * accuracy_score(Y_test,Y_pred))
st.write('precision_score: ',100 * precision_score(Y_test,Y_pred,average='macro'))
st.write('Recall_score: ',100 * recall_score(Y_test,Y_pred,average='macro'))
st.write('F1_score: ',100*f1_score(Y_test,Y_pred,average='macro'))


#st.table(dataset.iloc[0:10])
