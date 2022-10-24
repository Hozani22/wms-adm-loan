import pandas as pd 
import streamlit as st 
import plotly.express as px
from openpyxl import Workbook
from PIL import Image
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

st.dataframe(loan_data_conv)

# To separate the data and label 
st.subheader("To separate the data and label")
X = loan_data_conv.drop(columns=['Loan_Status'],axis=1)
Y = loan_data_conv['Loan_Status']

st.subheader("X Data"); st.dataframe(X)
st.subheader("Y Data"); st.dataframe(Y)
st.subheader("Bayes")
model=CategoricalNB()
model.fit(X,Y)

st.subheader("Train to test Split")  # Done
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)  
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)

# Predicting the Test set results
Y_pred = model.predict(X_test)

# Making the Confusion Matrix     #حساب مصفوفة الارتباك
cf_matrix = confusion_matrix(Y_test,Y_pred)

st.write(cf_matrix)

#رسم مصفوفة الارتباك
fig = plt.figure(figsize=(8, 4))
sns.heatmap(cf_matrix, annot=True)
st.pyplot(fig)