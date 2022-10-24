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


st.header('Naive_Bayes')

st.dataframe(loan_data_conv)

# To separate the data and label 
st.subheader("To separate the data and label")
X = loan_data_conv.drop(columns=['Loan_Status'],axis=1)
Y = loan_data_conv['Loan_Status']

st.subheader("X Data"); st.dataframe(X)
st.subheader("Y Data"); st.dataframe(Y)
model=CategoricalNB()
model.fit(X,Y)

st.subheader("Train to test Split")  # Done
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)  
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)


st.subheader("Training the model")
clf = svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)

st.subheader("accuracy score on training data")
X_train_prediction = clf.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)
st.write('Accuracy on training data : ', training_data_accuray)


st.subheader("accuracy score on Test data")
X_test_prediction = clf.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
st.write('Accuracy on test data : ', test_data_accuray)

#حساب عدد القيم من كل صف
cv = Y.value_counts()

#قائمة الفهارس
labels = Y.value_counts().index.to_list()

#معاينة نسب توزيع الصفوف
#رسم الفطيرة
st.subheader('معاينة نسب توزيع الصفوف')
fig1, ax1 = plt.subplots(figsize = (6, 6))
ax1.set_title('original classes distribution', fontsize = 14)
ax1.pie(x = cv, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
st.pyplot(fig1)


st.subheader("موازنة الصفوف بإضافة أمثلة جديدة")

#انشاء غرض
oversample = SMOTE()

#الموازنة عن طريق إضافة أمثلة جديدة
X, Y = oversample.fit_resample(X, Y)

#حساب عدد القيم من كل صف
cv = Y.value_counts()

#قائمة الفهارس
labels = Y.value_counts().index.to_list()

#رسم التوزيع الجديد
st.subheader('رسم التوزيع الجديد')
fig1, ax1 = plt.subplots(figsize = (6, 6))
ax1.set_title('original classes distribution', fontsize = 14)
ax1.pie(x = cv, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
st.pyplot(fig1)

Y_pred = model.predict(X)

# معايير التقييم
st.write('\naccuracy: {:.2f}\n'.format(100*accuracy_score(Y, Y_pred)))
st.write('\nprecision: {:.2f}\n'.format(100*precision_score(Y, Y_pred,average='macro')))
st.write('\nRecall: {:.2f}\n'.format(100*recall_score(Y, Y_pred, average='macro')))
st.write('\nF1: {:.2f}\n'.format(100*f1_score(Y,Y_pred,average='macro')))


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training a Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
st.subheader('Making the Confusion Matrix')
fig1, ax1 = plt.subplots(figsize = (6, 6))
ac = accuracy_score(Y_test,Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True)  #
st.pyplot(fig1)

st.write(ac,cm)  #