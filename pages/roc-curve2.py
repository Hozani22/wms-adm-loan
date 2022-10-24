import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
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


#Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)


from sklearn.model_selection import train_test_split
#Split the dataset into 70% training set and 30% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,   random_state=23)


from sklearn.neighbors import KNeighborsClassifier
 
#Logistic regression 
clf1 = LogisticRegression(max_iter=1000)
 
#KNN
clf2 = KNeighborsClassifier(n_neighbors=4)



#Fit the classifiers
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
 
#Predict the probabilities
pred_prob1 = clf1.predict_proba(X_test)
pred_prob2 = clf2.predict_proba(X_test)


from sklearn.metrics import roc_curve
 
#ROC curve for classifiers
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
 
#ROC curve for TPR=FPR
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


from sklearn.metrics import roc_auc_score
 
#AUC scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
 
print("Logistic Regression AUC Score:", auc_score1) 
print("KNN AUC Score:", auc_score2)


#Plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
#title
plt.title('ROC curve')
#x-label
plt.xlabel('False Positive Rate')
#y-label
plt.ylabel('True Positive rate')
 
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()

st.subheader("مكتبة موازنة الصفوف")

#انشاء غرض
oversample = SMOTE

#الموازنة عن طريق إضافة أمثلة جديدة
X,y = oversample.fit_resample(X,y)

#حساب عدد القيم من كل صف
cx = y.value_count()

#قائمة الفهارس
labels = Y.value_counts().index.to_list()

fig1, ax1 = plt.subplots(figsize = (6, 6))
ax1.pie(x = cx, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.set_title('original classes distribution', fontsize = 14)
plt.show()
st.pyplot(fig1)