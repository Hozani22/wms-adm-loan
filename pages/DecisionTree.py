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
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


st.subheader("Train to test Split")  # Done
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)  
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)

#التصريح عن النموذج
#مع اختيار معيار الانتروبي لاختيار الواصفات
model = DecisionTreeClassifier(criterion= "entropy")

# ملائمة النموذج مع بيانات التدريب
model.fit(X_train, Y_train)

#حساب تموذج التصنيف مع بيانات الاختبار
Y_pred = model.predict(X_test)



# معايير التقييم
st.write('accuracy: ',100 * accuracy_score(Y_test,Y_pred))
st.write('precision: ',100 * precision_score(Y_test,Y_pred,average='macro'))
st.write('Recall: ',100 * recall_score(Y_test,Y_pred,average='macro'))
st.write('F1: ',100*f1_score(Y_test,Y_pred,average='macro'))
