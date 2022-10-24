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
from sklearn.metrics import accuracy_score
from statistics import mean, median, variance, stdev
from openpyxl import load_workbook # to modify the data in excel
import pickle 
from sklearn.ensemble import RandomForestClassifier
from app import loan_data_conv 

# To separate the data and label 
#st.subheader("To separate the data and label")
#X = loan_data_conv.drop(columns=['Loan_Status'],axis=1)
#Y = loan_data_conv['Loan_Status']

st.header("Loan Prediction Automated for Learned Machine model")

train = loan_data_conv  #.dropna()
#train.isnull().sum()

X = train[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
Y = train.Loan_Status
X.shape, Y.shape

st.subheader("X Data"); st.dataframe(X)
st.subheader("Y Data"); st.dataframe(Y)
model=CategoricalNB()
model.fit(X,Y)

st.subheader("Train to test Split")  # Done
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 10)  
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)

model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(X_train, Y_train)

pred_test = model.predict(X_test)
accuracy_score(Y_test,pred_test)

pred_train = model.predict(X_train)
accuracy_score(Y_train,pred_train)

st.write('saving the model as classifier.pkl file')
# saving the model 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()

# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

st.subheader("accuracy score on Test data")
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
st.write('Accuracy on test data : ', test_data_accuray)

@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1
 
    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1  
 
    LoanAmount = LoanAmount / 1000
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred

def calc():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female"))
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.number_input("Applicants monthly income") 
    LoanAmount = st.number_input("Total loan amount")
    Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    result =""
    
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Your loan is {}'.format(result))
        st.write(LoanAmount)

calc()