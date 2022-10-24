#import pandas as pd 
#import streamlit as st 

#my_variable = "Home page"

#def main():
  #  st.set_page_config(page_title='Streamlit Multi-page')  # or st.title('Streamlit Multi-page')
   # st.header('MWS _ ADM _ Loan ')
    #st.write(my_variable)
    

#if __name__ == '__main__':
    #main()
    
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
import pickle 
from sklearn.ensemble import RandomForestClassifier 


my_variable = "Home page"

def main():
    st.set_page_config(page_title='Streamlit Multi-page')  # or st.title('Streamlit Multi-page')
    st.header('MWS _ ADM _ Loan ')
    #st.subheader('Was the tutorial helpful?')
    st.write(my_variable)
    
    #filterValue = st.selectbox(["option1", "option2", "option3"])
    #df = df[df["Type"]==filterValue]

    #choice = st.sidebar.selectbox("Submenu",["Pandas","Tensforflow"])
    #if choice == "Pandas":
    #    st.subheader("Pandas")

#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()

## load Data
excel_file = 'dataset.csv' #F:/xampp/htdocs/MS/firstPY/dataset.csv
loan_data = pd.read_csv(excel_file,header=0)

st.dataframe(loan_data)


# To change all ‘NaN’ value of the columns for which we have its mode except LoanAmount column.
def fill_na(loan_data):
    mGender = loan_data['Gender'].mode()[0]
    mMarried = loan_data['Married'].mode()[0]
    mLoan_Amount_Term = loan_data['Loan_Amount_Term'].mode()[0]
    mSelfEmp = loan_data['Self_Employed'].mode()[0]
    mDependents = loan_data['Dependents'].mode()[0]
    mCredit_History = loan_data['Credit_History'].mode()[0]
    loan_data['Gender'].fillna(mGender, inplace=True)
    loan_data['Married'].fillna(mMarried, inplace=True)
    loan_data['Loan_Amount_Term'].fillna(mLoan_Amount_Term, inplace=True)
    loan_data['Self_Employed'].fillna(mSelfEmp, inplace=True)
    loan_data['Dependents'].fillna(mDependents, inplace=True)
    loan_data['Credit_History'].fillna(mCredit_History, inplace=True)
    # To change all ‘NaN’ value of the LoanAmount columns for which we have its mean.
    mean_value = loan_data['LoanAmount'].mean()
    loan_data['LoanAmount'].fillna(mean_value, inplace=True) #, inplace=True
    loan_data = loan_data.replace(to_replace='3+', value=3) # replacing the value of 3+ to 3
    loan_data = loan_data.drop(columns=['Loan_ID'],axis=1)
    return(loan_data)

loan_data = fill_na(loan_data)

st.subheader("After correct the data")

st.dataframe(loan_data)

st.subheader("dependent values")

st.dataframe(loan_data['Dependents'].value_counts())

# marital status & Loan Status
def maritalPlot():
    st.subheader("marital status & Loan Status")
    fig = plt.figure(figsize=(4, 2))
    sns.countplot(data=loan_data, hue='Loan_Status', x='Married', palette=["#fc9272","#fee0ff"])
    st.pyplot(fig)

maritalPlot()

def eduPlot():
    st.subheader("education & Loan Status")
    fig = plt.figure(figsize=(4, 2))
    sns.countplot(data=loan_data, x='Education',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
    st.pyplot(fig)

eduPlot()

#----------------------------------------------------------------------------

def convert_val(loan_data):
    st.subheader("convert categorical columns to numerical values")
    loan_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
    return(loan_data)

loan_data_conv = convert_val(loan_data)

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
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 10)  
#x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)

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
