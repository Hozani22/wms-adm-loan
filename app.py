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
#import plotly.express as px
#from openpyxl import Workbook
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
    st.write(my_variable)


#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()

## load Data
excel_file = 'dataset.csv' #F:/xampp/htdocs/MS/firstPY/dataset.csv
loan_data = pd.read_csv(excel_file,header=0)

st.dataframe(loan_data)
