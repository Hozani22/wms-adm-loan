import pandas as pd 
import streamlit as st 

my_variable = "Home page"

def main():
    st.set_page_config(page_title='Streamlit Multi-page')  # or st.title('Streamlit Multi-page')
    st.header('MWS _ ADM _ Loan ')
    st.write(my_variable)
    

if __name__ == '__main__':
    main()
