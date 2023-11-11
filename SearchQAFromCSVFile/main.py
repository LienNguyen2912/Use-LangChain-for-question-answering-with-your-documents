import streamlit as st
import pandas as pd
from langchain_helper import getAnswer

st.title("Question answering with your documents")
question = st.text_input("Question: ")

if question:
    answer, context = getAnswer(question)
    st.header("Answer")
    st.write(answer)
    st.header("Context retrieved from vector DB")
    st.write(context)


df_train = pd.read_csv("./csv/mytrain.csv") 
st.header("Data used to create a vector database")
st.write(df_train)  # visualize my dataframe in the Streamlit app

df_test = pd.read_csv("./csv/mytest.csv") 
st.header("Test data")
st.write(df_test)  # visualize my dataframe in the Streamlit app