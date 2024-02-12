#integrate code with openai
import os
from constants import openai_key
from langchain_openai import OpenAI
os.environ["OPENAI_API_KEY"]=openai_key

import streamlit as st

#streamlit framework

st.title("Langchain Demo With OpenAI API")
input_text=st.text_input("Search the topic you want")

# openai LLMs
llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
