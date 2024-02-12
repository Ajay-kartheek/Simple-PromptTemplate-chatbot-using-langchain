#integrate code with openai
import os
from constants import openai_key
from langchain_openai import OpenAI

from langchain import PromptTemplate
#using the below llmchains because for prompt templates these LLMChains are very beneficial
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"]=openai_key



#Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}"

)
# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key="title",memory=person_memory)

#Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=["title"],
    template="When was {title} born"

)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key="dob",memory=dob_memory)

#Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened around {dob} in the world"

)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key="description",memory=descr_memory)

parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=["name"],output_variables=["title","dob","description"],verbose=True)
#streamlit framework
import streamlit as st

st.title("Celebrity Search Result")
input_text=st.text_input("Search the person you'd like to know")

# openai LLMs

if input_text:
    st.write(parent_chain({"name":input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)
