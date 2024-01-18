import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain.llms import VertexAI

from langchain import LLMChain,PromptTemplate

st.title('CSV Visualizer')

uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
if uploaded_file:
    # Read in the data, add it to the list of available datasets. Give it a nice name.
    file_name = uploaded_file.name[:-4].capitalize()
    data= pd.read_csv(uploaded_file)

st.write(data)

prompt = "Plot the JobTitle over the salary for each entry in the table, the table is already stored in python as the variable data.\n Make sure that the response only includes code.\n"

# Choose the LLM
llm = VertexAI(temperature=0, model="chat-bison")
llm_prompt = PromptTemplate.from_template(prompt)
llm_chain = LLMChain(llm=llm,prompt=llm_prompt,verbose=True)
llm_response = llm_chain.predict()


plot_area = st.empty()

llm_response = llm_response[10:]
tilde_line = llm_response.find("```")

llm_response = llm_response[:tilde_line]

print(llm_response)

plot_area.pyplot(exec(llm_response))      