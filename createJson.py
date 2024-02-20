import google.auth
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI

import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st

# PROJECT DEFINITIONS
project_id = "poc-ai-assist-tool"
dataset_id = "salesforce_data"
table_id = 'sample_data'

st.title('AI Assist Tool')

def create_agent():
    # Connect to URI with the table    
    sqlalchemy_url = f'bigquery://{project_id}/{dataset_id}'
    # Grab the DB
    db = SQLDatabase.from_uri(sqlalchemy_url)
    # Set our LLM to chat-bison
    llm = VertexAI(temperature=0, model="chat-bison")
    # Define the toolkit to be used
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(llm=llm,toolkit=toolkit,verbose=1,top_k=10,)

agent_executor = create_agent()

agent_executor.run("Select the EmployeeID and Name columns")