import google.auth
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from langchain.schema import ChatMessage
from langchain.agents import AgentExecutor
import time
import streamlit as st

import matplotlib.pyplot as plt
import pandas as pd
from langchain import LLMChain,PromptTemplate

# PROJECT DEFINITIONS
project_id = "poc-ai-assist-tool"
dataset_id = "salesforce_data"
table_id = 'sample_data'

# Message stream handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Creates Langchain agent and connects to Vertex and BigQuery
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

# Downloads the table in order to create visualizations
def download_table_to_csv(project_id, dataset_id, table_id):
    destination_csv_path = 'C:/Users/Marcelino/Desktop/66AI-Assist-Tool/data.csv'
    # Initialize a BigQuery client
    client = bigquery.Client(project=project_id)

    # Construct the BigQuery table reference
    table_ref = client.dataset(dataset_id).table(table_id)

    # Fetch the table data into a Pandas DataFrame
    table = client.get_table(table_ref)
    df = client.list_rows(table).to_dataframe()

    # Export the DataFrame to a CSV file
    df.to_csv(destination_csv_path, index=False)

    data = pd.read_csv(destination_csv_path)

    return data

# Creates a visualization using matplotlib
def create_vis(prompt):
    # Modify the users prompt
    prompt = prompt + "\nThe table is already stored in python as the variable table.\n Make sure that the response only includes code and utilizes streamlit."
    
    # Setup access to Verttex
    llm = VertexAI(temperature=0, model="chat-bison")
    llm_prompt = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm,prompt=llm_prompt,verbose=True)
    llm_response = llm_chain.predict()

    # Format the response to remove unwanted characters
    llm_response = llm_response[10:]
    tilde_line = llm_response.find("```")
    llm_response = llm_response[:tilde_line]
    
    print(llm_response)

    return llm_response

# Sets up basic page elements
def page_setup():
    # Sets the page name,title and sidebar header.
    st.set_page_config(page_title="66 Chatbot", page_icon="🤖")
        
    st.title('AI Assist Tool')
    
    st.sidebar.header("Welcome to the 66Degrees AI Assist Tool", divider='rainbow')

# Creates the sidebar, title and header
page_setup()

# When checked allows for visualizations to be generated
checkVis = st.sidebar.checkbox("Generate Visualization?")

# Creates the agent
agent_executor = create_agent()

# Sends the initial prompt message
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Finds any messages in the state and prints them
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Main loop
if prompt := st.chat_input():
    # Appends any user messages to the state and displays them
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    message_placeholder = st.empty()

    if checkVis:
        table = download_table_to_csv(project_id, dataset_id, table_id)
        response = create_vis(prompt)

        with st.chat_message("assistant"):
            plot_area = exec(response) 
            st.session_state.messages.append(ChatMessage(role="vis", content=response))   
    else:
        response = agent_executor.run(prompt)

        for i in range(len(response) + 1):
            message_placeholder.markdown("%s" % response[0:i])
            time.sleep(0.1)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))

    

