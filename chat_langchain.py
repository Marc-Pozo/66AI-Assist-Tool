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

import pandas as pd

# Message stream handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Creates Langchain agent and connects to Vertex and BigQuery
def create_agent(option):
    # Authenticates Google Acct.
    credentials, project_id = google.auth.default()
    # Define project and dataset and connect to URI
    project = "poc-ai-assist-tool"
    dataset = option
    sqlalchemy_url = f'bigquery://{project}/{dataset}'
    #Grab the DB
    db = SQLDatabase.from_uri(sqlalchemy_url)
    # Choose the LLM
    llm = VertexAI(temperature=0, model="chat-bison")
    # Define the toolkit to be used
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(llm=llm,toolkit=toolkit,verbose=1,top_k=10,)

# Sets the page name and icon
st.set_page_config(page_title="66 Chatbot", page_icon="🤖")

# Sets the title text    
st.title('AI Assist Tool')

# Sets the sidebar
st.sidebar.header("Welcome to the 66Degrees AI Assist Tool", divider='rainbow')

# Creates a dropdown 
option = st.selectbox(
    'Which dataset would you like to query?',
    ('salesforce_data','other'))

# Creates the agent
agent_executor = create_agent(option)
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

    response = agent_executor.run(prompt)

    for i in range(len(response) + 1):
        message_placeholder.markdown("%s" % response[0:i])
        time.sleep(0.1)

    st.session_state.messages.append({"role": "assistant", "content": response})

