import google.auth
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from langchain.agents import AgentExecutor
import time

import streamlit as st

# Authenticates Google Acct.
credentials, project_id = google.auth.default()

# Sets the page name and icon
st.set_page_config(page_title="66 Chatbot", page_icon="🤖")


def create_agent():
    # Define project and dataset and connect to URI
    project = "poc-ai-assist-tool"
    dataset = "salesforce_data"
    sqlalchemy_url = f'bigquery://{project}/{dataset}'
    #Grab the DB
    db = SQLDatabase.from_uri(sqlalchemy_url)
    # Choose the LLM
    llm = VertexAI(temperature=0, model="chat-bison")
    # Define the toolkit to be used
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(llm=llm,toolkit=toolkit,verbose=1,top_k=10,)


def login_page():
    st.title("Google Auth Test")
    st.subheader("Google Authentication")

    token = st.text_input("Enter your Google ID token", type="password")

    if st.button("Authenticate"):
        try:
            if token == "password":
                return true
            # Continue with the rest of your app logic here
        except ValueError as e:
            st.error("Authentication failed")
            st.error(e)
            return false

def chat_page():
    

    st.title('AI Assist Tool')

    st.sidebar.header("Welcome to the 66Degrees AI Assist Tool", divider='rainbow')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    agent_executor = create_agent()
    # Accept user input
    if prompt := st.chat_input("What are your questions?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

        response = agent_executor.run(prompt)

        for i in range(len(response) + 1):
            message_placeholder.markdown("%s" % response[0:i])
            time.sleep(0.1)

        st.session_state.messages.append({"role": "assistant", "content": response})


chat_page()
    
