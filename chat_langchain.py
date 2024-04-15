import json
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from langchain.schema import ChatMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
import streamlit as st

import matplotlib.pyplot as plt

# PROJECT DEFINITIONS
project_id = "poc-ai-assist-tool"
dataset_id = "salesforce_data"
table_id = 'sample_data'

# Creates Langchain agent and connects to Vertex and BigQuery
def create_agent():
    # Connect to URI with the table    
    sqlalchemy_url = f'bigquery://{project_id}/{dataset_id}'
    # Grab the DB
    db = SQLDatabase.from_uri(sqlalchemy_url)
    # Set our LLM to chat-bison
    llm = VertexAI(temperature=0, model="chat-bison",max_output_tokens=500)
    # Define the toolkit to be used
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(llm=llm,toolkit=toolkit,verbose=1,top_k=10)

# Sets up basic page elements
def page_setup():
    # Sets the page name,title and sidebar header.
    st.set_page_config(page_title="66 Chatbot", page_icon="🤖")
        
    st.title('AI Assist Tool')
    
    st.sidebar.header("Welcome to the 66Degrees AI Assist Tool", divider='rainbow')


# Creates and returns Bar Charts
def bar_chart(data_string, x_name, y_name):
    # Converts the data from string to JSON
    data = json.dumps(data_string)
    data = json.loads(data)

    # Extracts the key value pairs and plots them
    for key, values in data.items():
        plt.bar(x=key,height = float(values))



    # Formats the graph and returns it
    plt.title('Bar Graph')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(rotation=45, fontsize=8)
    return plt

# Creates and returns Scatter Plot charts
def scatter_plot(data_string, x_name, y_name):
    # Converts the data from string to JSON
    data = json.dumps(data_string)
    data = json.loads(data)

    # Extracts the key value pairs and plots them
    for key, values in data.items():
        plt.scatter(x=float(key),y = float(values))

    # Formats the graph and returns it
    plt.title('Scatter Plot')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(rotation=45, fontsize=8)
    return plt

# TODO_MARCELINO: Fix bug related to Pie Charts
def pie_chart(data_string):
    # Converts the data from string to JSON
    data = json.dumps(data_string)
    data = json.loads(data)

    # Extracts the key value pairs and plots them
    for key, values in data.items():
        plt.pie(x=float(values),labels=key)

    # Formats the graph and returns it
    plt.title('Pie Chart')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(rotation=45, fontsize=8)
    return plt

# Helper function that converts to a dict that can be dumped into JSON
def format_to_JSON(raw_data):
    raw_data = raw_data.split(", ")
    data  = {}
    for pair in raw_data:
        key, value = pair.split(": ")
        data[key] = value

    return data
    

# Creates the sidebar, title and header
page_setup()

# When checked allows for visualizations to be generated
checkVis = st.sidebar.checkbox("Generate Visualization?")

if checkVis:
    selectbox = st.sidebar.selectbox("Select Visualization Type:",("None", "Bar Chart", "Scatter Plot", "Pie Chart"))

    if selectbox == "Bar Chart" or selectbox == "Scatter Plot":
        x_name = st.sidebar.text_input(label="X-Axis Name")
        y_name = st.sidebar.text_input(label="Y-Axis Name")

# Sends the initial prompt message
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Finds any messages in the state and prints them
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# Creates the agent
agent_executor = create_agent()

# Main loop
if prompt := st.chat_input():
    # Appends any user messages to the state and displays them
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    message_placeholder = st.empty()

    if checkVis and selectbox != "None":
        # Generate a response
        # TODO_MARCELINO: Generate the data first, then format it in a seperate step with a new prompt. Could lead to more consistency.
        response = agent_executor.run(prompt+" Return the data in a key value lists with no added text, characters or [] brackets.")        
        print(response)
        
        # Sometimes the AI acts in unexpected ways and returns bad data, so we try and catch errors.
        try:

            # Format the data to JSON
            data = format_to_JSON(response)
            print(data)

            # Use the JSON to generate a visualization based on what was selected
            if selectbox == "Bar Chart":
                st.session_state.messages.append(ChatMessage(role="assistant", content="Bar Chart Visualization"))
                st.chat_message("assistant").pyplot(bar_chart(data, x_name, y_name))            
            elif selectbox == "Scatter Plot":
                st.session_state.messages.append(ChatMessage(role="assistant", content="Scatter Plot Visualization"))
                st.chat_message("assistant").pyplot(scatter_plot(data, x_name, y_name))
            elif selectbox == "Pie Chart":
                st.session_state.messages.append(ChatMessage(role="assistant", content="Pie Chart Visualization"))
                st.chat_message("assistant").write(pie_chart(data))        
        except Exception as e:
            print(e)
            st.chat_message("assistant").write("Sorry, an unexpected error occured. Please try your query again.")
            st.session_state.messages.append(ChatMessage(role="assistant", content="Sorry, an unexpected error occured. Please try your query again."))   
    # Displays the answer in text as opposed to a visualization 
    else:
        response = agent_executor.run(prompt)

        for i in range(len(response) + 1):
            message_placeholder.markdown("%s" % response[0:i])
            time.sleep(0.05)
        
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        st.chat_message("assistant").write(response)

    

