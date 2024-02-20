import json
import streamlit as st
import matplotlib.pyplot as plt

st.title('Bar Chart Test')

raw_data = "Employee2355 92585.89, Employee2516 99436.01, Employee6853 67194.92, Employee4411 51595.18, Employee3336 81449.96, Employee6245 57101.04, Employee8552 69402.14, Employee4235 62160.5, Employee8727 64283.42, Employee6307 56679.2"

raw_data = raw_data.split(", ")
data  = {}
for pair in raw_data:
    key, value = pair.split(" ")
    data[key] = value


data_string = json.dumps(data)
json_data = json.loads(data_string)



for key, values in json_data.items():
        plt.bar(x=key,height = float(values))

plt.title('Bar Graph from JSON')
plt.xticks(rotation=45, fontsize=8)
plt.show()
plot_area = st.empty()
plot_area.pyplot(plt)