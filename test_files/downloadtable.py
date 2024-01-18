from google.cloud import bigquery
import pandas as pd

def download_table_to_csv(project_id, dataset_id, table_id, destination_csv_path):
    # Initialize a BigQuery client
    client = bigquery.Client(project=project_id)

    # Construct the BigQuery table reference
    table_ref = client.dataset(dataset_id).table(table_id)

    # Fetch the table data into a Pandas DataFrame
    table = client.get_table(table_ref)
    df = client.list_rows(table).to_dataframe()

    # Export the DataFrame to a CSV file
    df.to_csv(destination_csv_path, index=False)

# Example usage
project_id = 'poc-ai-assist-tool'
dataset_id = 'salesforce_data'
table_id = 'sample_data'
destination_csv_path = 'C:/Users/Marcelino/Desktop/66AI-Assist-Tool/data.csv'

download_table_to_csv(project_id, dataset_id, table_id, destination_csv_path)
