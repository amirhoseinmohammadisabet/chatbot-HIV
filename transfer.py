import pandas as pd
import json


def excel_to_json(excel_file, json_file):
    # Read the Excel file into a pandas DataFrame
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Check if the DataFrame has at least two columns
    if len(df.columns) < 2:
        print("Error: The Excel file must have at least two columns.")
        return

    # Convert DataFrame to a dictionary with the first column as key and the second column as value
    data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    # Write the dictionary to a JSON file
    with open(json_file, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)


excel_file_path = 'Data/diagnosis.xlsx'
json_file_path = 'Data/diagnosis.json'

excel_to_json(excel_file_path, json_file_path)
