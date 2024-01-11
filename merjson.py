import json

def merge_json_files(output_file, *input_files):
    merged_data = {}

    # Function to convert all text to lowercase
    def convert_to_lowercase(data):
        return {key.lower(): value.lower() if isinstance(value, str) else value for key, value in data.items()}

    # Iterate through each input file
    for input_file in input_files:
        try:
            with open(input_file, 'r') as file:
                data = json.load(file)

                # Convert the data to lowercase and merge into the result dictionary
                merged_data.update(convert_to_lowercase(data))
        except FileNotFoundError:
            print(f"File not found: {input_file}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {input_file}")
            print(f"Details: {str(e)}")

    # Write the merged data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(merged_data, outfile, indent=2)

merge_json_files('merged_output.json', 'Preventions.json', 'Living-with-HIV.json', 'HIV-Symptoms.json', 'HIV-chatbot-data.json', 'Causes_of_HIV.json')
