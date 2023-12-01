import csv
import json

# Open the CSV file and read its contents
with open('input.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header row
    next(reader)
    # Create an empty dictionary to store the data
    data = {}
    # Iterate over each row in the CSV file
    for row in reader:
        # Add the question and answer to the dictionary
        data[row[0]] = row[1]

# Open a new file to write the JSON data
with open('output.json', 'w') as jsonfile:
    # Write the JSON data to the file
    json.dump(data, jsonfile)