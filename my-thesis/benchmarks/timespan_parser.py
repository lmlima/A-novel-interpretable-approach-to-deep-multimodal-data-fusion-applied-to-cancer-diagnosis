"""
Author: Leandro Lima
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import re

# Check if the command-line argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <base_dir>")
    sys.exit(1)

# Get the base directory from the command-line argument
base_dir = sys.argv[1]

# Initialize an empty list to store the results
results = []

# Define a regular expression pattern to match the expected directory structure
pattern = re.compile(r'(.+)_fold_(\d+)_(\d+)')

# Loop over the subdirectories in the base directory
for subdir in os.listdir(base_dir):
    experiment_dir = os.path.join(base_dir, subdir)

    # Check if it's a directory
    if os.path.isdir(experiment_dir):
        # Use the regular expression to extract experiment name and fold
        match = pattern.match(subdir)

        if match:
            experiment_name, fold, timestamp = match.groups()
            fold = int(fold)

            # Construct the JSON file path for this experiment
            json_file_path = os.path.join(experiment_dir, '1/run.json')

            # Load the JSON data from the file
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract the datetime strings
            heartbeat_str = data['heartbeat']
            start_time_str = data['start_time']

            # Parse the datetime strings into datetime objects
            heartbeat_datetime = datetime.fromisoformat(heartbeat_str)
            start_time_datetime = datetime.fromisoformat(start_time_str)

            # Calculate the time span
            time_span = heartbeat_datetime - start_time_datetime

            # Append the result to the list
            results.append({'experiment_name': experiment_name, 'fold': fold, 'time_span': time_span.total_seconds()})

# Create a Pandas DataFrame from the results
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('experiment_times.csv', index=False)

# Save the DataFrame to a Pickle (pkl) file
df.to_pickle('experiment_times.pkl')

df_desc = df.groupby(["experiment_name"]).describe()
df_desc.to_csv("timespan_desc.csv")
