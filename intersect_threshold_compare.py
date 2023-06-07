import subprocess
import re
import csv

# Define the command template
command_template = "./bin/approx_profile inputs/mico/graph 1 1024 1 1 {} 50"

# Initialize the data list
data_list = []

# Loop through <i> values from 3 to 100
for i in range(50, 500,5):
    # Run the command
    command = command_template.format(i)
    output = subprocess.check_output(command, shell=True, text=True)

    # Extract the required values using regular expressions
    sparse_match = re.search(r"sparse match: (\d+)", output)
    dense_match = re.search(r"dense match: (\d+)", output)
    mismatch = re.search(r"mismatch: (\d+)", output)
    num_vertices_match = re.search(r"num vertices match = (\d+)", output)

    # Create a data row with the extracted values
    data_row = [i, sparse_match.group(1), dense_match.group(1), mismatch.group(1), num_vertices_match.group(1)]
    data_list.append(data_row)

# Write the data to a CSV file
filename = "output.csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["i", "Sparse Match", "Dense Match", "Mismatch", "Num Vertices Match"])
    writer.writerows(data_list)

print(f"Data saved to {filename}.")