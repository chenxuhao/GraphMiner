import re
import subprocess

thresholds = range(1, 30)

for threshold in thresholds:
    command = f"./bin/approx_profile inputs/mico/graph 1 1024 1 0 {threshold}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output_line = result.stdout.strip()

    match = re.search(r"Sparse \|V\|: (\d+) Dense \|V\|: (\d+)", output_line)
    if match:
        sparse = int(match.group(1))
        dense = int(match.group(2))
        ratio = sparse / (sparse + dense)
        print(f"{ratio}")