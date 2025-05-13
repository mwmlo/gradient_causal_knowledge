import json
import pandas as pd
import requests

# Download CounterFact dataset as JSON
response = requests.get("https://rome.baulab.info/data/dsets/counterfact.json")
data = response.json()

df = pd.json_normalize(data)

# Step 3: Save as CSV
output_file = "datasets/counterfact.csv"
df.to_csv(output_file, index=False)

print(f"Successfully converted JSON to CSV: {output_file}")
