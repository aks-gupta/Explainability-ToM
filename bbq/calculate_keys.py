import json

# Load the JSON file
with open('data_bbq.json') as f:
    data = json.load(f)

# Count items under the 'fruits' key
num_items = len(data['genderIdentity'])

print(f"Number of items under 'genderIdentity': {num_items}")