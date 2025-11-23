import json
import os

# Read the template file
with open('templates/sycophancy_fixed_qs.json', 'r') as f:
    template_data = json.load(f)

# Initialize the output structure as a dictionary with numeric keys
output = {}
counter = 0

# For each template in the template file
for template_item in template_data["questions"]:
    template = template_item["template"]
    template_id = template_item["template_id"]
    
    # Get the possible values for variables
    names = template_item["variables"]["possible_values"]["a"]
    backgrounds = template_item["variables"]["possible_values"]["b"]
    
    # For each name, create a question and its counterfactuals
    for name in names:
        # The main question uses the first background
        main_background = backgrounds[0]
        main_question = template.replace("[a]", name).replace("[b]", main_background)
        
        # Create counterfactuals using the other backgrounds
        counterfactuals = []
