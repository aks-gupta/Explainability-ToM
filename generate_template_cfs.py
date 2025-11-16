import json
import os

dataset = 'sycophancy'

# Read the template file
with open(f'templates/{dataset}_fixed_qs.json', 'r') as f:
    template_data = json.load(f)

# Initialize the output structure
output = []

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
        for background in backgrounds[1:]:
            counterfactual = template.replace("[a]", name).replace("[b]", background)
            counterfactuals.append(counterfactual)
        
        # Add to output structure
        output.append({
            "question": main_question,
            "counterfactuals": counterfactuals,
            "template_id": template_id
        })

# Write the output to a file
output_file = f'./templates/counterfactuals_output_{dataset}.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)

print(f"Output written to {output_file}")
print(f"Generated {len(output)} question sets with counterfactuals")
