'''
Utility script to combine results from multiple JSON files into a single JSON file.
Takes in a folder name, extracts subfolders and precision results from each subfolder,
and combines them into a single JSON file for easier analysis.
'''
import os
import json

def combine_results(folder_name):
    combined_results = {}
    
    # List all subfolders in the given folder
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    
    for subfolder in subfolders:
        precision_file = os.path.join(subfolder, 'precision_results.json')
        if os.path.exists(precision_file):
            with open(precision_file, 'r') as f:
                precision_data = json.load(f)
                if "precision_results" in precision_data:
                    precision_data = precision_data["precision_results"]
                combined_results[os.path.basename(subfolder)] = precision_data
        else:
            print(f"Warning: {precision_file} does not exist and will be skipped.")
    
    # Write the combined results to the output file
    subfolder_name = os.path.basename(os.path.normpath(folder_name))

    with open(f"{folder_name}/{subfolder_name}.json", 'w') as f:
        json.dump(combined_results, f, indent=4)

    print(f"Combined results saved to {folder_name}/{subfolder_name}.json")

if __name__ == "__main__":
    folder_name = "../outputs/sycophancy_meta-llama_200"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    combine_results(folder_path)
