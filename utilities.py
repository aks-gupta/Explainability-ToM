import os
import pickle
import json
import configs


def list_files_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)
        max_version = 0
        for file_name in files:
            version = int(file_name[1:])
            max_version = max(version, max_version)
        return max_version
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")
    
def preprocess_label_balanced_counterfactuals(path_to_cfs):
	#Generate file for task model questions
	with open(path_to_cfs, "r") as f:
		data = json.load(f)

	# === Step 2: Prepare outputs ===
	original_questions = {}
	counterfactual_questions = {}
	original_questions = [{"question": value["question"]} for key, value in data.items()]
     
	for seq_key, (key, value) in enumerate(data.items()):
		seq_key = int(seq_key)  # ensure integer keys
		# Extract only the counterfactual questions
		counterfactual_questions[seq_key] = {"questions": value["counterfactual_questions"]}

	# === Step 3: Save both new PKL files ===
	with open(f"./data/preprocessed/label_balanced_original_questions_{configs.DOMAIN}.json", "w") as f:
		json.dump(original_questions, f)

	with open(f"./data/preprocessed/label_balanced_counterfactuals_{configs.DOMAIN}.pkl", "wb") as f:
		pickle.dump(counterfactual_questions, f)
          

def create_folder(parent_directory, new_folder_name):
    # Construct the full path
    full_path = os.path.join(parent_directory, new_folder_name)
    try:
        os.makedirs(full_path, exist_ok=False)
        print(f"Folder created at: {full_path}")
    except FileExistsError:
        print(f"Folder already exists: {full_path}")
    except PermissionError:
        print(f"Permission denied to create folder at: {full_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_folder_based_on_version(folder_path="outputs/versioned_output"):
    last_version = list_files_in_folder(folder_path)
    if last_version is None or last_version == 0:
        file_name = 'v1'
    else:
        if configs.GENERAL_CONFIGS['use_existing_folder']:
            return os.path.join(folder_path, f'v{last_version}')
        file_name = f'v{last_version+1}'
    create_folder(folder_path, file_name)
    return os.path.join(folder_path, file_name)

def return_last_max_version(folder_path="outputs/versioned_output"):
    last_version = list_files_in_folder(folder_path)
    file_name = f'v{last_version}' #switch to a targeted version, else= "f'v17'"
    return os.path.join(folder_path, file_name)