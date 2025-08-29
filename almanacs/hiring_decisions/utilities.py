import os

def list_files_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)
        print(f"Files in '{folder_path}':")
        max_version = 0
        for file_name in files:
            print(file_name)
            version = int(file_name[1:])
            max_version = max(version, max_version)
        return max_version
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{folder_path}'.")

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

def create_folder_based_on_version():
    folder_path = "almanacs/versioned_output"  # Change this to your folder path
    last_version = list_files_in_folder(folder_path)
    file_name = f'v{last_version+1}'
    create_folder(folder_path, file_name)
    return os.path.join(folder_path, file_name)

def return_last_max_version():
    folder_path = "almanacs/versioned_output"  # Change this to your folder path
    last_version = list_files_in_folder(folder_path)
    print(f"Calculating precision for v{last_version}")
    file_name = f'v{last_version}' #switch to a targeted version, else= "f'v4'"
    return os.path.join(folder_path, file_name)