import os

def get_input_files(folder_path):
    input_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return input_files