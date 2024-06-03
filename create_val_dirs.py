import os
import shutil

def create_directories_and_move_files(input_path, source_dir):
    with open(input_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        label, filename = line.strip().split('/')
        label_dir = os.path.join(source_dir, label)
        
        # Create the directory if it does not exist
        os.makedirs(label_dir, exist_ok=True)
        
        # Define source and destination file paths
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(label_dir, filename)
        
        # Move the file
        if os.path.exists(src_file):
            shutil.move(src_file, dest_file)
        else:
            print(f"File {src_file} not found.")

# Define the paths
input_filename = "./imagenet_data/data_val/filelist.txt"  # Path to the input text file
source_directory = './imagenet_data/data_val/data'  # Directory containing the original JPEG files

# Call the function to create directories and move files
create_directories_and_move_files(input_filename, source_directory)
