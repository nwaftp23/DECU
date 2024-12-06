import os
import shutil

def create_directories_and_move_files(input_path, source_dir):
    with open(input_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        # label, filename = line.strip().split('/')
        filename, label = line.strip().split(' ')
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
# input_filename = "./imagenet_data/data_val/filelist.txt"  # Path to the input text file
# source_directory = './imagenet_data/data_val/data'  # Directory containing the original JPEG files
# input_filename = "/nvmestore/mjazbec/ImageNet/val/image_2_wnid.txt"
# source_directory = '/nvmestore/mjazbec/ImageNet/val'

# input_filename = "/nvmestore/mjazbec/imagenet_data/data_val/filelist.txt"  # Path to the input text file
input_filename = "/nvmestore/mjazbec/ImageNet/val/image_2_wnid.txt"
source_directory = '/nvmestore/mjazbec/imagenet_data/data_val/data'  # Directory containing the original JPEG files


# Call the function to create directories and move files
# create_directories_and_move_files(input_filename, source_directory)

# # get a list of all the files in the source directory
# files = os.listdir(source_directory)
# print(f"Number of files in the source directory: {len(files)}")


# # root_dir = '/nvmestore/mjazbec/imagenet_data/data_train/data'
# root_dir = '/nvmestore/mjazbec/ImageNet/data_train/data'

# # Define the output file path
# # output_file = '/nvmestore/mjazbec/imagenet_data/file_list.txt'
# output_file = '/nvmestore/mjazbec/ImageNet/data_train/filelist_comp0_binned_classes.txt'

# # Collect all JPEG files along with their parent directories
# file_list = []
# for parent_dir, _, files in os.walk(root_dir):
#     for file in files:
#         if file.endswith('.JPEG'):
#             # Construct relative path: parent_folder/file_name
#             relative_path = os.path.relpath(parent_dir, root_dir)
#             file_path = os.path.join(relative_path, file)
#             file_list.append(file_path)

# # Create the output directory if it doesn't exist
# output_dir = os.path.dirname(output_file)
# os.makedirs(output_dir, exist_ok=True)

# # Write the collected file paths to a .txt file
# with open(output_file, 'w') as f:
#     for line in file_list:
#         f.write(line + '\n')

# read in pickle file

import pickle

with open('/nvmestore/mjazbec/ImageNet/imagenet_val.pkl', 'rb') as f:
    data = pickle.load(f)
