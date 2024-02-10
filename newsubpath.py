import os
import shutil

# Define your current test directory and the new subdirectory
current_test_dir = '/mnt/e/Cats&Dogs/test1'
new_subdir = 'images'
new_subdir_path = os.path.join(current_test_dir, new_subdir)

# Create the new subdirectory if it does not exist
if not os.path.exists(new_subdir_path):
    os.makedirs(new_subdir_path)

# Loop through all files in the current test directory
for filename in os.listdir(current_test_dir):
    # Construct the full file path
    file_path = os.path.join(current_test_dir, filename)
    # Check if it's a file (to skip directories)
    if os.path.isfile(file_path):
        # Move the file to the new subdirectory
        shutil.move(file_path, new_subdir_path)

print("All images have been moved to:", new_subdir_path)