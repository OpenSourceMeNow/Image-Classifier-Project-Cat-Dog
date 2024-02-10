import os
import shutil

# Path to the main 'train' directory
train_dir = '/mnt/e/Cats&Dogs/train/train'

# Subdirectories for cat and dog images
cat_dir = os.path.join(train_dir, 'cats')
dog_dir = os.path.join(train_dir, 'dogs')

# Create subdirectories
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# Loop through all files in the 'train' directory
for filename in os.listdir(train_dir):
    file_path = os.path.join(train_dir, filename)
    # Skip directories
    if os.path.isdir(file_path):
        continue
    if filename.startswith('cat'):
        shutil.move(file_path, os.path.join(cat_dir, filename))
    elif filename.startswith('dog'):
        shutil.move(file_path, os.path.join(dog_dir, filename))
