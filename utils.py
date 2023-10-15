import os
import shutil
import random

source_dir = "../data/cats"
dest_dirs = ["cats_stage_0", "cats_stage_1", "cats_stage_2"]

for dir_name in dest_dirs:
    os.makedirs(dir_name, exist_ok=True)

jpg_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

random.shuffle(jpg_files)

files_per_dir = len(jpg_files) // len(dest_dirs)

for i, dir_name in enumerate(dest_dirs):
    start = i * files_per_dir
    end = (i + 1) * files_per_dir if i < len(dest_dirs) - 1 else len(jpg_files)
    files_to_copy = jpg_files[start:end]
    
    for file in files_to_copy:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(f"../data/{dir_name}", file)
        shutil.copy(src_path, dest_path)
