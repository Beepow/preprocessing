import os
import random
import shutil

def move_random_folders(source_folder, destination_folder, num_folders_to_move):

    folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    random_folders = random.sample(folders, min(num_folders_to_move, len(folders)))

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for folder_name in random_folders:
        source_path = os.path.join(source_folder, folder_name)
        destination_path = os.path.join(destination_folder, folder_name)
        shutil.move(source_path, destination_path)


source_folder = 'C:/Users/최재원/Desktop/TBR_easy/truck'  # 원본 폴더 경로
destination_folder = 'C:/Users/최재원/Desktop/p1_train/truck'  # 대상 폴더 경로
num_folders_to_move = 95  # 이동할 폴더의 수

move_random_folders(source_folder, destination_folder, num_folders_to_move)
