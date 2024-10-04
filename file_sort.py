import os
import random
import shutil

def move_random_folders(source_folder, destination_folder, num_folders_to_move):
    # 소스 폴더에서 폴더 목록 가져오기
    folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # 랜덤하게 폴더를 선택
    random_folders = random.sample(folders, min(num_folders_to_move, len(folders)))

    # 대상 폴더가 없으면 생성
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 랜덤하게 선택한 폴더를 대상 폴더로 이동
    for folder_name in random_folders:
        source_path = os.path.join(source_folder, folder_name)
        destination_path = os.path.join(destination_folder, folder_name)
        shutil.move(source_path, destination_path)
        print(f"{folder_name} 폴더를 {destination_folder}로 이동했습니다.")

# 사용 예시
source_folder = 'C:/Users/최재원/Desktop/TBR_easy/truck'  # 원본 폴더 경로를 적절하게 변경하세요.
destination_folder = 'C:/Users/최재원/Desktop/p1_train/truck'  # 대상 폴더 경로를 적절하게 변경하세요.
num_folders_to_move = 95  # 이동할 폴더의 수를 적절하게 변경하세요.

move_random_folders(source_folder, destination_folder, num_folders_to_move)
