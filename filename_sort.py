import os
import shutil

# 원본 디렉토리와 대상 디렉토리 설정
# class_name = 'truck'

orig = 'C:/Users/최재원/Desktop/Sequence_Dataset/Sequence_Dataset/train/Train' # + class_name
targ1 = 'C:/Users/최재원/Desktop/Sequence_Dataset/Sequence_Dataset/train/sort1' #+ class_name
targ2 = 'C:/Users/최재원/Desktop/Sequence_Dataset/Sequence_Dataset/train/sort2' #+ class_name

# num_path = 'C:/Users/최재원/Desktop/Train_normal/' + class_name

file_num_list1 = []
file_num_list2 = []
for file_name in os.listdir(orig):
    file_num = file_name[5:-4]
    if int(file_num) % 2 == 0:
        file_num_list1.append(file_name)
        shutil.move(os.path.join(orig, file_name), os.path.join(targ1, file_name))
    else:
        file_num_list2.append(file_name)
        shutil.move(os.path.join(orig, file_name), os.path.join(targ2, file_name))
print(file_num_list1)
print(len(file_num_list1))

# # 숫자로 된 폴더를 orig에서 targ로 이동
# for folder_name in file_num_list1:
#     orig_folder_path = os.path.join(orig, class_name + '_' + str(folder_name))
#     targ_folder_path = os.path.join(targ, class_name + '_' + str(folder_name))
#
#     # orig에 해당 폴더가 있고, targ에 해당 폴더가 없다면 폴더 이동
#     if os.path.exists(orig_folder_path) and not os.path.exists(targ_folder_path):
#         shutil.move(orig_folder_path, targ)
