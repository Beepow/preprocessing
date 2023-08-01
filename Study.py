import os

file_path = 'Z:/1080_for_cls/train/images'
file_names = os.listdir(file_path)
print(len(file_names))
# print(file_names[1][0:5])
# k = 0
# for k in range(0,len(file_names)-1):
#     print(file_names[k][0:5])

# i = 1
# for name in file_names:
#     src = os.path.join(file_path, name)
#     dst = str(i) + '.jpg'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1
k = 0
t=0
i = 1
for k in range(0,len(file_names)-1):
    # if file_names[k][0] == '-':
    #     # print(file_names[k][0:5])
    #     fn = file_names[k]
    #     t = t +1
    #     # print(fn)
    #     src = os.path.join(file_path, fn)
    #     print(src)
    #     dst = fn[0:6] +str(i) + '.jpg'
    #     print(dst)
    #     dst = os.path.join(file_path,dst)
    #     os.rename(src, dst)
    #     i += 1
    # #
    # if file_names[k][0] == 'c':
    #     fn = file_names[k]
    #     t = t+1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[5:10] + str(i) + '.jpg'
    #     print(dst)
    #     dst = os.path.join(file_path,dst)
    #     os.rename(src,dst)
    #     i += 1
    #
    # if file_names[k][0] == '1':
    #     fn = file_names[k]
    #
    #     t = t + 1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[0:7] + '_' +str(i) + '.jpg'
    #     print(dst)
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)
    #     i += 1

    # if file_names[k][0] == '2':
    #     fn = file_names[k]
    #     t = t + 1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[0:5] + '_' +str(i) + '.jpg'
    #     print(dst)
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)
    #     i += 1
    #
    # if file_names[k][0] == '3':
    #     fn = file_names[k]
    #     t = t + 1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[0:6] + '_' + str(i) + '.jpg'
    #     print(dst)
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)
    #     i += 1
    #
    # if file_names[k][0] == '-':
    #     fn = file_names[k]
    #     t = t + 1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[1:]
    #     print(dst)
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)


    # if file_names[k][0] == '_':
    #     fn = file_names[k]
    #     t = t + 1
    #     src = os.path.join(file_path, fn)
    #     dst = fn[1:]
    #     print(dst)
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)
    # to
    if file_names[k][0] == 'R':
        fn = file_names[k]
        t = t + 1
        src = os.path.join(file_path, fn)
        dst = fn[0:4] + '_' + str(i) + '.jpg'
        print(dst)
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        i +=1
print(t)
d=1