import os

def rename_images(folder_path, new_folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # for file in files:
    #     print(file)
    
    # 排序
    files.sort()
    
    for file in files:
        print(file)

    # 遍历文件夹中的每个文件
    for idx, file in enumerate(files):
        # 检查文件是否为图片文件
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            # 构建新的文件名
            new_file_name = f'main_new_{idx:0>5d}.png'

            # 构建原文件的完整路径
            old_file_path = os.path.join(folder_path, file)

            # 构建新文件的完整路径
            new_file_path = os.path.join(new_folder_path, new_file_name)

            # 复制文件到新路径
            with open(old_file_path, "rb") as old_file, open(new_file_path, "wb") as new_file:
                new_file.write(old_file.read())

            print(f"重命名文件: {file} -> {new_file_name}")
            
            
# Specify the source and destination folders
source_folder = "F:\\jianyin\\videos\\outdoors\\20240410autel_bicycle\\autel_bicycle20240410\\images"
destination_folder = "F:\\jianyin\\videos\\outdoors\\20240410autel_bicycle\\autel_bicycle20240410\\images_new"

os.makedirs(destination_folder, exist_ok=True)

# 调用函数进行重命名
rename_images(source_folder, destination_folder)