file_path = 'new_file.txt'

# 打开文件并创建新文件
with open(file_path, 'w') as f:
    # 可以在文件中写入内容
    f.write('This is a new file.')