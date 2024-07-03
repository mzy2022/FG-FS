import matplotlib.pyplot as plt

with open("list.txt", "r") as file:
    # 读取文件内容
    content = file.read()

    # 使用eval()函数将字符串转换为列表
    my_list = eval(content)

test_list = []
for num,i in enumerate(my_list):
    test_list.append( (i - 0.8145574057029166) / 0.8145574057029166)
x = range(len(test_list))

# 绘制折线图
plt.plot(x,test_list)

plt.show()

