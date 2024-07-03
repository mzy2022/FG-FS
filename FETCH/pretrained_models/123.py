import pickle

# 自定义的persistent_load函数
def persistent_load(persid):
    # 对象类型判断和处理逻辑
    # 例如：
    if persid == b'my_custom_class':
        return MyCustomClass()

# 打开PKL文件，并指定persistent_load函数
with open('enc_c_param_mix.pkl', 'rb') as f:
    data = pickle.load(f,persistent_load=persistent_load)

# 使用读取的数据
print(data)
