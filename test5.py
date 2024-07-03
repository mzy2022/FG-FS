# import pandas as pd
# import shap
# import xgboost
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# from sklearn.datasets import load_iris

# 加载数据集
# iris = load_iris()
# X, y = iris.data, iris.target
# data = pd.read_csv("test.csv")
# data = data.apply(np.nan_to_num)
# X = data.iloc[:,0:10]
# y = data.iloc[:,80:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 初始化随机森林分类器
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#
# # 在训练集上训练模型
# rf_classifier.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = rf_classifier.predict(X_test)
#
# # 计算分类准确性
# accuracy = accuracy_score(y_test, y_pred)
# print("分类准确性：", accuracy)
import numpy as np
from sklearn.model_selection import train_test_split

# 初始化
def initialize_population(pop_size, tensor_shape):
    return [np.random.randn(*tensor_shape) for _ in range(pop_size)]

# 评价函数（以分类任务为例）
def evaluate_fitness(tensor, data, labels):
    tensor = np.array(tensor)
    encoder_output = data.dot(tensor)
    # 使用简单的分类器，比如线性分类器
    predictions = encoder_output.dot(np.random.randn(tensor.shape[1], len(np.unique(labels))))
    accuracy = np.mean(np.argmax(predictions, axis=1) == labels)
    return accuracy

# 选择
def selection(population, fitness_scores, num_parents):
    parents = np.array(population)[np.argsort(fitness_scores)[-num_parents:]]
    return parents.tolist()

# 交叉
def crossover(parents, num_offspring):
    offspring = []
    for _ in range(num_offspring):
        parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
        crossover_point = np.random.randint(0, len(parents[0]))
        parents[parent1] = np.array(parents[parent1])
        parents[parent2] = np.array(parents[parent2])
        parents[0] = np.array(parents[0])
        child = np.concatenate((parents[parent1].flat[:crossover_point], parents[parent2].flat[crossover_point:])).reshape(parents[0].shape)
        offspring.append(child)
    return offspring

# 变异
def mutation(offspring, mutation_rate):
    for child in offspring:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, child.size)
            child.flat[mutation_point] += np.random.randn()
    return offspring

# 主进化算法
def evolve_encoder(data, labels, tensor_shape, pop_size=200, num_generations=1000, mutation_rate=0.1):
    population = initialize_population(pop_size, tensor_shape)
    for generation in range(num_generations):
        fitness_scores = [evaluate_fitness(ind, data, labels) for ind in population]
        parents = selection(population, fitness_scores, pop_size // 2)
        offspring = crossover(parents, pop_size - len(parents))
        offspring = mutation(offspring, mutation_rate)
        population = parents + offspring
        best_fitness = max(fitness_scores)
        print(f'Generation {generation}: Best Fitness = {best_fitness}')
    print(np.max(fitness_scores))
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

# 示例调用
data, labels = np.random.randn(1000, 128), np.random.randint(0, 10, 1000)
tensor_shape = (128, 64)
best_encoder = evolve_encoder(data, labels, tensor_shape)
