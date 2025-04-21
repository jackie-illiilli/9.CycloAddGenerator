import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

deltaGa_limit = 22
new_data_csv = pd.read_csv(f"../Data/Predict_DI/deltaGa_LowThan_{deltaGa_limit}_stable.csv")
banned_bo_diene = []
banned_bo_ene = [17935, 13787, 16502, 13758, 17291]
new_data_csv = new_data_csv.loc[~(np.isin(new_data_csv['Diene_Index'], banned_bo_diene) | np.isin(new_data_csv['Ene_Index'], banned_bo_ene))]
BO_data_csv = pd.read_csv(rf"G:\work\Secondary_Selection\All_possible_DA_ZINC_BO\Result_BO_One_Low_22.csv")
BO_data_csv = pd.read_csv(rf"G:\work\Secondary_Selection\All_possible_DA_ZINC_BO\all_reported_BO_One.csv")
BO_data_csv = BO_data_csv.loc[BO_data_csv['deltaGa(Solvent)_p'] < 26]
# 随机生成两批三维点阵数据
np.random.seed(0)  # 固定随机数种子以便结果可重复
num_points = 100

# 第一批三维点阵
x1 = new_data_csv['Diene_Distort_p'].to_numpy()
y1 = new_data_csv['Ene_Distort_p'].to_numpy()
z1 = new_data_csv['Interaction_p'].to_numpy()

# 第二批三维点阵
x2 = BO_data_csv['Diene_Distort_p'].to_numpy()
y2 = BO_data_csv['Ene_Distort_p'].to_numpy()
z2 = BO_data_csv['Interaction_p'].to_numpy()

# 创建三维图形对象
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一批点阵
ax.scatter(x1, y1, z1, c='g', marker='o', label='High Reactive', alpha=0.3)

# 绘制第二批点阵
ax.scatter(x2, y2, z2, c='r', marker='*', label='Bioorthogonal')

# 添加标签
ax.set_xlabel('4π Dist.')
ax.set_ylabel('2π Dist.')
ax.set_zlabel('Int.')
ax.legend()

# 显示图形
plt.show()

