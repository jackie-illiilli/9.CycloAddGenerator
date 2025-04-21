from Code import model_train, logfile_process, xtb_process, cycle_process
from rdkit import Chem
import numpy as np

mol = Chem.MolFromMolFile('Figure/SVO/smilesid_05944.mol', removeHs=False)
log_file = "Figure/SVO/smilesid_05944_0000.log"
log = logfile_process.Logfile(log_file)
symbol_list, position = log.symbol_list, log.running_positions[-1]
mol = xtb_process.xtb_to_mol(mol, [symbol_list], [position], 1)

for new_mol, atom_lists in cycle_process.change_position(mol, prop='diene', return_tran_cis=False):
    # if atom_lists[0] == 6 and atom_lists[-1] == 7:
    if atom_lists[0] == 0 and atom_lists[-1] == 3:
        print(model_train.Calc_areas(new_mol, atom_lists))
        break   
atoms_ids = atom_lists
mol = new_mol
table = Chem.rdchem.GetPeriodicTable()
VAN_DER_WAALS_RADII = {each:table.GetRvdw(each) for each in ["H", "B", "C", "N", "O", "F", "S", "Cl", 'Br']}
mol_conformers = mol.GetConformers()
assert len(mol_conformers) == 1
symbol_lists = [atom.GetSymbol() for atom in mol.GetAtoms()]
geom = mol_conformers[0].GetPositions()

# 正方体边长和总点数
num = 20
radius = 3
cube_length = 2 * radius
total_points = num * num * num 
counts = np.zeros(8, dtype=np.int32)

# 生成均匀的网格点
x = np.linspace(0.1 -radius, radius - 0.1, num)
y = x; z = np.linspace(0.1 -radius, radius - 0.1, num)
# 调节格点均匀度
# z = (-0.5 * (np.abs(z) - 2) ** 2  + 2 ) * z / np.abs(z)
# 生成点
points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
points_inside = np.zeros(total_points, dtype=bool)
# 计算每个点到每个原子中心的距离
for atom_id, (symbol, sphere_center) in enumerate(zip(symbol_lists, geom)):
    if atom_id in [atoms_ids[0], atoms_ids[-1]]:
        continue
    radii = VAN_DER_WAALS_RADII[symbol]
    translated_points = points - sphere_center
    distances = np.linalg.norm(translated_points, axis=1)
    points_inside = points_inside | (distances <= radii)
points_inside_z = points_inside

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# After computing points_inside
inside_points = points[points_inside]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点
ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], 
           c='g', marker='.', s=2)

# 定义分割点
points_ = [-radius, 0, radius]

# 绘制x方向的边缘线
for y in points_:
    for z in points_:
        ax.plot([-radius, radius], [y, y], [z, z], 'k--', c='gray')

# 绘制y方向的边缘线
for x in points_:
    for z in points_:
        ax.plot([x, x], [-radius, radius], [z, z], 'k--', c='gray')

# 绘制z方向的边缘线
for x in points_:
    for y in points_:
        ax.plot([x, x], [y, y], [-radius, radius], 'k--', c='gray')

# 设置图形范围
ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.set_zlim(-radius, radius)

# 关闭坐标轴
ax.set_axis_off()

# 显示图形
plt.show()