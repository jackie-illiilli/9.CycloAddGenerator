from . import cycle_process, xtb_process, logfile_process, Tool

import copy, pickle, glob, os
import numpy as np
import pandas as pd
from rdkit import Chem

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import BaseCrossValidator

import seaborn as sns
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, Trials, STATUS_OK, tpe
from tqdm import tqdm

ALL_PROPERTIES_diene = ["diene_charge_a", "diene_charge_b", "diene_charge_c", "diene_charge_d",
    "diene_max_neighbor_charge", "diene_min_neighbor_charge", "diene_max_charge", "diene_min_charge", "bond_1", "bond2", "bond3", "distance_ad",
    "angle_a_cos", "angle_b_cos", "diene_max_angle_cos", "diene_min_angle_cos", "torsion",
    "diene_homo-1", "diene_homo", "diene_lumo", "diene_lumo+1", "diene_dipole", 
    "max_diene_L", "min_diene_L", "max_diene_B1", "min_diene_B1", "max_diene_B5", "min_diene_B5"]
ALL_PROPERTIES_ene = ["ene_charge_e", "ene_charge_f", "ene_max_neighbor_charge", "ene_min_neighbor_charge", 
    "diene_max_charge", "diene_min_charge", "bond4", 
    "ene_max_angle_cos", "ene_min_angle_cos", "ene_homo-1", "ene_homo", "ene_lumo", "ene_lumo+1", 
    "ene_dipole", "max_ene_L", "min_ene_L", "max_ene_B1", "min_ene_B1", "max_ene_B5", "min_ene_B5"]
New_des = ['diene_area_0', 'diene_area_1', 'diene_area_2', 'diene_area_3', 'diene_area_4', 'diene_area_5', 'diene_area_6', 'diene_area_7', 
'ene_area_0', 'ene_area_1', 'ene_area_2', 'ene_area_3', 'ene_area_4', 'ene_area_5', 'ene_area_6', 'ene_area_7', ]


# def calc_rdkit_mf(smiles):
#     return Tool.clean_nan(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)))

def descriptor_generator(dfs, des_map=None, mf_des=True, with_product=False, y_name=None, is_3d_map=False):
    input_array = []
    result_array = []
    idxs = []
    for df in dfs:
        for each in tqdm(range(len(df))):
            temp_list = []
            diene_smiles = df.iloc[each]["Diene"]
            ene_smiles = df.iloc[each]["Ene"]
            diene_id = df.iloc[each]["Diene_Index"]
            ene_id = df.iloc[each]["Ene_Index"]
            title = df.iloc[each]["Title"]
            if y_name:
                delta_G = df.iloc[each][y_name]
                result_array.append(delta_G)
            if mf_des:
                assert des_map != None
                diene = list(des_map[diene_smiles])
                dieno = list(des_map[ene_smiles])
                temp_list += diene
                temp_list += dieno
            if with_product:
                product_smiles = df.iloc[each]["Product"]
                if is_3d_map:
                    _title = [int(each) for each in title.split()][:6]
                    product_smiles = "%s___%s" % (product_smiles, " ".join([str(each) for each in _title]))
                product = list(des_map[product_smiles])
                temp_list += product
            input_array.append(temp_list)
            idxs.append(each)
    return np.array(input_array), np.array(result_array), idxs

def read_df(df, QM_MAP):
    input_array = []
    idxs = []
    map_dir = QM_MAP
    with open(map_dir, "rb") as f:
        target_map = pickle.load(f)
    for each in tqdm(range(len(df))):
        temp_list = []
        diene_smiles = df.iloc[each]["Diene"]
        ene_smiles = df.iloc[each]["Ene"]
        diene_id = df.iloc[each]["Diene_Index"]
        ene_id = df.iloc[each]["Ene_Index"]
        title = df.iloc[each]["Title"]
        diene_atoma, diene_atomb, ene_atomc, ene_atomd, _, title_1 = [int(each) for each in title.split(" ")[:6]]
        diene_key = "%.5d %d %d %d" % (diene_id, 1, diene_atoma, diene_atomb)
        ene_key = "%.5d %d %d %d" % (ene_id, 0, ene_atomc, ene_atomd)
        if diene_key not in target_map.keys() or ene_key not in target_map.keys():
            continue
        diene_property = target_map[diene_key]
        ene_property = target_map[ene_key]
        diene_result = diene_property[-8:]
        result = ene_property[-8:]
        if title_1 == 0:
            result = [result[3], result[2], result[1], result[0], result[7], result[6], result[5], result[4]]
        temp_list += diene_property[:-8] + ene_property[:-8] + diene_result + result
        input_array.append(temp_list)
        idxs.append(each)
    return np.array(input_array), idxs

def normalize_axis(arr, axis=0, mean=[], std=[]):
    """
    对数组中的某一维进行标准化（z-score normalization）
    
    参数：
    arr: ndarray，输入的数组
    axis: int，标准化的维度
    
    返回值：
    normalized_arr: ndarray，标准化后的数组
    """
    if len(mean) == 0 or len(mean) == 0:
        mean = np.mean(arr, axis=axis, keepdims=True)  # 计算均值
        std = np.std(arr, axis=axis, keepdims=True)  # 计算标准差
    normalized_arr = (arr - mean) / std  # 标准化
    normalized_arr = np.nan_to_num(normalized_arr, 0)
    return normalized_arr, mean, std

def calc_distribution_line(ys, eachsize=0.1, title=None, xlab=None, ylab="Freq", return_result = False, labels = None, colors = None, useSVG=False, save_name='test', figure_size=(5,4), xlimit = []):
    fig = plt.figure(figsize=figure_size)
    y_max = np.max([max(each) for each in ys])
    y_min = np.min([min(each) for each in ys])
    # y_max = 50
    # y_min = 0
    if labels == None:
        labels = [None] * len(ys)
    if colors == None:
        colors = ["blue"] * len(ys)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    all_max = []
    for idx, y in enumerate(ys):
        des = [0 for each in X]
        z = (y - y_min)/eachsize
        for each in z:
            try:
                assert int(each) < len(X)
                des[int(each)] += 1
            except:
                continue
        des = np.array(des)
        # des = des / len(y)
        x = np.linspace(y_min - eachsize, y_max + eachsize * 2, 1000)
        model = make_interp_spline(X, des)
        ys = model(x)
        all_max.append(max(ys))
        print(x[np.argmax(ys)])
        plt.plot(x, ys, color=colors[idx])
        plt.fill_between(x, ys, 0, where=(ys > 0), interpolate=True, color=colors[idx], alpha=0.3, label=labels[idx])
    if xlimit != []:
        plt.xlim(xlimit[0], xlimit[1])
    else:
        plt.xlim(y_min - eachsize, y_max + eachsize)
    plt.ylim(0, 1.1 * max(all_max))
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if labels[0] != None:
        plt.legend()
    if title != None:
        plt.title = title
    if useSVG:
        plt.savefig(f"{save_name}.svg", bbox_inches='tight', format='svg')
    else:
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.show()  
    if return_result:
        return des

def half_ood_folds(react_data, n_folds=3, seed=0, diene_ood=True):
    np.random.seed(seed)
    all_diene_ids = np.unique(react_data["Diene_Index"].to_numpy())
    all_ene_ids = np.unique(react_data["Ene_Index"].to_numpy())
    diene_random_list ={id: int(each) % n_folds for id, each in zip(all_diene_ids, np.random.randn(len(all_diene_ids)) * n_folds * 10)}
    ene_random_list ={id: int(each) % n_folds for id, each in zip(all_ene_ids, np.random.randn(len(all_ene_ids)) * n_folds * 10)}
    folds = [[[], []] for _ in range(n_folds)]
    for id, diene_index in enumerate(react_data["Diene_Index"]):
        ene_index = react_data['Ene_Index'][id]
        if diene_ood:
            idx = diene_random_list[diene_index]
            folds[idx][1].append(id)
            [folds[each][0].append(id) for each in range(n_folds) if each != idx]
        else:
            idx = ene_random_list[ene_index]
            folds[idx][1].append(id)
            [folds[each][0].append(id) for each in range(n_folds) if each != idx]
    return folds

def ood_split(react_data, train_percent, seed=0):
    np.random.seed(seed)
    all_diene_ids = np.unique(react_data["Diene_Index"].to_numpy())
    all_ene_ids = np.unique(react_data["Ene_Index"].to_numpy())
    diene_random_list ={id: int(each) % 100 for id, each in zip(all_diene_ids, np.random.randn(len(all_diene_ids)) * 100)}
    ene_random_list ={id: int(each) % 100 for id, each in zip(all_ene_ids, np.random.randn(len(all_ene_ids)) * 100)}
    train_ids = []
    test_ids = []
    for id, diene_index in enumerate(react_data["Diene_Index"]):
        ene_index = react_data['Ene_Index'][id]
        if diene_random_list[diene_index] <= train_percent and ene_random_list[ene_index] <= train_percent:
            train_ids.append(id)
        if diene_random_list[diene_index] > train_percent and ene_random_list[ene_index] > train_percent:
            test_ids.append(id)
    return train_ids, test_ids

def special_k_fold(react_data, n, seed=0, special=True, removed_ids = []):
    np.random.seed(seed)
    x_size = len(react_data)
    id_lists = [[] for each in range(n)]
    result_lists = []
    random_list = [int(each) % n for each in np.random.randn(x_size) * n]
    for id, diene_index in enumerate(react_data["Diene_Index"]):
        if id in removed_ids:
            continue
        ene_index = react_data['Ene_Index'][id]
        if id != 0:
            last_diene_index = react_data['Diene_Index'][id - 1]
            last_ene_index = react_data['Ene_Index'][id - 1]
            if diene_index == last_diene_index and ene_index == last_ene_index and special:
                last_rand_id = random_list[id - 1]
                id_lists[last_rand_id].append(id)
                random_list[id] = last_rand_id
                continue
        now_rand_id = random_list[id]
        id_lists[now_rand_id].append(id)
    for each in range(n):
        train_list = list(range(n))
        train_list.remove(each)
        train_set = []
        for train in train_list:
            train_set += id_lists[train]
        test_set = id_lists[each]
        result_lists.append([train_set, test_set])
    return result_lists

class SpecialKFold(BaseCrossValidator):
    def __init__(self, data, n_splits=3, seed = 0, train_test_reverse=0):
        self.data = data
        self.n_splits = n_splits
        self.seed = seed
        self.train_test_reverse = train_test_reverse
    
    def get_n_splits(self):
        return self.n_splits
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        for each in special_k_fold(self.data, self.n_splits, self.seed):
            if self.train_test_reverse:
                yield each[0]
            else:
                yield each[1]

def hyper_opt_treemodel(_model, space, hyperparam, X, Y, data, train_test_reverse, n_fold, max_evals=20):
    def hyperopt_RF(param):
        cv = SpecialKFold(data, n_fold, train_test_reverse=train_test_reverse)
        model = _model(**param)
        acc = cross_val_score(model, X, Y, n_jobs=n_fold, cv=cv).mean()
        return {"loss":-acc, "status": STATUS_OK}
    trials = Trials()
    best = fmin(
        fn = hyperopt_RF,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    result_dict = {}
    for idx, name in enumerate(space.keys()):
        result_dict[name] = hyperparam[idx][best[name]]
    return result_dict

def benchmark(Xs, Y, data, models, spaces=None, hyperparams=None, epoch_time=3, reverse_train_test=0, n_fold=5, name='4pD', max_evals=20):
    def print_(strs = '', name='test'):
        print(strs)
        if not os.path.isfile(f'{name}.txt'):
            with open(f'{name}.txt', 'w') as f:
                f.write('')
        with open(f'{name}.txt', 'at') as f:
            f.write(strs)
            f.write('\n')
            
    from sklearn.model_selection import KFold, cross_val_score
    print_('start', name)
    sum_r2 = np.zeros((len(Xs), len(models)))
    sum_mae = np.zeros((len(Xs), len(models)))
    if spaces == None:
        spaces = [None] * len(models)
        hyperparams = [None] * len(models)
    for seed in range(epoch_time):
        print("Epoch : %d           " % seed)
        for X_id, X in enumerate(Xs):
            # X = normalize_axis(X)
            # kf = special_k_fold(data, n=n_fold, seed=seed, special=True)
            
            for model_id, (model_, model_space, hyperparam) in enumerate(zip(models, spaces, hyperparams)):
                if model_space != None:
                    best_param = hyper_opt_treemodel(model_, model_space, hyperparam, X, Y, data, reverse_train_test, n_fold, max_evals=max_evals)
                    print_(str(model_)+'   '+ str(best_param), name)
                    try:
                        model = model_(n_jobs=-1, **best_param)
                    except:
                        model = model_(**best_param)
                else:
                    try:
                        model = model_(n_jobs=-1, )
                    except:
                        model = model_()
                kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(X)
                for each in kf:
                    train_id, test_id = each
                    if reverse_train_test:
                        train_id, test_id = test_id, train_id
                    x_train, y_train = X[train_id], Y[train_id]
                    x_test, y_test = X[test_id], Y[test_id]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    sum_r2[X_id][model_id] += r2
                    sum_mae[X_id][model_id] += mae
                    print_(f"{r2}_{mae}", name)
                print_(f"{model_id}, {X_id}, {sum_r2[X_id][model_id] / n_fold}, {sum_mae[X_id][model_id] / n_fold}", name)
                        # print(r2, desmap_label[map_id], model_labels[model_id])
    sum_r2 = np.array(sum_r2)
    sum_r2 /= epoch_time * n_fold
    sum_mae = np.array(sum_mae)
    sum_mae /= epoch_time * n_fold
    print_(str(sum_r2), name)
    print_(str(sum_mae), name)
    return sum_r2

def draw_correlation_map(X, figure_size=(5, 5), colors='coolwarm', useSVG=False, save_name='test', annot=True, show_label=False):
    df = pd.DataFrame(X)
    correlation_matrix = np.abs(df.corr())
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    print(np.max(correlation_matrix.to_numpy()[~mask]))
    f, ax = plt.subplots(figsize=figure_size, dpi=300)
    annot_kws = {"fontsize": 30}
    ax = sns.heatmap(correlation_matrix, 
            mask=mask,
            cmap='coolwarm',    
            annot=annot,         
            fmt='.1f',          
            center=0,           
            cbar=1,
            annot_kws=annot_kws,)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    if not show_label:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    if useSVG:
        plt.savefig(f"{save_name}.svg", format="svg", bbox_inches='tight')
    else:
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')

def draw_heatmap(x_labels=[], y_labels=[], values=None, title=None, figure_size=(40, 6), colors='Blues'):
    # desc_labels = ["rdkit_mf", "morgan_mf", "rdkit_des", "modred_des"]
    # model_labels = ["GB", "XGB", "RF", "ET", "AdaB", "Line", "MLP"]
    # model_labels = ["no_product", "with_product", "with_structure"]

    plt.rcParams['font.sans-serif']='Arial'#设置中文显示，必须放在sns.set之后

    uniform_data = values #设置二维矩阵
    f, ax = plt.subplots(figsize=figure_size, dpi=300)
    annot_kws = {"fontsize": 30}
    #heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
    #参数annot=True表示在对应模块中注释值
    # 参数linewidths是控制网格间间隔
    #参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
    #参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
    min_value = np.min(values)
    max_value = np.max(values)
    # min_value = 1
    # max_value = 3
    sns.heatmap(uniform_data, ax=ax,vmin=min_value,vmax=max_value,cmap=colors,linewidths=2,cbar=1, annot=True,annot_kws=annot_kws, fmt='.3f')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    if title!= None: ax.set_title(title, fontsize=40) #plt.title('热图'),均可设置图片标题
    # ax.set_ylabel('descriptor', fontsize=10)  #设置纵轴标签
    # ax.set_xlabel('model', fontsize=10)  #设置横轴标签
    if x_labels!= None: 
        ax.set_xticklabels(x_labels, fontsize=30)
        label_x =  ax.get_xticklabels()
        plt.setp(label_x, rotation=0, horizontalalignment='center')
    if y_labels!= None: 
        ax.set_yticklabels(y_labels, fontsize=30)
        label_y =  ax.get_yticklabels()
        plt.setp(label_y, rotation=0, horizontalalignment='right')
    # #设置坐标字体方向，通过rotation参数可以调节旋转角度

    plt.savefig('test.png', dpi=300, bbox_inches = 'tight' )
    plt.show()
    return plt

def plot_scatter_with_metrics(x, y, title=None, min_=None, max_=None, alpha=1, savename='result'):
    """
    绘制散点图并显示回归性能指标
    
    参数：
    x: 一维数组类型，表示x轴数据。
    y: 一维数组类型，表示y轴数据。
    title: 字符串类型，表示图的标题。
    
    返回值：
    None
    
    """
    # 计算回归性能指标
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)

    # 绘制散点图
    plt.figure(figsize=(5, 5), dpi=300)
    if not min_ or not max_:
        min_ = np.min([x,y]) - 5
        max_ = np.max([x,y]) + 5
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if title != None:
        plt.title("%s\nR2:%.3f, MAE:%.3f, MSE:%.3f" % (title, r2, mae, mse), fontsize=24)
        plt.title("%s"%title,fontsize=24)
    z = np.linspace(min_, max_, 10)
    plt.plot(z, z, alpha=0.2)

    # plt.scatter(x, y, marker=".", c="g", alpha=alpha)
    sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_adjust=1, thresh=0.01)
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    # 添加回归性能指标到图像的第二行
    
    # 显示图像
    plt.savefig(f"{savename}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_plot_with_metrics(x, y, title=None, min_=None, max_=None):
    """
    绘制折线图并显示回归性能指标
    
    参数：
    x: 一维数组类型，表示x轴数据。
    y: 一维数组类型，表示y轴数据。
    title: 字符串类型，表示图的标题。
    
    返回值：
    None
    
    """
    # 计算回归性能指标
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)

    # 绘制散点图
    plt.figure(figsize=(7, 5))
    if min_ and max_:
        plt.xlim(min_, max_)
        plt.ylim(min_, max_)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xlabel("Real", fontsize=18)
    # plt.ylabel("Prediction", fontsize=18)
    if title != None:
        plt.title("%s\nR2:%.3f, MAE:%.3f, MSE:%.3f" % (title, r2, mae, mse), fontsize=24)
        plt.title("%s"%title,fontsize=24)
    plt.plot(x, y, marker="*", c="g")
    # 添加回归性能指标到图像的第二行
    
    # 显示图像
    plt.savefig("test.png", dpi=300, bbox_inches='tight')
    plt.show() 

def index_of_highly_correlated_features(X, threshold=0.95):
    """
    删除具有高相关性的特征
    
    参数：
    X: 二维数组类型，表示特征矩阵。
    threshold: float类型，表示要删除的最大相关系数，默认为0.95。
    
    返回值：
    X_new: 删除高相关性特征后的新特征矩阵。
    
    """
    # 计算特征之间的相关系数矩阵
    corr_matrix = pd.DataFrame(X).corr().abs()
    
    # 获取相关系数矩阵中上三角部分的索引
    upper_tri = corr_matrix.where(~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool))
    
    # 获取具有高相关性的特征的索引
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    return to_drop

def calculate_DA_bond_length_difference(data_name, log_file_dir = r"G:\work\Secondary_Selection\ZINC_0"):
    """计算一个DA反应中过渡态两根键的差异

    Args:
        data_name (str): file_path

    Returns:
        list: list of difference
    """    
    result = []
    distas = []
    distbs = []
    csv = pd.read_csv(data_name)
    
    for idx, row in tqdm(csv.iterrows()):
        # if row['Diene'] == "C=CC=C":
        #     target_log_file_dir = log_file_dir + "/pre_ene_reaction/ts_eng"
        # else:
        #     target_log_file_dir = log_file_dir + "/pre_diene_reaction/ts_eng"
        target_log_file_dir = log_file_dir + '/ts_eng'
        diene_id, ene_id, structure_id = row['Diene_Index'], row['Ene_Index'], row['Structure_id']
        title = row["Title"]
        title = [int(each) for each in title.split()]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        log_files = glob.glob(target_log_file_dir + '/%s*.log' % mol_name)
        if len(log_files) != 1:
            print(mol_name, len(log_files))
            continue
        log_file = log_files[0]
        log = logfile_process.Logfile(log_file)
        final_position = log.first_atom_position
        distance_a = Tool.get_atoms_distance(final_position[title[0]], final_position[title[2] + title[4]])
        distance_b = Tool.get_atoms_distance(final_position[title[1]], final_position[title[3] + title[4]])
        result.append(abs(distance_a - distance_b))
        distas.append(distance_a)
        distbs.append(distance_b)
    return np.array(result), np.array(distas), np.array(distbs)

def calc_distribution(y, eachsize=0.01, title=None, xlab=None, ylab="Freq"):
    y_max = np.max(y)
    y_min = np.min(y)
    X = np.arange(y_min, y_max + eachsize, eachsize)
    des = [0 for each in X]
    z = (y - y_min)/eachsize
    for each in z:
        des[int(each)] += 1
    des = np.array(des)
    des = des / len(y)
    plt.bar(X, des, width=eachsize/2, color="g")
    plt.xlim(y_min - eachsize, y_max + eachsize)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title != None:
        plt.title = title
    plt.savefig('test.svg', format='svg')
    plt.show()
    return des

def Find_intermoleculars_hydrogen_bond(data_name, inter_mols=True):
    csv = pd.read_csv(data_name)
    all_result = []
    log_file_dir = r"G:\work\Secondary_Selection\subset_data"
    for idx, row in tqdm(csv.iterrows()):
        # if row['Diene'] == "C=CC=C":
        #     target_log_file_dir = log_file_dir + "/pre_ene_reaction/ts_eng"
        # else:
        #     target_log_file_dir = log_file_dir + "/pre_diene_reaction/ts_eng"
        target_log_file_dir = log_file_dir + '/ts_eng'
        target_mol_file_dir = log_file_dir + '/mol'
        diene_id, ene_id, structure_id = row['Diene_Index'], row['Ene_Index'], row['Structure_Id']
        title = row["Title"]
        title = [int(each) for each in title.split()]
        mol_name = "%.5d_%.5d_%.5d" % (diene_id, ene_id, structure_id)
        mol_files = glob.glob(target_mol_file_dir + "/%s.mol" % mol_name)
        assert len(mol_files) == 1
        log_files = glob.glob(target_log_file_dir + '/%s*.log' % mol_name)
        assert len(log_files) == 1
        mol = Chem.MolFromMolFile(mol_files[0], removeHs=False)
        log_file = log_files[0]
        log = logfile_process.Logfile(log_file)
        final_position = log.first_atom_position
        symbol_list = log.symbol_list
        banned_list = []
        result = []
        for idx_a, (symbol_a, position_a) in enumerate(zip(symbol_list, final_position)):
            if symbol_a not in ["O", "N", "C"]:
                continue
            for idx_b, (symbol_b, position_b) in enumerate(zip(symbol_list, final_position)):
                if (idx_a < title[4] and idx_b < title[4]) or (idx_a >= title[4] and idx_b >= title[4]):
                    continue
                if idx_b in banned_list:
                    continue
                if symbol_b != "H":
                    continue
                if mol.GetBondBetweenAtoms(idx_a,idx_b) != None:
                    continue
                distance = Tool.get_atoms_distance(position_a, position_b)
                if distance < 2.2:
                    result.append([idx_a, idx_b])
                    banned_list.append(idx_b)
        all_result.append(result)        
    return all_result

def H_bond_count(diene_mol, ene_mol, title, hf_charges, distance = 2.1):   
    for mol, atom_lists in cycle_process.change_position(diene_mol, prop='diene', return_tran_cis=False):
        if atom_lists[0] == title[0] and atom_lists[-1] == title[1]:
            new_diene_mol = mol
            diene_list = atom_lists
            break
    for mol, atom_lists in cycle_process.change_position(ene_mol, prop='dieno'):
        if atom_lists[0] == title[2] and atom_lists[-1] == title[3]:
            new_ene_mol = mol
            ene_list = atom_lists
            break
        elif atom_lists[0] == title[3] and atom_lists[-1] == title[2]:
            new_ene_mol = cycle_process.rot_mol(mol, np.array([0, 1, 0]), sin=0, cos=-1)
            ene_list = [atom_lists[-1], atom_lists[0]]
            break
    if title[-1] == 0:
        new_ene_mol = cycle_process.move_mol(cycle_process.rot_mol(new_ene_mol, np.array([1, 0, 0]), sin=0, cos=-1), np.array([0, 0, distance]))
    else:
        new_ene_mol = cycle_process.move_mol(new_ene_mol, np.array([0, 0, distance]))
    diene_num1, diene_num2, diene_num3, diene_num4 = diene_list
    ene_num1, ene_num2 = ene_list
    start = new_diene_mol.GetNumAtoms()
    comb = Chem.CombineMols(new_diene_mol, new_ene_mol)
    rwcomb = Chem.RWMol(comb)
    Chem.Kekulize(rwcomb)
    another_bond1 = rwcomb.GetBondBetweenAtoms(diene_num1, diene_num2)
    another_bond2 = rwcomb.GetBondBetweenAtoms(diene_num3, diene_num4)
    bond3 = rwcomb.GetBondBetweenAtoms(diene_num2, diene_num3)
    rwcomb.AddBond(diene_num1, ene_num1 + start, Chem.BondType.SINGLE)
    rwcomb.AddBond(diene_num4, ene_num2 + start, Chem.BondType.SINGLE)
    another_bond1.SetBondType(Chem.BondType.SINGLE)
    another_bond2.SetBondType(Chem.BondType.SINGLE)
    bond4 = rwcomb.GetBondBetweenAtoms(ene_num1 + start, ene_num2 + start)
    bond3.SetBondType(Chem.BondType.DOUBLE)
    if bond4.GetBondType() == Chem.BondType.DOUBLE:
        bond4.SetBondType(Chem.BondType.SINGLE)
    elif bond4.GetBondType() == Chem.BondType.TRIPLE:
        bond4.SetBondType(Chem.BondType.DOUBLE)
    else:
        print(bond4.GetBondType())
        raise TypeError("Bond Type is Error")
    Chem.SanitizeMol(rwcomb)
    sum_nwmol = rwcomb.GetMol()
    ff = Chem.AllChem.UFFGetMoleculeForceField(sum_nwmol)
    ff.UFFAddDistanceConstraint(diene_num1, ene_num1 + start, False, 2.1, 2.2, 10000)
    ff.UFFAddDistanceConstraint(diene_num4, ene_num2 + start, False, 2.1, 2.2, 10000)
    ff.Initialize()
    ff.Minimize()
    base_energy = ff.CalcEnergy()

    all_engs = []
    all_results = []
    all_mols = []
    group_a = np.arange(start)
    group_b = np.arange(start, sum_nwmol.GetNumAtoms())

    group_a_charges = hf_charges[group_a]
    group_b_charges = hf_charges[group_b]
    group_a_min_idxs = [each for each in group_a if sum_nwmol.GetAtomWithIdx(int(each)).GetSymbol() in ['N', 'O', 'F']]
    group_a_max_idxs = group_a[np.argsort(group_a_charges)[-10:]]
    group_b_min_idxs = [each for each in group_b if sum_nwmol.GetAtomWithIdx(int(each)).GetSymbol() in ['N', 'O', 'F']]
    group_b_max_idxs = group_b[np.argsort(group_b_charges)[-10:]]
    for max_idxs, min_idxs in [[group_b_max_idxs, group_a_min_idxs], [group_a_max_idxs, group_b_min_idxs]]:
        for i in max_idxs:
            for j in min_idxs:
                mol = copy.deepcopy(sum_nwmol)
                ff = Chem.AllChem.UFFGetMoleculeForceField(mol)
                ff.UFFAddDistanceConstraint(diene_num1, ene_num1 + start, False, 2.1, 2.2, 10000)
                ff.UFFAddDistanceConstraint(diene_num4, ene_num2 + start, False, 2.1, 2.2, 10000)
                ff.UFFAddDistanceConstraint(int(i),int(j),False,2.0,2.4, 10000.0)
                ff.Initialize()
                ff.Minimize()
                ff = Chem.AllChem.UFFGetMoleculeForceField(mol)
                all_engs.append(ff.CalcEnergy() + hf_charges[i] * hf_charges[j])
                all_results.append([i,j])
                all_mols.append(mol)
    
    # ON_atom_ids = [each.GetIdx() for each in new_comb.GetAtoms() if each.GetSymbol() in ["O", "N"]]
    # all_result = []
    # Chem.AllChem.MMFFOptimizeMolecule(new_comb)
    # pre_eng = Chem.AllChem.MMFFGetMoleculeForceField(new_comb, Chem.AllChem.MMFFGetMoleculeProperties(new_comb)).CalcEnergy()
    # for atom_id in ON_atom_ids:
    #     atom = new_comb.GetAtomWithIdx(atom_id)
    #     if "H" not in [each.GetSymbol() for each in atom.GetNeighbors()]:
    #         continue
    #     next_atom_ids = [each for each in ON_atom_ids if each >= start] if atom_id < start else [each for each in ON_atom_ids if each < start]
    #     for next_atom_id in next_atom_ids:
    #         new_mol = copy.deepcopy(new_comb)
    #         rwmol = Chem.RWMol(new_mol)
    #         atom_id_0 = rwmol.AddAtom(Chem.Atom(6))
    #         rwmol.AddBond(atom_id, atom_id_0, Chem.BondType.SINGLE)
    #         rwmol.AddBond(next_atom_id, atom_id_0, Chem.BondType.SINGLE)
    #         rwmol.GetAtomWithIdx(atom_id).SetFormalCharge(1)
    #         rwmol.GetAtomWithIdx(next_atom_id).SetFormalCharge(1)
    #         rwmol.GetAtomWithIdx(atom_id_0).SetFormalCharge(2)
    #         new_hb_mol = rwmol.GetMol()
    #         Chem.SanitizeMol(new_hb_mol)
    #         Chem.AllChem.MMFFOptimizeMolecule(new_hb_mol)
    #         new_eng = Chem.AllChem.MMFFGetMoleculeForceField(new_hb_mol, Chem.AllChem.MMFFGetMoleculeProperties(new_hb_mol)).CalcEnergy()
    #         all_result.append(new_eng - pre_eng)
    #         # if new_eng - pre_eng < 40 and new_eng - pre_eng > 0:
    #         #     break
    # result_1 = len([each for each in all_result if each < 50 and each > 0])
    # result_2 = len([each for each in all_result if each < 100 and each > 0])
    if len(all_engs) == 0:
        return [0]
    else:
        return [np.min(all_engs) - base_energy]

def Diene_Ene_Process(diene_mol, ene_mol, title, calc_bv=0):
    areas = []
    bv = []                      
    for mol, atom_lists in cycle_process.change_position(diene_mol, prop='diene', return_tran_cis=False):
        if atom_lists[0] == title[0] and atom_lists[-1] == title[1]:
            areas += Calc_areas(mol, atom_lists)
            if calc_bv:
                areas += Calc_bv(mol, atom_lists)
            break
    for mol, atom_lists in cycle_process.change_position(ene_mol, prop='dieno'):
        if atom_lists[0] == title[2] and atom_lists[-1] == title[3]:
            result = Calc_areas(mol, atom_lists)
            if calc_bv:
                bv = Calc_bv(mol, atom_lists)
            break
        if atom_lists[0] == title[3] and atom_lists[-1] == title[2]:
            result = Calc_areas(mol, atom_lists)
            result = [result[5], result[4], result[7], result[6], result[1], result[0], result[3], result[2]]
            if calc_bv:
                bv = Calc_bv(mol, atom_lists)
                bv = [bv[2], bv[3], bv[0], bv[1]]
            break
    if title[-1] == 0:
        result = [result[3], result[2], result[1], result[0], result[7], result[6], result[5], result[4]]
        if calc_bv:
            bv = [bv[1], bv[0], bv[3], bv[2]]
    areas += result
    areas += bv
    return areas

def Calc_bv(mol, title):
    from morfeus import BuriedVolume
    result = []
    symbol_lists = [atom.GetSymbol() for atom in mol.GetAtoms()]
    mol_conformers = mol.GetConformers()
    geom = mol_conformers[0].GetPositions()
    for id, each in enumerate(title):
        neighbors = [each for each in mol.GetAtomWithIdx(each).GetNeighbors() if each.GetIdx() not in title]
        for eachnei in neighbors:
            a = cycle_process.find_neighbor_atom_number(mol, [[eachnei.GetIdx()]], times=3, exclude_atoms=title)
            a = list(a)
            a.append(each)
            exclude_atoms = [each for each in range(mol.GetNumAtoms()) if each not in a]
            center_atom = each
            # z_atom = eachnei.GetIdx()
            # xz_atom = [each for each in a if each not in [center_atom, z_atom]][0]
            a = BuriedVolume(symbol_lists, geom, center_atom + 1, include_hs=1, radius=2, excluded_atoms=exclude_atoms)
            result.append(a.buried_volume / (a.buried_volume + a.free_volume))
        if len(neighbors) == 0:
            if id in [1,2] and len(title) == 4:
                result += [0] 
            else:
                result += [0,0]
        elif len(neighbors) == 1:
            if len(title) == 4 and id in [1,2]:
                continue
            result += [0]
    return result

table = Chem.rdchem.GetPeriodicTable()
VAN_DER_WAALS_RADII = {each:table.GetRvdw(each) for each in ["H", "B", "C", "N", "O", "F", "S", "Cl", 'Br']}
# {'H': 1.2,'B': 1.8,'C': 1.7,'N': 1.6,'O': 1.55,'F': 1.5,'S': 1.8,'Cl': 1.8,'Br': 1.9}
def Calc_areas(mol, atoms_ids, ):
    # 非均匀格点积分
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
    y = x; z = np.linspace(0.1 -2, 2 - 0.1, num)
    # 调节格点均匀度
    z = (-0.5 * (np.abs(z) - 2) ** 2  + 2 ) * z / np.abs(z)
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
    # points_inside_z = points_inside / (2 - np.abs(points[:, 2]))
    distance2 = np.linalg.norm(points, axis=1)
    for i in range(8):
        x_positive = i // 4 % 2 == 1
        y_positive = i // 2 % 2 == 1
        z_positive = i % 2 == 1
        
        counts[i] = np.sum(points_inside_z[np.all([(points[:, 0] > 0) == x_positive, (points[:, 1] > 0) == y_positive, (points[:, 2] > 0) == z_positive], axis=0)])

    counts = counts / total_points * 8
    # for a,b in [[0,2], [1,3], [4,6], [5,7]]:
    #     counts[a], counts[b] = (counts[a] + counts[b]) / 2, (counts[a] + counts[b]) / 2

    # for a,b in [[0,1], [2,3], [4,5], [6,7]]:
    #     counts[a], counts[b] = (counts[a] + counts[b]) / 2, (counts[a] + counts[b]) / 2
    
    return counts.tolist()
