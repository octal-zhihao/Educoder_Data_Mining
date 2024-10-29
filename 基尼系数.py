import numpy as np

def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''

    #********* Begin *********#
    # 将特征转换为数组
    feature = np.array(feature)
    
    # 获取该特征的不同取值
    unique_values = np.unique(feature[:, index])

    # 初始化基尼系数
    total_gini = 0.0

    for value in unique_values:
        # 获取该特征值下的子集
        subset = label[feature[:, index] == value]

        # 计算该子集的基尼指数
        label_counts = np.bincount(subset)
        subset_size = len(subset)
        gini_sub = 1.0 - sum((count / subset_size) ** 2 for count in label_counts)
        
        # 计算加权基尼指数
        total_gini += (subset_size / len(label)) * gini_sub

    return total_gini
    #********* End *********#