import numpy as np


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''

    #*********** Begin ***********#
    # 计算标签的总体信息熵
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    # 计算总体信息熵
    total_entropy = entropy(label)
    
    # 计算条件熵
    unique_values, counts = np.unique(feature[:, index], return_counts=True)
    conditional_entropy = 0.0
    
    for value, count in zip(unique_values, counts):
        subset_labels = label[feature[:, index] == value]
        conditional_entropy += (count / len(feature)) * entropy(subset_labels)

    # 计算信息增益
    info_gain = total_entropy - conditional_entropy
    
    return info_gain
    #*********** End *************#