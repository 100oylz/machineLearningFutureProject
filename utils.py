import os
import sys

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as IO
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


# 读取文件
def loadMatFile(filepath):
    """
    从MAT文件加载数据。

    Args:
        filepath (str): MAT文件的路径。

    Returns:
        dict: 包含MAT文件数据的字典。
    """
    data = IO.loadmat(filepath)
    data_dict = {}
    for key, value in data.items():
        # 排除MAT文件的元信息
        if key not in ('__version__', '__globals__', '__header__'):
            value = np.ascontiguousarray(value)
            data_dict[key] = value.astype('float64')
    return data_dict


ADNI = loadMatFile('dataset/dataset/ADNI.mat')
ADNI_fMRI = loadMatFile('dataset/dataset/ADNI_90_120_fMRI.mat')
FTP_fMRI = loadMatFile('dataset/dataset/FTD_90_200_fMRI.mat')
OCD_fMRI = loadMatFile('dataset/dataset/OCD_90_200_fMRI.mat')
PPMI = loadMatFile('dataset/dataset/PPMI.mat')


def normalizedata(data: np.ndarray):
    """
    对数据进行标准化处理。

    Args:
        data (np.ndarray): 输入的数据数组。

    Returns:
        np.ndarray: 标准化后的数据数组。
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # 检查标准差是否为零，如果为零则设置为1.0，以避免除以零的情
    std[std == 0] = 1.0

    normalized_data = (data - mean) / std
    return normalized_data


def generateDataSet(dataDict: dict):
    """
    生成完整的数据集，包括数据、标签和标签的映射。

    Args:
        dataDict (dict): 包含标签和相应数据的字典。键是标签，值是数据列表。

    Returns:
        tuple: 包含三个元素的元组，依次为：
        - np.ndarray: 数据的NumPy数组。
        - np.ndarray: 标签的NumPy数组，与数据数组一一对应。
        - np.ndarray: 标签到标签名称的映射的NumPy数组。
    """
    data = []
    label = []
    labelMap = []
    pointer = 0
    for key, value in dataDict.items():
        labelMap.append(key)
        for item in value:
            label.append(pointer)
            data.append(item)
        pointer += 1
    return np.array(data), np.array(label), np.array(labelMap)


def plot_training_history(loss_history, acc_history, valid_acc_history, title, datasetname):
    """
    绘制训练历史的损失和准确度曲线，并保存为图像文件。

    Args:
        loss_history (list): 包含每个epoch损失的列表。
        acc_history (list): 包含每个epoch准确度的列表。
        title (str): 图像的整体标题。
        datasetname (str): 数据集的名称，用于保存图像文件。
    """
    # 创建一个新的图像
    plt.figure(figsize=(20, 6))

    # 添加整体标题
    plt.suptitle(title)

    # 绘制损失曲线model
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确度曲线
    plt.subplot(1, 3, 2)
    plt.plot(acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(valid_acc_history, label='Valid_Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Valid_Accuracy')
    plt.legend()

    # 保存图像文件到指定路径
    plt.savefig(f'images/MLP/{datasetname}.png')

    # 如果需要显示图像，可以取消下一行的注释
    # plt.show()
    plt.close()


def plot_loss_history(loss_history, title, datasetname):
    """
    绘制损失历史曲线并保存为图像文件。

    Args:
        loss_history (list): 包含每个epoch损失的列表。
        title (str): 图像的标题。
        datasetname (str): 数据集的名称，用于保存图像文件。
    """
    # 创建一个新的图像
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    # 保存图像文件到指定路径
    plt.savefig(f'images/AutoEncoder/{datasetname}.png')

    # 如果需要显示图像，可以取消下一行的注释
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, datasetname, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵的热度图表格。

    Args:
        y_true (array-like): 真实标签数组。
        y_pred (array-like): 预测标签数组。
        classes (list): 类别名称的列表。
        datasetname (str): 数据集的名称，用于保存图像文件。
        title (str, optional): 图表标题。默认为'Confusion Matrix'。
        cmap (colormap, optional): 热度图的颜色映射。默认为蓝色。

    Returns:
        None
    """
    # print(y_true)
    # print(y_pred)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图表
    plt.figure(figsize=(len(classes) + 2, len(classes) + 2))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, square=True,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(f'images/MLP/{datasetname}_confusion_matrix.png')
    plt.close()
    # 如果需要显示图表，可以取消下一行的注释
    # plt.show()


def accuracy(output, label):
    """
    计算模型的准确度。

    Args:
        output (torch.Tensor): 模型的输出张量，通常是 softmax 或 sigmoid 的结果。
        label (torch.Tensor): 真实标签张量。

    Returns:
        float: 准确度，范围在 0 到 1 之间。
    """
    _, predicted = torch.max(output, dim=1)  # 返回当前张量的最大值的数值和索引
    correct_items = (predicted == label).sum().item()  # 计算正确的预测数
    samples = label.shape[0]  # 样本总数
    accuracy = correct_items / samples  # 计算准确度
    return accuracy


def logTransform(datadict):
    """
    对给定的数据字典进行对数变换并返回新的数据字典。

    参数：
    - datadict: 包含键值对的字典，其中键是数据集的名称，值是NumPy数组。

    返回：
    - logdatadict: 包含对数变换后的数据的新字典，采用相同的键。

    注释：
    1. 创建一个空的字典logdatadict来存储对数变换后的数据。
    2. 对于输入字典中的每个键值对，将值的副本存储在nonZerovalue中，以便不修改原始数据。
    3. 将nonZerovalue中的所有零值替换为1e-18，以避免对数运算中的除零错误。
    4. 计算非零值的自然对数，并将结果存储在logdatadict中，使用相同的键。
    5. 最后返回logdatadict。
    """
    logdatadict = {}
    for key, value in datadict.items():
        nonZerovalue = value.copy()
        nonZerovalue[nonZerovalue == 0] = 1e-18
        logdatadict[key] = np.log(nonZerovalue)
    return logdatadict


def MaskdataDict(datadict, maskKey, maskValue):
    """
    根据指定的掩码键和掩码值，对给定的数据字典进行掩码操作并返回新的数据字典。

    参数：
    - datadict: 包含键值对的字典，其中键是数据集的名称，值是NumPy数组。
    - maskKey: 一个列表，包含需要进行掩码操作的键的名称。
    - maskValue: 用于掩码的新键名称。

    返回：
    - maskdatadict: 包含掩码操作后的数据的新字典，采用相同的键。

    注释：
    1. 创建一个空的字典maskdatadict来存储掩码操作后的数据。
    2. 遍历输入字典中的每个键值对。
    3. 如果键不在maskKey列表中，则将原始值的副本存储在maskdatadict中，使用相同的键。
    4. 如果键在maskKey列表中，检查maskValue是否已经在maskdatadict中存在。
    5. 如果maskValue不存在，创建一个新的键maskValue，并将原始值的副本存储在其中。
    6. 如果maskValue已经存在，将原始值的副本与现有值进行连接。
    7. 最后返回maskdatadict。
    """
    maskdatadict = {}
    for key, value in datadict.items():
        if key not in maskKey:
            maskdatadict[key] = np.copy(value)
        else:
            if maskValue not in maskdatadict:
                maskdatadict[maskValue] = np.copy(value)
            else:
                maskdatadict[maskValue] = np.concatenate((maskdatadict[maskValue], np.copy(value)))
    print(maskdatadict.keys())
    return maskdatadict


def slicedataDict(datadict, sliceKey):
    """
    根据指定的切片键，从给定的数据字典中提取数据并返回新的数据字典。

    参数：
    - datadict: 包含键值对的字典，其中键是数据集的名称，值是NumPy数组。
    - sliceKey: 一个列表，包含需要提取的键的名称。

    返回：
    - slicedatadict: 包含提取数据后的数据的新字典，采用相同的键。

    注释：
    1. 创建一个空的字典slicedaDict来存储提取的数据。
    2. 遍历输入字典中的每个键值对。
    3. 如果键在sliceKey列表中，将原始值的副本存储在slicedaDict中，使用相同的键。
    4. 最后返回slicedaDict。
    """
    slicedatadict = {}
    for key, value in datadict.items():
        if key in sliceKey:
            slicedatadict[key] = np.copy(value)
    return slicedatadict


def train_valid_test_split(datadict, randomstate):
    data, label, labelMap = generateDataSet(datadict)
    data = normalizedata(data)
    # 使用Stratified Sampling划分训练集和临时集（包括验证集和测试集）
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    # 使用Stratified Sampling划分验证集和测试集
    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    # 使用SMOTE进行过采样，保持训练集的平衡
    smote = SMOTE(random_state=randomstate)
    train_data_resampled, train_label_resampled = smote.fit_resample(train_data, train_label)

    return train_data_resampled, train_label_resampled, valid_data, valid_label, test_data, test_label, labelMap


def reshapeDataDict(datadict):
    for key, value in datadict.items():
        arrayvalue = np.array(value)
        datadict[key] = arrayvalue.reshape(arrayvalue.shape[0], -1)
        # print(datadict[key].shape)
    return datadict


def calculatequanzhong(datadict):
    quanzhong = []
    for key, value in datadict.items():
        # print(value.shape[0])
        quanzhongvalue = value.shape[0]
        quanzhong.append(1.0 / quanzhongvalue)
    return quanzhong


if __name__ == '__main__':
    # train_data, train_label, valid_data, valid_label, test_data, test_label, labelMap = train_valid_test_split(ADNI, 0)
    # print(train_label.shape, valid_label.shape, test_label.shape)
    # print(train_label, test_label, test_label)
    # ADNI = logTransform(ADNI)
    # PPMI = logTransform(ADNI)
    # ADNI_fMRI = logTransform(ADNI_fMRI)
    # OCD_fMRI = logTransform(OCD_fMRI)
    # FTP_fMRI = logTransform(FTP_fMRI)
    print(calculatequanzhong(FTP_fMRI))
