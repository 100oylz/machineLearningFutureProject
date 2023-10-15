import torch

# 定义MLP数据集类
class MLPdataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        """
        MLP数据集类的初始化函数。

        Args:
            data (numpy.ndarray): 包含数据的NumPy数组。
            label (numpy.ndarray): 包含标签的NumPy数组。

        Returns:
            None
        """
        super(MLPdataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        """
        获取数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        获取数据集中的单个样本。

        Args:
            item (int): 样本的索引。

        Returns:
            torch.FloatTensor: 包含数据的浮点数张量。
            torch.LongTensor: 包含标签的长整数张量。
        """
        data = torch.FloatTensor(self.data[item])
        label = torch.LongTensor([self.label[item]])  # 创建一个包含一个元素的一维长整数张量
        return data, label.squeeze()

# 定义自编码器数据集类
class Autoencoderdataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        自编码器数据集类的初始化函数。

        Args:
            data (numpy.ndarray): 包含数据的NumPy数组。

        Returns:
            None
        """
        super(Autoencoderdataset, self).__init__()
        self.data = data

    def __len__(self):
        """
        获取数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        获取数据集中的单个样本。

        Args:
            item (int): 样本的索引。

        Returns:
            torch.FloatTensor: 包含数据的浮点数张量。
        """
        data = torch.FloatTensor(self.data[item])
        return data
