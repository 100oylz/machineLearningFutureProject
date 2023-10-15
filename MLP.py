import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
from dataset import MLPdataset
from utils import *

# DEFAULTHIDDENFEATURES = [256, 512, 1024, 2048, 4096, 8192]
# 下面专属于ADNI
DEFAULTHIDDENFEATURES = [256, 128]
DEFAULTBigHIDDENFEATURES = [2048, 1024, 512, 256]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(24)


class ResLinear(nn.Module):
    def __init__(self, in_out_features, hidden_features, out_features):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_out_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_features, in_out_features),
            nn.BatchNorm1d(in_out_features)
        )
        self.fc = nn.Sequential(nn.Linear(in_out_features, out_features), nn.PReLU(),
                                nn.Dropout(p=0.5))
        self.prelu = nn.PReLU()

    def forward(self, x):
        res = x  # 保存输入
        out = self.linear(x)  # 通过线性层进行前向传播
        out += res  # 将输入和输出相加（残差连接）
        out = self.prelu(out)  # 通过ReLU激活函数进行激活
        out = self.fc(out)
        return out


class NetWork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        """
        初始化神经网络模型。

        Args:
            in_features (int): 输入特征的数量。
            hidden_features (list): 隐藏层特征的列表。
            out_features (int): 输出特征的数量。

        Returns:
            None
        """
        super(NetWork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc = self._make_fc_layers()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)  # 初始化 Batch Normalization 的权重为1
                nn.init.constant_(m.bias.data, 0)  # 初始化 Batch Normalization 的偏置为0

    def _make_fc_layers(self):
        """
        创建全连接层的私有辅助函数。

        Returns:
            nn.Sequential: 包含全连接层的序列模型。
        """
        layers = []
        in_features = self.in_features
        hidden_features = self.hidden_features

        # 构建残差连接层
        for hidden_feature in hidden_features:
            if (in_features > 256):
                reshidden_features = 256
            else:
                reshidden_features = int(in_features / 2)
            # res_linear = ResLinear(in_features, reshidden_features, hidden_feature)
            # layers.append(res_linear)
            layers.append(nn.Linear(in_features, hidden_feature))
            # layers.append(nn.BatchNorm1d(hidden_feature, eps=1e-18))  # 批归一化层
            layers.append(nn.Dropout(p=0.5))  # 随机失活层
            layers.append(nn.ReLU())  # 激活函数
            in_features = hidden_feature

        # 输出层
        layers.append(nn.Linear(in_features, self.out_features))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.fc(x)


def trainmlp(traindata, validdata, labelMap, num_epoch, batch_size, datasetname, randomstate, quanzhong,
             preload=False, validpatience=50, Big=False):
    """
    训练多层感知器(MLP)模型。

    Args:
        traindata (torch.Tensor): 包含数据的张量。
        labelMap (list): 类别标签的映射列表。
        num_epoch (int): 训练的总轮次。
        batch_size (int): 批处理大小。
        datasetname (str): 数据集名称。

    Returns:
        None
    """
    # 确定输入和输出特征数量
    in_feature = traindata.data.shape[-1]
    out_feature = len(labelMap)
    min_loss = float('inf')
    max_accuracy = 0.0
    max_valid_accuracy = 0.0
    early_stopping_counter = 0    # 创建数据加载器
    data_loader = DataLoader(traindata, batch_size, shuffle=True)
    valid_data_loader = DataLoader(validdata, batch_size)
    # 创建数据加载器
    data_loader = DataLoader(traindata, batch_size, shuffle=True)
    valid_data_loader = DataLoader(validdata, batch_size)
    # 初始化MLP模型
    if (Big == False):
        hidden_features = DEFAULTHIDDENFEATURES  # 请确保此处的DEFAULTHIDDENFEATURES已定义
    else:
        hidden_features = DEFAULTBigHIDDENFEATURES
    # 检查是否可用GPU，如果可用，将模型移至GPU上
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quanzhong1 = torch.FloatTensor([2 if label != 'NC' else 1 for label in labelMap]).to(device)
    quanzhong = torch.FloatTensor(quanzhong).to(device)
    quanzhong = quanzhong1 * quanzhong
    criterion = nn.CrossEntropyLoss(quanzhong)


    if (preload == False):
        model = NetWork(in_feature, hidden_features, out_feature)
        optimizer = optim.NAdam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        model.to(device)
    else:
        model = torch.load(f'./checkpoint/MLP/{datasetname}_randomstate_{randomstate}_best_accuracy.pt')
        optimizer = optim.NAdam(model.parameters(), lr=5e-5, weight_decay=5e-4)
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_test_acc = 0.0
        total_valid_loss = 0.0
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            output = model(batch_data)

            loss = criterion(output, batch_label)
            acc = accuracy(nn.functional.softmax(output, dim=1), batch_label)



            total_loss += loss.item()
            total_acc += acc
        for batch_data, batch_label in valid_data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            output = model(batch_data)
            loss = criterion(output, batch_label)
            acc = accuracy(nn.functional.softmax(output, dim=1), batch_label)
            total_valid_loss += loss.item()
            total_test_acc += acc

        min_loss = total_loss / len(data_loader)
        max_accuracy_acc = total_acc / len(data_loader)
        max_valid_accuracy = total_test_acc / len(valid_data_loader)
        epoch_valid_loss = total_valid_loss / len(valid_data_loader)
        print(min_loss,max_accuracy_acc,max_valid_accuracy)


    # 创建日志文件并打印模型信息
    f = open(f'./journal/MLP/{datasetname}_randomstate_{randomstate}.txt', 'w')
    print("MLP Model:", file=f)
    print(model, file=f)
    print("MLP Model:")
    print(model)
    print(criterion.weight, file=f)
    print(criterion.weight)


    # 定义优化器和损失函数


    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    # 打印优化器和损失函数信息
    print("Optimizer:", optimizer)
    print("Criterion:", criterion)

    num_epochs = num_epoch



    # 初始化存储每个epoch损失和准确度的列表
    epoch_loss_history = []
    epoch_acc_history = []
    epoch_valid_acc_history = []
    # 初始化最小损失和最大准确度

    # 开始训练循环
    for num_epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        total_test_acc = 0.0
        total_valid_loss = 0.0
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            output = model(batch_data)

            loss = criterion(output, batch_label)
            acc = accuracy(nn.functional.softmax(output, dim=1), batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
        model.eval()

        for batch_data, batch_label in valid_data_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            output = model(batch_data)
            loss = criterion(output, batch_label)
            acc = accuracy(nn.functional.softmax(output, dim=1), batch_label)
            total_valid_loss += loss.item()
            total_test_acc += acc
        model.train()

        # 计算平均损失和准确度
        epoch_loss = total_loss / len(data_loader)
        epoch_acc = total_acc / len(data_loader)
        epoch_valid_acc = total_test_acc / len(valid_data_loader)
        epoch_valid_loss = total_valid_loss / len(valid_data_loader)
        # scheduler.step(epoch_loss)

        # 检查是否需要保存模型
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model, f'checkpoint/MLP/{datasetname}_randomstate_{randomstate}_min_loss.pt')
            print(f'Epoch [{num_epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Saved!', file=f)
            print(f'Epoch [{num_epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Saved!')

        # 记录每个epoch的损失和准确度
        epoch_loss_history.append(epoch_loss)
        epoch_acc_history.append(epoch_acc)
        epoch_valid_acc_history.append(epoch_valid_acc)

        print(
            f'Epoch [{num_epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f},ValidSet Acc: {epoch_valid_acc:.4f}',
            file=f)
        print(
            f'Epoch [{num_epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f},ValidSet Acc: {epoch_valid_acc:.4f}')
        if epoch_valid_acc > max_valid_accuracy:
            print(f'Epoch [{num_epoch + 1}/{num_epochs}],ValidSet Acc: {epoch_valid_acc:.4f} Updated!')
            print(f'Epoch [{num_epoch + 1}/{num_epochs}],ValidSet Acc: {epoch_valid_acc:.4f} Updated!', file=f)
            max_valid_accuracy = epoch_valid_acc
            early_stopping_counter = 0
            torch.save(model, f'checkpoint/MLP/{datasetname}_randomstate_{randomstate}_best_accuracy.pt')
        else:
            # 如果验证集的准确度没有提高，增加早停的计数器
            print(f'Valid Loss:{epoch_valid_loss},Valid Acc:{epoch_valid_acc}')
            early_stopping_counter += 1
        print(early_stopping_counter)
        # 判断是否触发早停
        if early_stopping_counter >= validpatience:
            print("Early stopping triggered.")
            break
        if (epoch_valid_acc > 1 - 1e-5):
            break
        if (epoch_loss < 1e-3):
            break
    # 绘制训练历史
    plot_training_history(epoch_loss_history, epoch_acc_history, epoch_valid_acc_history,
                          f"MLP-{datasetname}-randomstate_{randomstate}",
                          datasetname + f'_randomstate_{randomstate}')

    # 关闭日志文件
    f.close()


def TrainMLP(dataset, datasetname, num_epochs=100, validpatience=50, Big=False):
    """
    训练多层感知器(MLP)模型的入口函数。

    Args:
        dataset (dict): 包含数据集的字典，包括标签、数据和标签的映射。
        datasetname (str): 数据集名称。
        num_epochs (int, optional): 训练的总轮次，默认为100。

    Returns:
        None
    """
    # 从数据集字典中提取数据、标签和标签映射
    randomstateList = [0, 1, 2, 3, 4]
    quanzhong = calculatequanzhong(dataset)
    for randomstate in randomstateList:
        print(f'RandomState:{randomstate}', end='\n\n\n')
        train_data, train_label, valid_data, valid_label, _, _, labelMap = train_valid_test_split(dataset, randomstate)
        print(train_data.shape, valid_data.shape)
        # 数据预处理
        train_data, valid_data = np.array(train_data), np.array(valid_data)
        train_data, valid_data = normalizedata(train_data), normalizedata(valid_data)

        # 创建MLP数据集
        traindataset = MLPdataset(train_data, train_label)
        validdataset = MLPdataset(valid_data, valid_label)

        # 调用trainmlp函数进行训练
        trainmlp(traindataset, validdataset, labelMap, num_epochs, 16, datasetname, randomstate,
                 validpatience=validpatience, Big=Big, quanzhong=quanzhong, preload=True)


def testMLPmodel(data, labelMap, batch_size, datasetname, randomstate, Big):
    """
    使用已经训练好的MLP模型进行测试，并返回真实标签和预测标签的列表。

    Args:
        data (torch.Tensor): 用于测试的数据集。
        labelMap (list): 标签的映射。
        batch_size (int): 批处理大小。
        datasetname (str): 数据集名称。

    Returns:
        list: 真实标签的列表。
        list: 预测标签的列表。
    """
    # model = torch.load(f'checkpoint/MLP/{datasetname}_max_accuracy.pt')
    # 确定输入和输出特征数量
    # print(batch_size)
    in_feature = data.data.shape[-1]
    # print(data.data.shape[-1])
    out_feature = len(labelMap)

    if (Big == False):
        hidden_features = DEFAULTHIDDENFEATURES  # 请确保此处的DEFAULTHIDDENFEATURES已定义
    else:
        hidden_features = DEFAULTBigHIDDENFEATURES
    # model = NetWork(in_feature, hidden_features, out_feature)
    pathformat = f'checkpoint/MLP/{datasetname}_randomstate_{randomstate}_best_accuracy.pt'
    # pathformat = f'checkpoint/MLP/{datasetname}_randomstate_{randomstate}_min_loss.pt'

    # modeldict = torch.load(pathformat)
    # model.load_state_dict(modeldict)
    model = torch.load(pathformat)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = DataLoader(data, batch_size, shuffle=True)
    # print(data.data.shape)
    model.to(device)

    model.eval()  # 设置模型为评估模式
    # model.train()
    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储预测标签

    with torch.no_grad():
        total_acc_items = 0
        for data, labels in data_loader:
            # print(data.shape)
            # if(data.shape)
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            # print(output.shape)
            # print(nn.functional.softmax(output, 1).shape)
            _, predicted = torch.max(output, 1)
            acc = accuracy(output, labels)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            total_acc_items += acc
    print(f'Accuracy:{total_acc_items / len(data_loader)}')
    return true_labels, predicted_labels


if __name__ == '__main__':
    pass
    # # 启用自动梯度异常检测
    # torch.autograd.set_detect_anomaly(True)
    # # reshapeDataDict(FTP_fMRI)
    # # testADNI()
    # # trainADNIMultiMLP()
    # TrainMLP(ADNI, 'ADNI', 10, validpatience=500, Big=True)
    # # TrainMLP(PPMI, 'PPMI', 500)
    # OCD_fMRI = reshapeDataDict(OCD_fMRI)
    # # ADNI_fMRI = reshapeDataDict(ADNI_fMRI)
    FTP_fMRI = reshapeDataDict(FTP_fMRI)
    # # TrainMLP(OCD_fMRI, 'OCD_fMRI', 500)
    # # TrainMLP(ADNI_fMRI, 'ADNI_fMRI', 500)
    TrainMLP(FTP_fMRI, 'FTP_fMRI', 1, Big=True)
