# Import relevant functions and modules from other scripts

import MLP
import dataset
from MLP import *

# Alias some functions for easier use
TrainMLP = MLP.TrainMLP
TestMLP = MLP.testMLPmodel
mydataset = dataset.MLPdataset


# Define a function to draw confusion matrices for a given dataset
def draw_confusion_matrix(dataset, datasetname, need_encode=False, Big=False):
    """
    绘制混淆矩阵函数。

    Args:
        dataset: 包含数据集的字典，包括标签、数据和标签的映射。
        datasetname (str): 数据集的名称。
        need_encode (bool, optional): 是否需要编码，默认为 False。

    Returns:
        None
    """
    # Load data and labels from the dataset
    print(datasetname)
    randomstateList = [0, 1, 2, 3, 4]
    quanzhong = calculatequanzhong(dataset)
    for randomstate in randomstateList:
        print(randomstate)
        traindata, trainlabel, validdata, validlabel, testdata, testlabel, labelMap = train_valid_test_split(dataset,
                                                                                                             randomstate)
        testdata = np.array(testdata, dtype=np.float64)

        # Normalize the data
        # data = normalizedata(data)

        if need_encode:
            pass
        else:
            print('test')
            drawplotimage(datasetname, labelMap, randomstate, testdata, testlabel, 'test', Big)
            print('valid')
            drawplotimage(datasetname, labelMap, randomstate, validdata, validlabel, 'valid', Big)
            print('train')
            drawplotimage(datasetname, labelMap, randomstate, traindata, trainlabel, 'train', Big)


def drawplotimage(datasetname, labelMap, randomstate, testdata, testlabel, datasetnameformat, Big=False):
    # Create a dataset object using mydataset class
    ADNIdataset = mydataset(testdata, testlabel)
    # Test only the MLP model on the dataset
    true_labels, pred_labels = TestMLP(ADNIdataset, labelMap, 16, datasetname, randomstate, Big)
    # Plot a confusion matrix based on the true and predicted labels
    plot_confusion_matrix(true_labels, pred_labels, labelMap,
                          datasetname + f'_randomstate_{randomstate}_{datasetnameformat}')


# Define a function to train all datasets
def train_all_datasets():
    """
    训练所有数据集的函数。

    Returns:
        None
    """
    TrainMLP(ADNI, 'ADNI', 1000, validpatience=100, Big=False)
    TrainMLP(PPMI, 'PPMI', 1000, validpatience=50, Big=False)
    TrainMLP(OCD_fMRI, 'OCD_fMRI', 1000, validpatience=50, Big=True)
    TrainMLP(ADNI_fMRI, 'ADNI_fMRI', 1000, validpatience=100, Big=True)
    TrainMLP(FTP_fMRI, 'FTP_fMRI', 1000, validpatience=50, Big=True)


# Define a function to test all datasets and draw confusion matrices
def test_all_datasets():
    """
    测试所有数据集并绘制混淆矩阵的函数。

    Returns:
        None
    """
    # Test MLP and AutoEncoder+MLP models on various datasets

    draw_confusion_matrix(ADNI, 'ADNI', need_encode=False, Big=False)
    draw_confusion_matrix(PPMI, 'PPMI', need_encode=False, Big=False)
    draw_confusion_matrix(ADNI_fMRI, 'ADNI_fMRI', need_encode=False, Big=True)
    draw_confusion_matrix(OCD_fMRI, 'OCD_fMRI', need_encode=False, Big=True)
    draw_confusion_matrix(FTP_fMRI, 'FTP_fMRI', need_encode=False, Big=True)


# Run the test_all_datasets function
if __name__ == '__main__':
    ch = input('ready to Train(y to Train.other to avoid Train)')
    OCD_fMRI = reshapeDataDict(OCD_fMRI)
    ADNI_fMRI = reshapeDataDict(ADNI_fMRI)
    FTP_fMRI = reshapeDataDict(FTP_fMRI)
    if (ch == 'y'):
        for _ in range(10):
            train_all_datasets()
    test_all_datasets()
    # draw_confusion_matrix(ADNI, 'ADNI', need_encode=False)
    # draw_confusion_matrix(OCD_fMRI, 'OCD_fMRI', need_encode=False, Big=True)
