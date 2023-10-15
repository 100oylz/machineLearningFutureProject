import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils as utils


class RFClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=self.max_depth)

            # 进行随机有取样放回
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            x_s = X[sample_indices]
            y_s = y[sample_indices]

            # 训练决策树模型
            model.fit(x_s, y_s)

            # 将训练好的决策树模型添加到随机森林中
            self.models.append(model)

    def predict(self, X):

        predictions = np.array([model.predict(X) for model in self.models])

        return np.mean(predictions, axis=0).astype(int)

def acc_list(predictions,y_test):
    num_correct = 0
    for i in range(len(predictions)):
        if (predictions[i] == y_test[i]):
            num_correct = num_correct+1
    return num_correct/len(predictions)

if __name__ == '__main__':

#-------------------------------------测试数据集--------------------------------------------
    ADNI = utils.loadMatFile('dataset/dataset/ADNI.mat')
    ADNI_fMRI = utils.loadMatFile('dataset/dataset/ADNI_90_120_fMRI.mat')
    FTP_fMRI = utils.loadMatFile('dataset/dataset/FTD_90_200_fMRI.mat')
    OCD_fMRI = utils.loadMatFile('dataset/dataset/OCD_90_200_fMRI.mat')
    PPMI = utils.loadMatFile('dataset/dataset/PPMI.mat')


#----------------------------------降维处理数据集------------------------------------------
    data, label, labelMap = utils.generateDataSet(ADNI)
    # data = np.array(data)
    # num_samples = data.shape[0]
    # num_rows = data.shape[1]
    # num_cols = data.shape[2]
    # data = data.reshape(num_samples, num_rows * num_cols)

    X = np.array(data)
    print(X.shape)
    y = np.array(label)
    y_test = y
    print(y.shape)
    print(y)
    count=0
    for i in range(len(y)):
        if y[i] == 0:
            count=count+1
    print(count)
#----------------------------------------------------过采样处理--------------------------------------------------
    # OCD_fMRI = utils.reshapeDataDict(OCD_fMRI)
    # x_train, y_train, valid_data, valid_label, x_test, y_test, labelMap=utils.train_valid_test_split(OCD_fMRI,42)

    for i in range(20):
       pca = PCA(n_components=0.993)
       X_pca = pca.fit_transform(X)
       print('降维：',np.array(X_pca).shape)
       #X_pca = X

       #random.seed(42)
       x_train, x_rest, y_train, y_rest = train_test_split(X_pca, y, test_size=0.40,random_state=i)  # 降维处理方式
       x_test, x_v, y_test, y_v = train_test_split(x_rest, y_rest, test_size=0.50,random_state=i)

#------------------------------------调用随机森林函数--------------------------------------------
       # rf_classifier = RandomForestClassifier(n_estimators=100,random_state=20)
       # # for class_label in np.unique(y):
       # #     svm_classfier = SVM()
       # #     y_pair = np.where(y_train == class_label,1,0)
       # #
       # #     svm_classfier.fit(x_train,y_pair)
       # #     rf_classifier.estimators_.append(svm_classfier)
       # rf_classifier.fit(x_train,y_train)
       # y_pred = rf_classifier.predict(x_v)
       # print(y_pred)
       # #print(y_test)
       #
       # acc_rate = acc_list(y_pred, y_v)
       # print(i,acc_rate)

#-----------------------------------2-------------------------------------------------
       n_estimators = 200
       rf = RFClassifier(n_estimators,max_depth=10)
       rf.fit(x_train,y_train)
       y_pred1 = rf.predict(x_test)
       acc_rate1 = acc_list(y_pred1,y_test)
       print(acc_rate1)
