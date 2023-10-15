import time
# import AutoEncoder
from itertools import combinations

import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import utils as utils


class SVM:
    #初始化参数
    def __init__(self, C=1.0, kernel='gaussian', tol=1e-3, max_iter=1000):
        self.C = 1
        self._kernel = kernel
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.alpha = np.ones(self.m)
        self.b = 0.0
        self.errors = self.decision_function(X) - y
        self._optimize()

    def kernel(self,x1,x2):            #四种核函数
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel =='poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        elif self._kernel == 'sigmoid':
            return np.tanh(0.15 * np.dot(x1, x2.T) + 1.0)
        elif self._kernel =='gaussian':
            gamma = 0.0019
            distance = np.linalg.norm(x1-x2)**2
            kernel_value = np.exp(-gamma * distance)
            return kernel_value
        elif self._kernel == 'laplacian':
            gamma = 0.30
            distance = np.linalg.norm(x1 - x2)
            kernel_value = np.exp(-gamma * distance)
            return kernel_value


    def decision_function(self, X):
        n_samples = self.m
        decision_values = np.zeros(n_samples)

        for i in range(n_samples):
            decision_values[i] = self._decision_function_single(i)

        return decision_values

    def _decision_function_single(self, i):
        result = 0.0

        for j in range(self.m):
            result += self.alpha[j] * self.y[j] * self.kernel(self.X[i], self.X[j])

        return result + self.b



    def _optimize(self):
        num_changed_alphas = 0   #记录在当前迭代中发生改变的拉格朗日乘子的数量
        examine_all = True       #是否要遍历所有样本进行检查
        iteration = 0            #迭代次数

    #开始循环：迭代次数未到最大迭代次数，上一次迭代中有乘子发生改变。
        while iteration < self.max_iter and (num_changed_alphas > 0 or examine_all):
            num_changed_alphas = 0

            if examine_all:
                for i in range(self.m):#对索引为i的样本进行检查，并将返回的改变的拉格朗日乘子数量累加
                    num_changed_alphas += self._examine_example(i)
            else:
                non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]#找到满足条件的非边界样本的索引
                for i in non_bound_indices:  #检查非边界样本
                    num_changed_alphas += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed_alphas == 0:   #无拉格朗日乘子变化，则在下一次迭代当中需要遍历所有样本
                examine_all = True

            iteration += 1

    def _examine_example(self, i1):
        y1 = self.y[i1]            #类别标签
        alpha1 = self.alpha[i1]    #拉格朗日乘子
        error1 = self.errors[i1]   #预测错误
        r1 = error1 * y1
#判别条件1：函数间隔小于负容差并且拉格朗日乘子小于惩罚参数 self.C。
#判别条件2：函数间隔大于容差并且拉格朗日乘子大于 0。
#采用SMO方法来优化选定的两个拉格朗日乘子
        if (r1 < -self.tol and alpha1 < self.C) or (r1 > self.tol and alpha1 > 0):
            if len(self.alpha[(self.alpha > 0) & (self.alpha < self.C)]) > 1:
                i2 = self._select_second_example(i1) #选择第二个索引
                if self._take_step(i1, i2):
                    return 1
            #非边界样本索引
            non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            np.random.shuffle(non_bound_indices) #打乱
            for i2 in non_bound_indices:         #遍历优化
                if self._take_step(i1, i2):
                    return 1

            random_indices = np.random.permutation(self.m)  #样本索引的随机排序
            for i2 in random_indices:
                if self._take_step(i1, i2):
                    return 1

        return 0

    def _select_second_example(self, i1):#更加最大误差选择第二个样本
        error1 = self.errors[i1]
        max_delta_error = 0
        i2 = -1

        for i in range(self.m):
            if i == i1:
                continue

            error2 = self.errors[i]
            delta_error = abs(error1 - error2)

            if delta_error > max_delta_error:
                max_delta_error = delta_error
                i2 = i

        if i2 >= 0:
            return i2

        return np.random.randint(0, self.m)

    def _take_step(self, i1, i2):   #优化
        if i1 == i2:
            return False

        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        X1 = self.X[i1]
        X2 = self.X[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2
        #给出乘子的更新边界
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False
        #计算两个乘子之间的核函数
        k11 = self.kernel(X1, X1)
        k22 = self.kernel(X2, X2)
        k12 = self.kernel(X1, X2)
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta

            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:  #更新参数
            # f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            # f2 = y2 * (E2 + self.b) - s * alpha1 * k12
            # L1 = alpha1
            a1 = alpha1
            a2 = alpha2
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - alpha2 * k22 - s * alpha1 * k12
            b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
            b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22

            self.alpha[i1] = a1
            self.alpha[i2] = a2

            if 0 < a1 < self.C:
                self.b = b1
            elif 0 < a2 < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

                self.errors[i1] = self._decision_function_single(i1) - y[i1]
                self.errors[i2] = self._decision_function_single(i2) - y[i2]
                #self.update_errors(i1, i2)

            return True

        # if eta <= 0:
        #     return False
        #
        # if eta > 0:
        #     a2 = alpha2 + y2 * (E1 - E2) / eta
        #
        #     if a2 < L:
        #         a2 = L
        #     elif a2 > H:
        #         a2 = H
        #     # f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
        #     # f2 = y2 * (E2 + self.b) - s * alpha1 * k12
        #     # L1 = alpha1
        #     a1 = alpha1 + y1 * y2 * (alpha2 - a2)
        #     b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
        #     b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
        #
        #     if 0 < a1 < self.C:
        #         self.b = b1
        #     elif 0 < a2 < self.C:
        #         self.b = b2
        #     else:
        #         self.b = (b1 + b2) / 2
        #
        #     self.alpha[i1] = a1
        #     self.alpha[i2] = a2
        #     self.errors[i1] = self._decision_function_single(i1) - y[i1]
        #     self.errors[i2] = self._decision_function_single(i2) - y[i2]
        #     # self.update_errors(i1, i2)
        #
        #     return True

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else 0

    def score(self,X_test,y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        #print('正确个数：%s'right_count)
        return right_count/len(X_test)

    def _weight(self):

        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w
#测试最佳降维参数
def n_components_analysis(n, x_train, y_train, x_test, y_test):
    # 记录开始时间
    start = time.time()
    # PCA降维实现
    pca = PCA(n_components=n)
    print("特征降维，传递的参数:{}".format(n))

    pca.fit(x_train)  # 这是在训练前先进行标准化、归一化，来获取一些必要参数
    x_train_pca = pca.transform(x_train)  # 测试集和训练集用同样的参数，所以刚刚fit()都是x_train
    x_test_pca = pca.transform(x_test)

    print("开始使用svc进行训练")
    ss = svm.SVC()
    ss.fit(x_train_pca, y_train)

    # 获取准确率
    accuracy = ss.score(x_test_pca, y_test)

    # 记录结束时间
    end = time.time()

    print("准确率：{}, 花费时间：{}, 保留的特征:{}".format(accuracy, int(end-start), pca.n_components_))
    return accuracy

def train(x_train,y_train,x_test,y_test):
    svm = SVM()
    svm.fit(x_train, y_train)
    rate = svm.score(x_test, y_test)
    print(rate)

def acc_list(predictions,y_test):
    num_correct = 0
    for i in range(len(predictions)):
        if (predictions[i] == y_test[i]):
            num_correct = num_correct+1
    return num_correct/len(predictions)

class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = []

    def fit(self, X, y):
        unique_labels = np.unique(y)

        for label in unique_labels:
            classifier = self.base_classifier()
            binary_labels = np.where(y == label, 1, 0)
            classifier.fit(X, binary_labels)
            self.classifiers.append(classifier)

    def predict(self, X):
        predictions = []

        for classifier in self.classifiers:
            pred = classifier.predict(X)
            predictions.append(pred)

        y_pred = np.array(predictions).T
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

# 创建SVM分类器对象
def create_svm_classifier():
    return SVM(kernel='rbf')

if __name__ == '__main__':

#-------------------------------------测试数据集--------------------------------------------
    ADNI = utils.loadMatFile('dataset/dataset/ADNI.mat')
    ADNI_fMRI = utils.loadMatFile('dataset/dataset/ADNI_90_120_fMRI.mat')
    FTP_fMRI = utils.loadMatFile('dataset/dataset/FTD_90_200_fMRI.mat')
    OCD_fMRI = utils.loadMatFile('dataset/dataset/OCD_90_200_fMRI.mat')
    PPMI = utils.loadMatFile('dataset/dataset/PPMI.mat')


#----------------------------------降维处理数据集------------------------------------------
    data, label, labelMap = utils.generateDataSet(ADNI_fMRI)
    data = np.array(data)
    num_samples = data.shape[0]
    num_rows = data.shape[1]
    num_cols = data.shape[2]
    data = data.reshape(num_samples, num_rows * num_cols)

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

    for q in range(10):
       pca = PCA(n_components=0.79)
       X_pca = pca.fit_transform(X)
       print('降维：',np.array(X_pca).shape)
       #X_pca = X

       #random.seed(42)
       #x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.11) #降维处理方式
       #print(np.array(x_train).shape)
       #print('x_test',np.array(x_test).shape)
       #random.seed(42)
       x_train, x_rest, y_train, y_rest = train_test_split(X_pca, y, test_size=0.40,random_state=q)  # 降维处理方式
       x_test, x_v, y_test, y_v = train_test_split(x_rest, y_rest, test_size=0.50,random_state=q)


#-----------------------------------------one-to-one-------------------------------------------------------
   # 对类别标签进行独热编码
       n_classes = 4
       y_encoded = np.eye(n_classes)[y_train]
       #y_encoded = np.eye(n_classes)[y]

    # 2. 训练分类器
       classifiers = []
       for i, j in combinations(range(n_classes), 2):
        # 提取类别i和类别j的样本
           X_pair = x_train[(y_train == i) | (y_train == j)]
        #print("X_pair:", X_pair.shape)
           y_pair = y_encoded[(y_train == i) | (y_train == j)]
           #print(y_pair)
        #print("Y_pair:", y_pair.shape)

           # X_pair = X_pca[(y == i) | (y == j)]
           #
           # y_pair = y_encoded[(y == i) | (y == j)]

        # 将类别i的标签设置为0，类别j的标签设置为1
           y_pair = np.where(y_pair[:, i] == 1, 0, 1)
           # for l in range(len(y_pair)):
           #     if y_pair[l] == i:
           #         y_pair[l] = 0
           #     else:
           #         y_pair[l] = 1
           #print(y_pair)
        #print(y_pair)

        # 使用已实现的二分类SVM分类器进行训练
           svm = SVM()
           svm.fit(X_pair, y_pair)

        # 保存分类器模型
           classifiers.append((i, j, svm))

    #print('classifiers:',classifiers)
    # 3. 预测和投票
       predictions = []
       for sample in x_v:
           votes = np.zeros(n_classes)
           for i, j, svm in classifiers:
            # 对每对分类器进行预测
               prediction = svm.predict(sample)

            # 根据预测结果进行投票
               if prediction == 0:
                   votes[i] += 1
               elif prediction == 1:
                   votes[j] += 1

        # 根据投票结果确定最终预测类别
           #print('vote:',votes[0],votes[1],votes[2],votes[3])
           final_prediction = np.argmax(votes)
           predictions.append(final_prediction)
       print('predictions:',predictions)
       #print(np.array(predictions).shape)
       #print(y_test.shape)
       print(y_v)

       acc_rate = acc_list(predictions,y_v)
       print(q,acc_rate)

#------------------------------------------one-to-res---------------------------------------------------------
       ovr_classifier = OneVsRestClassifier(create_svm_classifier)

       #ovr_classifier.fit(x_train, y_train)

       #y_pred = ovr_classifier.predict(x_test)

       svm_classifier = SVC(kernel='rbf')

       #ovr_classifier = OneVsRestClassifier(svm_classifier)
       ovr_classifier = svm_classifier

       ovr_classifier.fit(x_train, y_train)

       y_pred = ovr_classifier.predict(x_v)
       print(y_pred)

       acc_rate = acc_list(y_pred, y_v)
       print(q, acc_rate)