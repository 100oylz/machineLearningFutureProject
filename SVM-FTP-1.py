import random
import time

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import utils as utils


class SVM:

    def __init__(self, C=1.0, kernel='gaussian', tol=1e-3, max_iter=100):
        self.C = 1.0
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

    def kernel(self,x1,x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel =='poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        elif self._kernel == 'sigmoid':
            return np.tanh(0.15 * np.dot(x1, x2.T) + 1.0)
        elif self._kernel =='gaussian':
            gamma = 0.025
            distance = np.linalg.norm(x1-x2)**2
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
        num_changed_alphas = 0
        examine_all = True
        iteration = 0

        while iteration < self.max_iter and (num_changed_alphas > 0 or examine_all):
            num_changed_alphas = 0

            if examine_all:
                for i in range(self.m):
                    num_changed_alphas += self._examine_example(i)
            else:
                non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
                for i in non_bound_indices:
                    num_changed_alphas += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed_alphas == 0:
                examine_all = True

            iteration += 1

    def _examine_example(self, i1):
        y1 = self.y[i1]
        alpha1 = self.alpha[i1]
        error1 = self.errors[i1]
        r1 = error1 * y1

        if (r1 < -self.tol and alpha1 < self.C) or (r1 > self.tol and alpha1 > 0):
            if len(self.alpha[(self.alpha > 0) & (self.alpha < self.C)]) > 1:
                i2 = self._select_second_example(i1)
                if self._take_step(i1, i2):
                    return 1

            non_bound_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            np.random.shuffle(non_bound_indices)
            for i2 in non_bound_indices:
                if self._take_step(i1, i2):
                    return 1

            random_indices = np.random.permutation(self.m)
            for i2 in random_indices:
                if self._take_step(i1, i2):
                    return 1

        return 0

    def _select_second_example(self, i1):
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

    def _take_step(self, i1, i2):
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

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False

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
            # f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            # f2 = y2 * (E2 + self.b) - s * alpha1 * k12
            # L1 = alpha1
            a1 = alpha1 + y1 * y2 * (alpha2 - a2)
            b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
            b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22

            if 0 < a1 < self.C:
                self.b = b1
            elif 0 < a2 < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            self.alpha[i1] = a1
            self.alpha[i2] = a2
            self.errors[i1] = self._decision_function_single(i1) - y[i1]
            self.errors[i2] = self._decision_function_single(i2) - y[i2]
            # self.update_errors(i1, i2)

            return True
        if eta <= 0:
            return False

        # if eta > 0:
        #     a2 = alpha2 + y2 * (E1 - E2) / eta
        #
        #     if a2 < L:
        #         a2 = L
        #     elif a2 > H:
        #         a2 = H
        # else:  #更新参数
        #     # f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
        #     # f2 = y2 * (E2 + self.b) - s * alpha1 * k12
        #     # L1 = alpha1
        #     a1 = alpha1
        #     a2 = alpha2
        #     L1 = alpha1 + s * (alpha2 - L)
        #     H1 = alpha1 + s * (alpha2 - H)
        #     f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
        #     f2 = y2 * (E2 + self.b) - alpha2 * k22 - s * alpha1 * k12
        #     b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
        #     b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
        #
        #     self.alpha[i1] = a1
        #     self.alpha[i2] = a2
        #
        #     if 0 < a1 < self.C:
        #         self.b = b1
        #     elif 0 < a2 < self.C:
        #         self.b = b2
        #     else:
        #         self.b = (b1 + b2) / 2
        #
        #         self.errors[i1] = self._decision_function_single(i1) - y[i1]
        #         self.errors[i2] = self._decision_function_single(i2) - y[i2]
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

def n_components_analysis(n, x_train, y_train, x_test, y_test):
    # 记录开始时间
    start = time.time()

    # PCA降维实现
    pca = PCA(n_components=n)
    print("特征降维，传递的参数:{}".format(n))

    pca.fit(x_train)  # 这是在训练前先进行标准化、归一化，来获取一些必要参数
    x_train_pca = pca.transform(x_train)  # 测试集和训练集用同样的参数，所以刚刚fit()都是x_train
    x_test_pca = pca.transform(x_test)

    # 机器学习 - SVC
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
    #print(x_test.shape)
    #print(y_test.shape)
    rate = svm.score(x_test, y_test)
    print(rate)

if __name__ == '__main__':

#-------------------------------------测试数据集--------------------------------------------
    ADNI = utils.loadMatFile('dataset/dataset/ADNI.mat')
    ADNI_fMRI = utils.loadMatFile('dataset/dataset/ADNI_90_120_fMRI.mat')
    FTP_fMRI = utils.loadMatFile('dataset/dataset/FTD_90_200_fMRI.mat')
    OCD_fMRI = utils.loadMatFile('dataset/dataset/OCD_90_200_fMRI.mat')
    PPMI = utils.loadMatFile('dataset/dataset/PPMI.mat')


#----------------------------------降维处理数据集------------------------------------------
    data, label, labelMap = utils.generateDataSet(FTP_fMRI)
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
#-----------------------------------------------------------------------------------------------------------
#----------------------------------------------------过采样处理--------------------------------------------------
    # FTP_fMRI = utils.reshapeDataDict(FTP_fMRI)
    # x_train, y_train, valid_data, valid_label, x_test, y_test, labelMap=utils.train_valid_test_split(FTP_fMRI,42)

    for i in range(5):
        pca = PCA(n_components=0.7857)
        X_pca = pca.fit_transform(X)
        #print(np.array(X_pca).shape)
        random.seed(42)
        x_train, x_rest, y_train, y_rest = train_test_split(X_pca, y, test_size=0.40,random_state=i) #降维处理方式
        x_test, x_v, y_test, y_v = train_test_split(x_rest, y_rest, test_size=0.50,random_state=i)
        #print(np.array(x_train).shape)
        #print('x_test',np.array(x_test).shape)

#-----------------------------------------训练数据集-------------------------------------------------------
        print(i,train(x_train,y_train,x_v,y_v))