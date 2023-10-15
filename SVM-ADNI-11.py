import numpy as np
from scipy.optimize import minimize
# import AutoEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import utils as utils


#----------------------------------------手写---------------------------------------------------
#定义SVM模型的类
class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='linear', learning_rate=0.01, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            gamma = 1.0 / X1.shape[1]
            return np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=-1) ** 2)
        else:
            raise ValueError('Unsupported kernel type.')

    def objective(self, alpha):
        n_samples = self.X.shape[0]
        y = self.y
        K = self.compute_kernel(self.X, self.X)
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        A = y.astype(float)
        b = np.array([0.0])

        return 0.5 * np.dot(alpha, np.dot(P, alpha)) + np.dot(q, alpha)

    def gradient(self, alpha):
        n_samples = self.X.shape[0]
        y = self.y
        K = self.compute_kernel(self.X, self.X)
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)

        return np.dot(P, alpha) + q

    def train(self, X, y):
        self.X = X
        self.y = y

        n_samples = X.shape[0]
        alpha0 = np.zeros(n_samples)
        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}

        res = minimize(fun=self.objective, x0=alpha0, method='SLSQP', jac=self.gradient, bounds=bounds,
                       constraints=constraints, options={'maxiter': self.max_iter})
        alpha = res.x

        support = alpha > 1e-5
        self.alpha = alpha[support]
        self.support_vectors = X[support]
        self.support_labels = y[support]

    def predict(self, X):
        K = self.compute_kernel(X, self.support_vectors)
        pred = np.dot(K, self.alpha * self.support_labels)
        pred[pred >= 0] = 1
        pred[pred < 0] = 0

        return pred.astype(int)

class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = []

    def fit(self, X, y):
        unique_labels = np.unique(y)

        for label in unique_labels:
            classifier = self.base_classifier()
            binary_labels = np.where(y == label, 1, 0)
            classifier.train(X, binary_labels)
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
    #return  SVC(kernel='rbf')




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
    # ADNI_fMRI = utils.reshapeDataDict(ADNI_fMRI)
    # x_train, y_train, valid_data, valid_label, x_test, y_test, labelMap=utils.train_valid_test_split(ADNI_fMRI,42)

    for q in range(10):
       pca = PCA(n_components=0.993)
       X_pca = pca.fit_transform(X)
       print('降维：',np.array(X_pca).shape)
       #X_pca = X
       #random.seed(42)
       x_train, x_rest, y_train, y_rest = train_test_split(X_pca, y, test_size=0.40,random_state=q)  # 降维处理方式
       x_test, x_v, y_test, y_v = train_test_split(x_rest, y_rest, test_size=0.50,random_state=q)

       ovr_classifier = OneVsRestClassifier(create_svm_classifier)


       ovr_classifier.fit(x_train, y_train)


       y_pred = ovr_classifier.predict(x_test)
       acc_rate = acc_list(y_pred, y_test)
       #print(q, acc_rate)

       svm_classifier = SVC(kernel='rbf')
       #ovr_classifier = OneVsRestClassifier(svm_classifier)
       ovr_classifier = svm_classifier
       ovr_classifier.fit(x_train, y_train)
       y_pred1 = ovr_classifier.predict(x_v)
       #print(y_pred1)
       acc_rate = acc_list(y_pred1, y_v)
       print(q, acc_rate)

