import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def mnist_data():
    data = (pd.read_csv('mnist_test.csv')).values
    x = data[:, 1:]
    y = data[:, 0]
    target = np.where(y == 1)[0]
    y[target] = 1
    not_target = np.where(y != 1)[0]
    y[not_target] = 0

    N = len(target)
    not_target = np.random.choice(not_target, N)

    idx = np.append(target, not_target)
    data_x = x[idx, :]
    data_y = y[idx]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                        test_size=0.3,
                                                        random_state=42)

    return train_x, train_y, test_x, test_y


class my_logistic:
    '''
    # of train samples: 1589
    # of positive sampls: 789
    # of test samples: 681

    random state = 42
    Acc is: 0.9706314243759178
    Running duration: 335.45216 seconds

    random state = 32
    Acc is: 0.9412628487518355
    Running duration: 343.776519 seconds
    '''
    def __init__(self, train_x, train_y, step=1e-3):
        self.step = step
        self.X = train_x
        self.Y = np.reshape(train_y, [-1, 1])
        self.N, self.D = np.shape(self.X)
        self.theta = np.zeros(shape=(self.D, 1))

    def initializer(self):
        self.theta = np.random.rand(self.D, 1)

    def optimizer(self, max_iter, threshold):
        for j in range(self.D):
            for it in range(max_iter):
                old_theta = self.theta[j, 0]

                bracket = self.sum_vector(j)
                new_theta = old_theta + self.step*bracket
                self.theta[j, 0] = new_theta

                diff = np.abs(old_theta - new_theta)
                if diff < threshold:
                    break

    def train(self, max_iter=1e3, threshold=1e-3):
        self.initializer()
        self.optimizer(np.int32(max_iter), threshold)

    def sum_vector(self, j):
        return np.dot(self.X[:, j], self.Y - self.prob_y(self.X))

    def prob_y(self, x):
        exponential = np.exp(-np.dot(x, self.theta))
        return 1/(1 + exponential)

    def predict(self, test):
        test_result = self.prob_y(test)
        pos = np.where(test_result >= 0.5)[0]
        pred_y = np.zeros(shape=np.shape(test_result))
        pred_y[pos] = 1
        return pred_y


train_x, train_y, test_x, test_y = mnist_data()
model = my_logistic(train_x, train_y)

tic = time.clock()
model.train(max_iter=1e2, threshold=1e-2)
toc = time.clock()

pred_y = model.predict(test_x)
print("Acc is: {}".format(accuracy_score(np.reshape(test_y, [-1, 1]), pred_y)))
print("Training: {} seconds".format(toc-tic))
