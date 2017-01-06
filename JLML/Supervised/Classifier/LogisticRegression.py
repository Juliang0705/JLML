import numpy as np
from JLML.Supervised.Regression.LinearRegression import LinearRegression, ModelNotTrainedError
from collections import Counter


class NotBinaryLabelsError(Exception):
    pass


class BinaryLogisticRegression(object):

    def __init__(self,
                 learning_rate=0.05,
                 max_steps=2000,
                 threshold=0.0001,
                 regularization_rate=0.0):

        self.has_trained_flag = False
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.threshold = threshold
        self.regularization_rate = regularization_rate
        self.thetas = None

    @staticmethod
    def __is_binary_labels(labels):
        """
        check if input is a vector with only 1 and 0 as its value
        :param labels: np.matrix
        :return: bool
        """
        label_count = Counter(sum(labels.tolist(), []))
        return labels.shape[1] == 1 and len(label_count) <= 2 and 1 in label_count and 0 in label_count

    @staticmethod
    def sigmoid(value):
        """
        logistic function
        :param value: number
        :return: number in range [0, 1]
        """
        return 1 / (1 + np.exp(-value))

    def __hypothesis_function(self, xs):
        """
        h(x) = sigmoid( theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetan * xn )
        :param xs: np.matrix (1 x # of features)
        :return: float
        """
        value = (xs * self.thetas.transpose()).item()
        return BinaryLogisticRegression.sigmoid(value)

    def __cost_function(self, features, labels, index):
        """
        calculate the learning cost
        :param features: np.matrix (data size x # of features)
        :param labels: np.matrix (data size x 1)
        :param index: int, index of current row
        :return: float
        """
        data_size, feature_size = features.shape
        total = 0
        for m in xrange(data_size):
            row = features[m, :]
            total += (self.__hypothesis_function(row) - labels[m, 0]) * features[m, index]
        regularization = self.regularization_rate * self.thetas[0, index]
        return (self.learning_rate * total + regularization) / float(data_size)

    def __convergence_test(self, cost):
        """
        check if the gradient descent has converged according to preset threshold
        :param cost: float
        :return: bool
        """
        return cost <= self.threshold

    def train(self, features, labels):
        """
        use gradient descent algorithm to fit the data set.
        only support labels with 1 and 0 as its value
        :param features: np.matrix (data size x # of features)
        :param labels: np.matrix (data size x 1)
        :return: None
        """
        if not BinaryLogisticRegression.__is_binary_labels(labels):
            raise NotBinaryLabelsError("Labels can only have 1 or 0 as values")

        features = LinearRegression.add_ones_column(features)
        data_size, feature_size = features.shape
        self.thetas = np.zeros((1, feature_size), dtype='f')

        current_learning_cost = float('-inf')
        previous_learning_cost = None
        cost_diff = float('inf')
        steps = 0

        while steps < self.max_steps and not self.__convergence_test(cost_diff):
            updated_thetas = np.zeros((1, feature_size), dtype=features.dtype)
            for i in xrange(feature_size):
                cost = self.__cost_function(features, labels, i)
                updated_thetas[0, i] = self.thetas[0, i] - cost
                current_learning_cost = max(current_learning_cost, cost)

            self.thetas = updated_thetas

            if previous_learning_cost:
                cost_diff = abs(previous_learning_cost - current_learning_cost)
            previous_learning_cost = current_learning_cost

            current_learning_cost = float('-inf')
            steps += 1

        self.has_trained_flag = True

    def predict(self, features):
        """
        predict the labels using the trained model.
        :param features: np.matrix (data size x # of features)
        :return: np.matrix (data size x 1)
        """
        if not self.has_trained_flag:
            raise ModelNotTrainedError('Logistic regression model is used before being trained.')

        features = LinearRegression.add_ones_column(features)
        vectorized_sigmoid = np.vectorize(BinaryLogisticRegression.sigmoid)
        return vectorized_sigmoid(features * self.thetas.transpose())


class LogisticRegression(object):

    def __init__(self,
                 learning_rate=0.05,
                 max_steps=2000,
                 threshold=0.0001,
                 regularization_rate=0.0):

        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.threshold = threshold
        self.regularization_rate = regularization_rate
        self.label_set = None
        self.label_type = None
        self.classifier_list = None

    @staticmethod
    def __transform_label(labels, value):
        """
        transform the label to binary 1 and 0
        :param labels: np.matrix
        :param value: target positive type
        :return: np.matrix
        """
        transformer = np.vectorize(lambda v: 1 if v == value else 0)
        return transformer(labels)

    def train(self, features, labels):
        """
        Build a model with the data set.
        Support multi-label data set using one v.s all method
        :param features: np.matrix
        :param labels: np.matrix
        :return: None
        """
        self.label_type = labels.dtype
        self.label_set = set(sum(labels.tolist(), []))
        self.classifier_list = {}
        for label in self.label_set:
            transformed_labels = LogisticRegression.__transform_label(labels, label)
            classifier = BinaryLogisticRegression(learning_rate=self.learning_rate,
                                                  max_steps=self.max_steps,
                                                  threshold=self.threshold,
                                                  regularization_rate=self.regularization_rate)
            classifier.train(features, transformed_labels)
            self.classifier_list[label] = classifier

    def predict(self, features):
        """
        predict the labels using the trained model.
        :param features: np.matrix (data size x # of features)
        :return: np.matrix (data size x 1)
        """
        data_size, feature_size = features.shape
        prediction = np.zeros((data_size, 1), dtype=self.label_type)
        for m in xrange(data_size):
            row = features[m, :]
            max_label = None
            max_label_probability = float('-inf')
            for label in self.classifier_list.keys():
                classifier = self.classifier_list[label]
                label_probability = classifier.predict(row).item()
                if label_probability > max_label_probability:
                    max_label = label
                    max_label_probability = label_probability
            prediction[m, 0] = max_label
        return prediction


def test():

    blr = BinaryLogisticRegression(learning_rate=0.001,
                                   max_steps=3000,
                                   threshold=0.0001,
                                   regularization_rate=0.1)
    features = np.matrix([[-20, -10, -3, 1, 4, 7, 10, 11, 13, 15, 40, 50]], dtype='f').transpose()
    labels = np.matrix([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype='f').transpose()
    blr.train(features, labels)
    print blr.thetas
    test_features = np.matrix([[-5, 0, 2, 4, 1, 14, 100]], dtype='f').transpose()
    print blr.predict(test_features)

    lr = LogisticRegression(learning_rate=0.01,
                            max_steps=2000,
                            threshold=0.001,
                            regularization_rate=0.1)
    features = np.matrix([[-20, -10, -3, 1, 4, 7, 10, 11, 13, 15, 40, 50]], dtype='f').transpose()
    labels = np.matrix([['red', 'red', 'red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']]).transpose()
    test_features = np.matrix([[-5, 0, 2, 4, 1, 14, 100]], dtype='f').transpose()
    lr.train(features, labels)
    print lr.predict(test_features)

if __name__ == '__main__':
    test()
