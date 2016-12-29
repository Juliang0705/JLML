import numpy as np


class ModelNotTrainedError(Exception):
    pass


class LinearRegression(object):
    default = 0
    gradient_descent = 1
    normal_equation = 2

    def __init__(self,
                 learning_rate=0.05,
                 learning_method=default,
                 max_steps=1000,
                 threshold=0.001,
                 enable_feature_scale=False):

        self.has_trained_flag = False
        self.learning_rate = learning_rate
        self.learning_method = learning_method
        self.max_steps = max_steps
        self.threshold = threshold
        self.enable_feature_scale = enable_feature_scale
        self.thetas = None

    @staticmethod
    def add_ones_column(features):
        """
        add a column of 1s to the matrix in the left most position
        :param features: np.matrix (data size x # of features)
        :return: np.matrix (data size x # of features)
        """
        data_size, feature_size = features.shape
        x0 = np.matrix([1] * data_size).transpose()
        features = np.concatenate((x0, features), axis=1)
        return features

    @staticmethod
    def scale_features(features):
        """
        scale the features to make the value fit between 0 and 1
        :param features: np.matrix (data size x # of features)
        :return: np.matrix (data size x # of features)
        """
        row_size, col_size = features.shape
        for col in xrange(col_size):
            column = features[:, col]
            average = np.mean(column)
            std = np.std(column)
            for row in xrange(row_size):
                new_cell = (features[row, col] - average) / std
                features[row, col] = new_cell
        return features

    def __hypothesis_function(self, xs):
        """
        describe the relationship among the data
        h(x) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetan * xn
        :param xs: np.matrix (1 x # of features)
        :return: float
        """
        return (xs * self.thetas.transpose()).item()

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
        return self.learning_rate * total / float(data_size)

    def __convergence_test(self, cost):
        """
        check if the gradient descent has converged according to preset threshold
        :param cost: float
        :return: bool
        """
        return cost <= self.threshold

    def __gradient_descent(self, features, labels):
        """
        use gradient descent algorithm to fit the data set
        good for large data set
        :param features: np.matrix (data size x # of features)
        :param labels: np.matrix (data size x 1)
        :return: None
        """

        if self.enable_feature_scale:
            features = LinearRegression.scale_features(features)
            labels = LinearRegression.scale_features(labels)

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
            else:
                previous_learning_cost = current_learning_cost

            current_learning_cost = float('-inf')
            steps += 1

    def __normal_equation(self, features, labels):
        """
        use normal equation algorithm to fit the data set.
        good for small data set
        :param features: np.matrix (data size x # of features)
        :param labels: np.matrix (data size x 1)
        :return: None
        """
        features = LinearRegression.add_ones_column(features)
        self.thetas = (np.linalg.pinv(features.transpose() * features) * features.transpose() * labels).transpose()

    def train(self, features, labels):
        """
        train a model using linear regression.
        :param features: np.matrix (data size x # of features)
        :param labels: np.matrix (data size x 1)
        :return: None
        """
        data_size, feature_size = features.shape

        if self.learning_method == LinearRegression.gradient_descent:
            self.__gradient_descent(features, labels)
        elif self.learning_method == LinearRegression.normal_equation:
            self.__normal_equation(features, labels)
        else:
            if feature_size >= 100:
                self.__gradient_descent(features, labels)
            else:
                self.__normal_equation(features, labels)
        self.has_trained_flag = True

    def predict(self, features):
        """
        predict the labels using the trained model.
        :param features: np.matrix (data size x # of features)
        :return: np.matrix (data size x 1)
        """
        if not self.has_trained_flag:
            raise ModelNotTrainedError('Linear regression model is used before being trained.')

        if self.enable_feature_scale and self.learning_method == LinearRegression.gradient_descent:
            features = LinearRegression.scale_features(features)

        features = LinearRegression.add_ones_column(features)
        return features * self.thetas.transpose()


def test():
    lr = LinearRegression(learning_method=LinearRegression.gradient_descent,
                          max_steps=2000,
                          threshold=0.001,
                          enable_feature_scale=False)

    features = np.matrix([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], dtype='f').transpose()
    labels = np.matrix([[10, 18, 26, 34, 45]], dtype='f').transpose()
    lr.train(features, labels)
    print lr.thetas
    test_features = np.matrix([[10, 20, 30, 40], [11, 22, 33, 44]], dtype='f').transpose()
    print lr.predict(test_features)

if __name__ == '__main__':
    test()


