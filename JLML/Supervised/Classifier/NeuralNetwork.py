import numpy as np
from JLML.Supervised.Classifier.LogisticRegression import BinaryLogisticRegression, ModelNotTrainedError


class FeedForwardNeuralNetwork(object):

    DEFAULT = [-1]

    def __init__(self,
                 learning_rate=0.05,
                 max_steps=1000,
                 threshold=float('-inf'),
                 regularization_rate=0.0,
                 random_initialization_factor=1):

        self.has_trained_flag = False
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.threshold = threshold
        self.regularization_rate = regularization_rate
        self.random_initialization_factor = random_initialization_factor

        self.label_list = None
        self.label_table = None
        self.label_type = None
        self.output_units = None
        self.hidden_layers = None
        self.output_layer = None

    def __build_randomized_layer(self, row, col):
        """
        construct a randomized matrix given the row and col size
        each cell's value would be in range [-self.random_initialization_factor, self.random_initialization_factor]
        each cell's value represents the weight of a neuron
        :param row: int
        :param col: int
        :return: np.matrix
        """
        layer = np.random.rand(row, col)
        return layer * (2 * self.random_initialization_factor) - self.random_initialization_factor

    def __construct_architecture(self, features, labels, hidden_layer):
        """
        build the hidden layer and output layer
        :param features: np.matrix
        :param labels: np.matrix
        :param hidden_layer: list<int>
        :return: None
        """
        data_size, feature_size = features.shape
        self.input_units = feature_size  # without bias unit

        self.label_list = list(set([row.item() for row in labels]))
        # map labels to index of output vector
        self.label_table = {label: index for index, label in enumerate(self.label_list)}

        self.output_units = len(self.label_list)
        self.hidden_layers = []
        self.label_type = labels.dtype

        if hidden_layer == FeedForwardNeuralNetwork.DEFAULT:
            # default is 1 hidden layer with the same units as the input layer
            # matrix size is input unit x (input unit + 1). Plus one for bias unit
            layer = self.__build_randomized_layer(row=self.input_units, col=self.input_units + 1)
            self.hidden_layers.append(layer)
        else:
            for index, units in enumerate(hidden_layer):
                if index == 0:
                    layer = self.__build_randomized_layer(row=units, col=self.input_units + 1)
                    self.hidden_layers.append(layer)
                else:
                    layer = self.__build_randomized_layer(row=units, col=hidden_layer[index - 1] + 1)
                    self.hidden_layers.append(layer)

        if self.hidden_layers:
            number_of_last_hidden_layer_rows = self.hidden_layers[-1].shape[0]
            self.output_layer = self.__build_randomized_layer(row=self.output_units, col=number_of_last_hidden_layer_rows + 1)
        # handle 0 hidden layer case
        else:
            self.output_layer = self.__build_randomized_layer(row=self.output_units, col=self.input_units + 1)

    def __forward_propagation(self, xs):
        """
        activate the neural network and record the result from each layer
        :param xs: np.matrix (vector of input)
        :return: list<np.matrix>
        """
        layer_result_list = []
        vectorized_sigmoid = np.vectorize(BinaryLogisticRegression.sigmoid)
        xs = np.concatenate(([[1]], xs), axis=0)
        layer_result_list.append(xs)
        all_layers = self.hidden_layers + [self.output_layer]
        for layer in all_layers:
            xs = vectorized_sigmoid(layer * xs)
            xs = np.concatenate(([[1]], xs), axis=0)
            layer_result_list.append(xs)
        # output layer doesn't need to account for bias unit
        layer_result_list[-1] = np.delete(layer_result_list[-1], 0, axis=0)
        return layer_result_list

    def __back_propagation(self, layer_result_list, label):
        """
        activate the neural network backward to calculate the error from the correct output
        :param layer_result_list: list<np.matrix> (the results of each layer from forward propagation)
        :param label: np.matrix (a vector of correct label)
        :return: list<np.matrix>
        """
        error_list = []
        layer_error = layer_result_list[-1] - label
        error_list.append(layer_error)

        all_layers = self.hidden_layers + [self.output_layer]
        for layer_result, layer in zip(reversed(layer_result_list[1:-1]), reversed(all_layers)):
            answer = layer.transpose() * layer_error
            layer_error = np.multiply(answer, np.multiply(layer_result, 1 - layer_result))
            error_list.append(layer_error)
            # don't include bias unit
            layer_error = np.delete(layer_error, 0, axis=0)
        error_list.reverse()
        return error_list

    def __transform_label(self, label):
        """
        map the label to a output matrix
        for instance, if labels are [red, blue, green]
        blue would map to
                         [[0,
                           1,
                           0]]
        :param label: instance of self.label_type
        :return: np.matrix
        """
        vectorized_label = [0] * self.output_units
        if label in self.label_table:
            index = self.label_table[label]
            vectorized_label[index] = 1
        return np.matrix(vectorized_label).transpose()

    def __regularization(self, accumulated_changes):
        """
        regularize the accumulated changes
        :param accumulated_changes: list<np.matrix>
        :return: list<np.matrix>
        """
        if not self.regularization_rate:
            return accumulated_changes
        all_layers = self.hidden_layers + [self.output_layer]
        for l in xrange(len(accumulated_changes)):
            rows, cols = accumulated_changes[l].shape
            for i in xrange(rows):
                for j in xrange(cols):
                    if j != 0:
                        accumulated_changes[l][i, j] += self.regularization_rate * all_layers[l][i, j]
        return accumulated_changes

    def __cost_function(self, features, labels):
        """
        calculate the cost for each neuron weight
        :param features: np.matrix
        :param labels: np.matrix
        :return: list<np.matrix>
        """
        data_size, feature_size = features.shape
        accumulated_changes = []
        for m in xrange(data_size):
            row = features[m, :]
            label = labels[m, 0].item()
            transformed_label_vector = self.__transform_label(label)
            layer_result_list = self.__forward_propagation(row.transpose())
            error_list = self.__back_propagation(layer_result_list, transformed_label_vector)

            for index, (error, layer) in enumerate(zip(error_list, layer_result_list)):
                result = error * layer.transpose()
                if len(accumulated_changes) == index:
                    accumulated_changes.append(result)
                else:
                    accumulated_changes[index] += result

        for index, changes in enumerate(accumulated_changes):
            accumulated_changes[index] /= float(data_size)

        accumulated_changes = self.__regularization(accumulated_changes)
        for index in xrange(len(accumulated_changes)-1):
            accumulated_changes[index] = np.delete(accumulated_changes[index], 0, axis=0)
        return accumulated_changes

    def __convergence_test(self, cost):
        """
        check if the gradient descent has converged according to preset threshold
        :param cost: float
        :return: bool
        """
        return cost <= self.threshold

    def __gradient_descent(self, features, labels):
        """
        use gradient descent algorithm to minimize the network error
        :param features: np.matrix
        :param labels: np.matrix
        :return: None
        """

        previous_learning_cost = None
        cost_diff = float('inf')
        steps = 0

        while steps < self.max_steps and not self.__convergence_test(cost_diff):
            costs = self.__cost_function(features, labels)
            current_learning_cost = np.max([np.max(cost)for cost in costs])
            for index in xrange(len(self.hidden_layers)):
                self.hidden_layers[index] -= costs[index]
            self.output_layer -= costs[-1]

            if previous_learning_cost:
                cost_diff = abs(previous_learning_cost - current_learning_cost)
            previous_learning_cost = current_learning_cost

            steps += 1

    def train(self, features, labels, hidden_layers=DEFAULT):
        """
        construct and train the neural network given data and correct labels
        each integer in hidden_layers represents how many neuron in each layer
        if hidden_layers = [3,4,2],
        the neural network architecture would 3 hidden layers with 3, 4 and 2 neurons respectively
        the default is one layer with the same number of neurons as input layer excluding bias unit
        :param features: np.matrix
        :param labels: np.matrix
        :param hidden_layers: list<int>
        :return:
        """
        self.__construct_architecture(features, labels, hidden_layers)
        self.__gradient_descent(features, labels)
        self.has_trained_flag = True

    def predict(self, features):
        """
        activate the neural network on new data
        :param features: np.matrix
        :return: np.matrix
        """
        if not self.has_trained_flag:
            raise ModelNotTrainedError('Neural Network model is used before being trained.')
        data_size, feature_size = features.shape
        prediction = []
        for m in xrange(data_size):
            row = features[m, :]
            result = self.__forward_propagation(xs=row.transpose())
            list_representation = sum(result[-1][:, 0].tolist(), [])
            max_index, max_value = max(enumerate(list_representation), key=lambda (index, value): value)
            prediction.append(self.label_list[max_index])
        return np.matrix(prediction).transpose()


def test():
    net = FeedForwardNeuralNetwork()

    xor_features = np.matrix('0 0; 0 1; 1 0; 1 1')
    xor_labels = np.matrix('0; 1; 1; 0')
    net.train(xor_features, xor_labels, hidden_layers=[3])
    print net.predict(xor_features)

    nand_features = np.matrix('0 0; 0 1; 1 0; 1 1')
    nand_labels = np.matrix('1; 1; 1; 0')
    net.train(nand_features, nand_labels, hidden_layers=[3, 2])
    print net.predict(nand_features)


if __name__ == '__main__':
    test()
