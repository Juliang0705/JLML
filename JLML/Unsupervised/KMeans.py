import numpy as np
import random


class KMeans(object):

    def __init__(self,
                 max_steps=1000,
                 threshold=0.001,
                 repeats=10):

        self.max_steps = max_steps
        self.threshold = threshold
        self.repeats = repeats
        self.centroids = None

    def __random_initialize(self, features, k):
        data_size, feature_size = features.shape
        random_row_indice = random.sample(xrange(data_size), k)
        self.centroids = [features[row_index, :] for row_index in random_row_indice]

    @staticmethod
    def __compute_vector_distance(vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def __find_closest_centroid(self, row):
        closest_centroid_index = 0
        closest_centroid_distance = float('inf')
        for index, centroid in enumerate(self.centroids):
            distance = KMeans.__compute_vector_distance(row, centroid)

            if distance <= closest_centroid_distance:
                closest_centroid_distance = distance
                closest_centroid_index = index

        return closest_centroid_index

    def __cluster_assignment(self, features):
        assignments = [[] for _ in xrange(len(self.centroids))]
        data_size, feature_size = features.shape
        for m in xrange(data_size):
            row = features[m, :]
            closest_centroid_index = self.__find_closest_centroid(row)
            assignments[closest_centroid_index].append(row)
        return assignments

    def __move_centroids(self, assignments):
        new_centroids_list = []
        for i in xrange(len(self.centroids)):
            new_centroid = sum(assignments[i]) / float(len(assignments[i]))
            new_centroids_list.append(new_centroid)
        return new_centroids_list

    def __train_once(self, features, k):
        steps = 0
        self.__random_initialize(features, k)
        distance_diff = float('inf')
        assignments = None
        while steps < self.max_steps and distance_diff > self.threshold:
            assignments = self.__cluster_assignment(features)
            new_centroid_list = self.__move_centroids(assignments)

            for old_centroid, new_centroid in zip(self.centroids, new_centroid_list):
                distance_diff = min(distance_diff, KMeans.__compute_vector_distance(old_centroid, new_centroid))

            self.centroids = new_centroid_list
            steps += 1
        return assignments, self.centroids

    @staticmethod
    def __compute_average_distance(assignments, centroids):
        total = 0
        for assignment, centroid in zip(assignments, centroids):
            for data in assignment:
                total += KMeans.__compute_vector_distance(data, centroid)
        return total / len(centroids)

    def train(self, features, k):
        repeat_count = 0
        minimal_centroids = None
        minimal_centroids_distance = float('inf')
        while repeat_count < self.repeats:
            assignments, centroids = self.__train_once(features, k)
            distance = self.__compute_average_distance(assignments, centroids)
            if distance <= minimal_centroids_distance:
                minimal_centroids_distance = distance
                minimal_centroids = centroids
            repeat_count += 1
        self.centroids = minimal_centroids
        return self.predict(features)

    def predict(self, features):
        prediction = []
        data_size, feature_size = features.shape
        for m in xrange(data_size):
            row = features[m, :]
            index = self.__find_closest_centroid(row)
            prediction.append(index)
        return np.matrix(prediction).transpose()


def test():
    kmeans = KMeans(max_steps=1000,
                    threshold=0.001,
                    repeats=5)
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    print kmeans.train(X, k=2)
    print kmeans.predict(np.matrix([[0, 0], [4, 4]]))
    print kmeans.centroids

if __name__ == '__main__':
    test()
