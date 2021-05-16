from csv import reader
from math import inf
from random import shuffle


def euclid_distance(a, b):
    s = 0
    for ind in range(len(a)):
        s += (a[ind] - b[ind]) ** 2
    return s ** 0.5

def find_minimum(distances):
    min_val, min_ind = distances[0], 0
    for ind in range(1, len(distances)):
        if distances[ind] < min_val:
            min_val, min_ind = distances[ind], ind
    return min_ind

def knn(k, x_train, y_train, x_test):
    predicts = []
    for sample in x_test:
        distances = []
        for row in x_train:
            distances.append(euclid_distance(sample, row))
        votes = {}
        for _ in range(k):
            min_ind = find_minimum(distances)
            v = y_train[min_ind]
            votes[v] = votes.get(v, 0) + 1
            distances[min_ind] = inf

        max_ind, max_val = 0, 0
        for key, value in votes.items():
            if value > max_val:
                max_ind, max_val = key, value
        predicts.append(max_ind)
    return predicts


all_data = []
all_labels = []
with open('iris.csv') as csv_file:
    f1 = reader(csv_file)
    for row in f1:
        all_data.append(list(map(float, row[:4])))
        all_labels.append(row[4])


data = list(zip(all_data, all_labels))
shuffle(data)
all_data, all_labels = list(zip(*data))

train_test_split = int(0.8 * len(all_data))
x_train, y_train = all_data[:train_test_split], all_labels[:train_test_split]
x_test, y_test = all_data[train_test_split:], all_labels[train_test_split:]

predictions = knn(5, x_train, y_train, x_test)
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correct += 1
print('Accuracy: ', correct / len(predictions))

#####################################3
from sklearn.datasets import load_iris as load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


x_train, x_test, y_train, y_test = train_test_split(load().data, load().target, test_size=0.2, shuffle=True)
predictions = knn(5, x_train, y_train, x_test)
print('My accuracy:', accuracy_score(y_test, predictions))

##########################################################################################
model = KNeighborsClassifier(n_neighbors=5, algorithm='brute', weights='uniform')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print('Scikit accuracy:', accuracy_score(y_test, predictions))

##########################################################################################
from math import inf
class KNN:
    def __init__(self, k):
        self.k = k

    def euclid_distance(self, a, b):
        s = 0
        for ind in range(len(a)):
            s += (a[ind] - b[ind]) ** 2
        return s ** 0.5

    def find_minimum(self, distances):
        min_val, min_ind = distances[0], 0
        for ind in range(1, len(distances)):
            if distances[ind] < min_val:
                min_val, min_ind = distances[ind], ind
        return min_ind

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predicts = []
        for sample in x_test:
            distances = []
            for row in self.x_train:
                distances.append(self.euclid_distance(sample, row))
            votes = {}
            for _ in range(self.k):
                min_ind = self.find_minimum(distances)
                v =self.y_train[min_ind]
                votes[v] = votes.get(v, 0) + 1
                distances[min_ind] = inf
            min_ind, min_val = 0, 0
            for key, value in votes.items():
                if value > min_val:
                    min_ind, min_val = key, value
            predicts.append(min_ind)
        return predicts



from sklearn.datasets import load_wine as load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


x_train, x_test, y_train, y_test = train_test_split(load().data, load().target, test_size=0.2, shuffle=True)


model = KNN(k=5)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print('My accuracy:', accuracy_score(y_test, predictions))


