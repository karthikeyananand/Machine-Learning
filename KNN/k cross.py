import struct
import numpy as np
from numpy import float64
import pandas as pd
import math
import operator
from random import shuffle


def dataexp(f_name):
    with open(f_name, 'rb') as f:
        zero, d_type, dims = struct.unpack('>HBB', f.read(4))
        data = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(data)


tr_img = dataexp("train-images.idx3-ubyte")
trainLb = dataexp("train-labels.idx1-ubyte")
traindat = np.reshape(tr_img, (60000, 28 * 28))

te_img = dataexp("t10k-images.idx3-ubyte")
testLb = dataexp("t10k-labels.idx1-ubyte")
testdat = np.reshape(te_img, (10000, 28 * 28))

traindat = np.array(traindat, dtype=float64)
testdat = np.array(testdat, dtype=float64)


def euclidean_distance(testing, training):
    dist = np.linalg.norm(np.array(testing) - np.array(training))
    return math.sqrt(dist)


def getneighb(tr_set, testins, k, trainlab):
    distance = []
    for i in range(int(len(tr_set))):
        dis = euclidean_distance(testins, tr_set[i])
        distance.append((tr_set[i], trainlab[i], dis))
    distance.sort(key=operator.itemgetter(2))
    neigh = []

    for i in range(k):
       neigh.append(distance[i][1])
    neigh = np.array(neigh)
    neigh = np.reshape(neigh, (10, 1))
    print(neigh)
    return neigh


def near_k(neigh_cnt):
    return sorted(neigh_cnt.items(), key=operator.itemgetter(1), reverse=True)


def major(node):
    node_c = {}

    for r in range(len(node)):

        majval = node[r, :]
        print(majval)

        if majval in tuple(node_c):
            node_c[tuple(majval)] += 1

        else:

            node_c[tuple(majval)] = 1
            print(node_c)

        total = near_k(node_c)
    return total[0][0]


def knn(tra_da, tra_la, te_da, te_la):
    k = [1, 2]  # run for 10 values of k

    for i in k:
        corr = 0
        pred = []
        for x in range(int(len(te_da))):
            neighb = np.array(getneighb(tra_da, te_da[x], k[i - 1], tra_la))
            label_val = major(neighb)
            pred = label_val
            y_act= te_la[x]
            if label_val == te_la[x]:
                corr += 1

        accuracy = corr / (int(len(te_da)))
        print(accuracy * 100)
        confusion = pd.crosstab(y_act, pred)
        print(confusion)


y = [i for i in range(0, 60000)]

shuffle(y)
print(len(y))

train_d = np.array(traindat[y])

label_d = np.array(trainLb[y])

print(train_d.shape)
print("label", label_d.shape)


for i in range(10):
    te_set = train_d[(i * 6000):(6000 * (i + 1)), :]
    print(te_set.shape)
    label_d = np.reshape(label_d, (60000, 1))
    testset_lbl = label_d[(i * 6000):(6000 * (i + 1)), :]
    print(testset_lbl.shape)
    print(type(te_set))
    trainset = []
    trainset = np.empty((1200, 784))
    trainset_lbl = np.empty((1200, 1))

    for j in range(10):
        if j != i:
            part = train_d[(j * 6000):(6000 * j + 1200), :]
            partlbl = label_d[(j * 6000):(6000 * j + 1200), :]
            trainset = np.append(trainset, part, 0)
            trainset_lbl = np.append(trainset_lbl, partlbl, 0)
    trainset = trainset[1200:12000, :]
    trainset_lbl = trainset_lbl[1200:12000, :]
    print(trainset.shape)
    print(trainset_lbl.shape)
    knn(trainset, te_set, trainset_lbl, testset_lbl)



