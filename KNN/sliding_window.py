import struct
import numpy as np
from sklearn import neighbors,metrics
import matplotlib.pyplot as plt
import random
import math
import operator
from numpy import float64
from random import shuffle


def readda(f_name):
    with open(f_name, 'rb') as f:
        zero,data_type,dims=struct.unpack('>HBB',f.read(4))
        shape=tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(),dtype=np.uint8).reshape(shape)

r_train=readda("train-images.idx3-ubyte")
tra_d=np.reshape(r_train, (60000, 28 * 28))
tra_la=readda("train-labels.idx1-ubyte")


r_test=readda("t10k-images.idx3-ubyte")
te_d=np.reshape(r_test, (10000, 28 * 28))
te_lab=readda("t10k-labels.idx1-ubyte")


r_train=np.array(r_train, dtype=float64)
r_test=np.array(r_test, dtype=float64)



def euclidean_distance(test_instance, train_instance):
     dis_1 = np.linalg.norm(np.array(test_instance) - np.array(train_instance))
     return math.sqrt(dis_1)

def getNeigh(traset, testinstance, k, tra_lab):
	distances_main =[]


	for x in range(int(len(traset))):

		sample = np.array(traset[x, :])
		sample = np.pad(sample, pad_width=1, mode='constant', constant_values=1)

		s1 = np.array(sample[0:28, 0:28])
		s2 = np.array(sample[0:28, 1:29])
		s3 = np.array(sample[0:28, 2:30])
		s4 = np.array(sample[1:29, 0:28])
		s5 = np.array(sample[1:29, 1:29])
		s6 = np.array(sample[1:29, 2:30])
		s7 = np.array(sample[2:30, 0:28])
		s8 = np.array(sample[2:30, 1:29])
		s9 = np.array(sample[2:30, 2:30])

		distances = np.empty([1, 1])
		image = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9])

		for i in range(9):
			test = np.reshape(image[i], (1, 784))
			testinstance=np.reshape(testinstance,(1,784))
			dist = euclidean_distance(testinstance, test)
			dist = np.reshape(dist, (1, 1))
			distances = np.append(distances, dist, 0)

		distances = np.array(distances[1:10, :])
		min_dist = np.amin(distances)

		distances_main.append((traset[x], tra_lab[x], min_dist))
	distances_main.sort(key=operator.itemgetter(2))
	neigh =[]



	for x in range(k):
			neigh.append(distances_main[x][1])
	neigh=np.array(neigh)
	neigh=np.reshape(neigh,(4,1))
	return neigh


def findKnearest(c_neigh):
	return sorted(c_neigh.items(), key=operator.itemgetter(1), reverse=True)

def find_majority(neigh):
	countneighbours = {}

	for m in range(len(neigh)):
		value = neigh[m, :]

		if value in tuple(countneighbours):
			countneighbours[tuple(value)] += 1
		else:
			countneighbours[tuple(value)] = 1

		total = findKnearest(countneighbours)
	return total[0][0]

k=4

corr_v = 0
for x in range(int(len(r_test))):
	neighbors =np.array(getNeigh(r_train, r_test[x], k, tra_la))

	label_pred=find_majority(neighbors)
	if label_pred==te_lab[x]:
		corr_v +=1
	print("image %d is done"%(x))

accuracy= corr_v / (int(len(r_test)))
print(accuracy*100)