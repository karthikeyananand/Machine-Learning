import math
import pandas as pd
import struct
import numpy as np
from math import sqrt
from collections import Counter

def Data_Read_Knn(file):
    with open(file, 'rb') as f:
        zero, type, Dimension_Value= struct.unpack('>HBB', f.read(4))
        KNN_Shape = list(struct.unpack('>I', f.read(4))[0] for d in range(Dimension_Value))
        print(KNN_Shape)
        if file == "t10k-images.idx3-ubyte":
            X_Read= np.fromstring(f.read(), dtype=np.uint8).reshape(KNN_Shape[0] ,KNN_Shape[1] * KNN_Shape[2])

        if file == "t10k-labels.idx1-ubyte":
            X_Read=np.fromstring(f.read(), dtype=np.uint8).reshape(KNN_Shape)

        if file == "train-images.idx3-ubyte":
            X_Read=np.fromstring(f.read(), dtype=np.uint8).reshape(KNN_Shape[0] ,KNN_Shape[1] * KNN_Shape[2])

        if file == "train-labels.idx1-ubyte":
            X_Read=np.fromstring(f.read(), dtype=np.uint8).reshape(KNN_Shape)

        return X_Read



def knn_neighbors(MNIST_Training_img, MNIST_Training_Label, MNIST_Test_img, MNIST_Test_Label, Value_K):
    knn_Res = []
    knn_cnt = 0
    knn_Dist = []




    for i in MNIST_Test_img:

            for rows in MNIST_Training_img:
                Dist_Euclidean = np.sqrt(np.sum([(int(a) - int(b)) ** 2 for a, b in zip(i, rows)]))

                knn_Dist.append([Dist_Euclidean, MNIST_Training_Label[knn_cnt]])
                knn_cnt = knn_cnt + 1
            knn_vote_count = [i[1] for i in sorted(knn_Dist)[:Value_K]]
            vote_res = Counter(knn_vote_count).most_common(1)[0][0]
            knn_Res.append(vote_res)
            knn_cnt = 0
            knn_Dist.clear()


    Y_Act=pd.Series(MNIST_Test_Label)
    Y_Predict=pd.Series(knn_Res)
    Confusion = pd.crosstab(Y_Act, Y_Predict)
    Accuracy_Prediction = (Y_Act == Y_Predict).sum() / float(len(Y_Act))
    print(Confusion)
    print(Accuracy_Prediction)

MNIST_Test_img= Data_Read_Knn("t10k-images.idx3-ubyte")
MNIST_Test_Label= Data_Read_Knn("t10k-labels.idx1-ubyte")
MNIST_Training_img=Data_Read_Knn("train-images.idx3-ubyte")
MNIST_Training_Label=Data_Read_Knn("train-labels.idx1-ubyte")
knn_neighbors(MNIST_Training_img, MNIST_Training_Label, MNIST_Test_img, MNIST_Test_Label, Value_K=3)