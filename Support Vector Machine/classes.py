import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
class svm_class:


   def Readdataset(self, f_name):
      glasses=pd.read_csv(f_name)
      return glasses

   def dataprepare(self, dataset_glasses):
       feat = dataset_glasses.drop('x', axis=1)
       labels = dataset_glasses['x']
       return feat,labels

   def data_split(self, dataset_glasses, feat, labels):
       xtrain, xtest, ytrain, ytest = train_test_split(feat, labels, test_size=0.20, random_state=42)
       return xtrain, xtest, ytrain, ytest

   def kfold_crossvalid(self, kernel, xtrain, ytrain):
        accuracy_rbf=[]
        C_rbf=[]
        gamma_rbf=[]
        for i in range(-5, 5):
            for j in range(-5, 5):
                svc = SVC(kernel=kernel, C=(2**i), gamma=(2**j),class_weight='balanced')
                scores = cross_val_score(svc, xtrain, ytrain, cv=5)
                accuracy_rbf.append(scores.mean())
                C_rbf.append(i)

                gamma_rbf.append(j)



        index = accuracy_rbf.index(max(accuracy_rbf))
        optimal_c = C_rbf[index]
        optimalgamma = gamma_rbf[index]

        return optimal_c,optimalgamma


   def one_vs_rest(self, xtrain, xtest, ytrain, ytest, Opt_C, Opt_g, kernel):
       start = time.time()
       svclassifier = OneVsRestClassifier(SVC(kernel=kernel, C=2 ** Opt_C, gamma=2 ** Opt_g))
       svclassifier.fit(xtrain, ytrain)
       y_pred = svclassifier.predict(xtest)

       print("\n{0} Accuracy One vs Rest".format(kernel),":",accuracy_score(ytest, y_pred))
       print("Training Time:", time.time() - start,"seconds")
       print("---------------------------------------------------------")

   def svm_method(self, X_train, X_test, y_train, y_test, Opt_C, Opt_g, kernel):
       start=time.time()
       svclassifier = SVC(kernel=kernel, C=2 ** Opt_C, gamma=2 ** Opt_g, decision_function_shape='ovo')
       svclassifier.fit(X_train, y_train)
       y_pred = svclassifier.predict(X_test)
       print("\n{0} Accuracy SVC".format(kernel),":",accuracy_score(y_test, y_pred))
       print("Training Time:",time.time()-start,"seconds")
       print("---------------------------------------------------------")

   def svm_method_balanced(self, X_train, X_test, y_train, y_test, Opt_C, Opt_G, kernel):
       start = time.time()
       svclassifier = SVC(kernel=kernel, C=2 ** Opt_C, gamma=2 ** Opt_G, decision_function_shape='ovo', class_weight='balanced')
       svclassifier.fit(X_train, y_train)
       y_pred = svclassifier.predict(X_test)
       print("\n{0} Accuracy SVC Balanced".format(kernel),":",accuracy_score(y_test, y_pred))
       print("Training Time:", time.time() - start,"seconds")
       print("---------------------------------------------------------")
