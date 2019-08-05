from classes import svm_class
import pandas as pd
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

glasses_obj= svm_class()
glasses_dataset=glasses_obj.Readdataset("uci_glasses_dataset.csv")

features,labels=glasses_obj.dataprepare(glasses_dataset)

X_train, X_test, y_train, y_test=glasses_obj.data_split(glasses_dataset, features, labels)

OC_sigmod,Og_sigmoid=glasses_obj.kfold_crossvalid("sigmoid", X_train, y_train)
OC_rbf,Og_rbf=glasses_obj.kfold_crossvalid("rbf", X_train, y_train)
OC_linear,Og_linear=glasses_obj.kfold_crossvalid("linear", X_train, y_train)
OC_poly,Og_poly=glasses_obj.kfold_crossvalid("poly", X_train, y_train)
glasses_obj.svm_method(X_train, X_test, y_train, y_test,OC_sigmod,Og_sigmoid,"sigmoid")
glasses_obj.svm_method(X_train, X_test, y_train, y_test,OC_rbf,Og_rbf,"rbf")
glasses_obj.svm_method(X_train, X_test, y_train, y_test,OC_linear,Og_linear,"linear")
glasses_obj.svm_method(X_train, X_test, y_train, y_test,OC_poly,Og_poly,"poly")


glasses_obj.one_vs_rest(X_train, X_test, y_train, y_test,OC_sigmod,Og_sigmoid,"sigmoid")
glasses_obj.one_vs_rest(X_train, X_test, y_train, y_test,OC_rbf,Og_rbf,"rbf")
glasses_obj.one_vs_rest(X_train, X_test, y_train, y_test,OC_linear,Og_linear,"linear")
glasses_obj.one_vs_rest(X_train, X_test, y_train, y_test,OC_poly,Og_poly,"poly")


glasses_obj.svm_method_balanced(X_train, X_test, y_train, y_test,OC_sigmod,Og_sigmoid,"sigmoid")
glasses_obj.svm_method_balanced(X_train, X_test, y_train, y_test,OC_rbf,Og_rbf,"rbf")
glasses_obj.svm_method_balanced(X_train, X_test, y_train, y_test,OC_linear,Og_linear,"linear")
glasses_obj.svm_method_balanced(X_train, X_test, y_train, y_test,OC_poly,Og_poly,"poly")