import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

alpha = 0.02
n_epoch = 500

#########################################
## INSERT YOUR CODE HERE

w = train(Xtrain, Ytrain, alpha, n_epoch)   # Trainaing the model and find the W vector

yhat_train = compute_yhat(Xtrain, w)        # Making prediction on trianing set

train_loss = compute_L(yhat_train,Ytrain)   # Computing Training Loss

yhat_test = compute_yhat(Xtest, w)          # Using the trained model predicting Y labels for the given tets data

test_loss = compute_L(yhat_test,Ytest)      # Calculating Testing Loss

print("alpha is " + str(alpha) )
print("No. of Epoch is " + str(n_epoch) )
print("Training Loss is " + str(train_loss) )
print("Testing Loss is " + str(test_loss) )







#########################################

