from linear_regression import *
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

n_samples = 200
X, y = make_regression(n_samples=n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

ALPHA = [0.001, 0.005, 0.01, 0.05, 0.1, 0.50, 1, 1.5]
N_EPOCH = np.linspace(1,1000,1000)
temp = np.zeros(1000)

for a in range(0,len(ALPHA)):
    alpha = ALPHA[a]
    for b in range(1,1001):
        w = train(Xtrain, Ytrain, alpha, b)
        yhat_test = compute_yhat(Xtest, w)
        test_loss = compute_L(yhat_test, Ytest)
        temp[b-1] = test_loss
    plt.plot(N_EPOCH,temp)
    temp = np.zeros(1000)
plt.title('No. Epoch v/s log(Test_Loss)')
plt.xlabel('Epoch')
plt.ylabel('log(Test_Loss)')

plt.legend([0.001, 0.005, 0.01, 0.05, 0.1, 0.50, 1, 1.5])
plt.yscale("log")
plt.show()

