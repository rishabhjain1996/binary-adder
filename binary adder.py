import numpy as np


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)


def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x


X = np.loadtxt("data.txt").T  # read input file

# ground label Truth or output

y = np.loadtxt("ans.txt")  # read ans

# seeding random numbers

# weight matrix
w1 = 2 * np.random.random((4, np.shape(X)[0])) - 1  # (4,8) as we have 4 nodes in hidden layer 2
b1 = 2 * np.random.random((4, 1)) - 1
w2 = 2 * np.random.random((np.shape(X)[0] / 2 + 1, 4)) - 1  # (5,4) as we have 5 nodes in output layer
b2 = 2 * np.random.random((np.shape(X)[0] / 2 + 1, 1)) - 1

m = np.shape(X)[1]  # calc no of test cases automatically , here dimension of X is returned

# Learning Rate
alpha = 0.01
iter = 1000
for i in xrange(iter):
    # forward propogation
    z1 = np.dot(w1, X) + b1  # z1(4,m)
    a1 = relu(z1)
    z2 = np.dot(w2, z1) + b2  # z2(5,m)
    a2 = sigmoid(z2)

    # backward propogation

    dz2 = a2 - y.T
    dw2 = np.dot(dz2, a1.T) / m  # dw2(5,4)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * relu(z1, True)  # dz2(4,m)
    dw1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

x = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
z1 = np.dot(w1, x) + b1
a1 = relu(z1)
z2 = np.dot(w2, z1) + b2
a2 = sigmoid(z2)
print
print 'Predicted Value : ', a2
