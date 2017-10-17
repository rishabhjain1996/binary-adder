import numpy as np

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)


X = np.loadtxt("data.txt").T   # read input file

# ground label Truth or output

y = np.loadtxt("ans.txt")    #read ans

# seeding random numbers

# weight matrix
w1 = 2 * np.random.random((4, np.shape(X)[0])) - 1    #(4,8) as we have 4 nodes in hidden layer 2
b1 = 2 * np.random.random((4, 1)) - 1
w2 = 2 * np.random.random((np.shape(X)[0] /2 +1,4)) - 1   #(5,4) as we have 5 nodes in output layer
b2 = 2 * np.random.random((np.shape(X)[0] /2 +1, 1)) - 1

m=np.shape(X)[1]      #calc no of test cases automatically , here dimension of X is returned

# Learning Rate
alpha = 0.1
iter=1
for i in xrange(iter):
    # forward propogation
    z1 = np.dot(w1, X) + b1  # z1(4,m)
    a1 = tanh(z1)
    z2=np.dot(w2,z1)+b2
    a2=sigmoid(z2)

    # backward propogation
