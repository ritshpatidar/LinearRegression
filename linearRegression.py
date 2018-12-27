import numpy as np
from matplotlib import pyplot as plt

no_of_examples = 4
no_of_features = 3

X = np.array([
	[1,1], #example 1
	[1,2], #example 2
	[2,2], #example 3
	[2,3]])#example 4
x_2 = np.ones((no_of_examples,1)) #x_2=1 is for w2(intercept term)

x_train = np.concatenate((X,x_2),axis=1)
y_train = np.array([[6],[8],[9],[11]]) # y = 1 * x_0 + 2 * x_1 + 3

np.random.seed(4)
w = np.random.random((no_of_features,1))
alpha = 0.01
iters = 10000
cost = []

for i in range(iters):
	pred = np.matmul(x_train,w) #y = x_0*w_0 + x_1*w_1 + x_2*w_2
	delta_w = np.matmul(x_train.T,(pred - y_train))
	cost.append(np.mean(np.square(pred - y_train)))
	w = w - alpha*delta_w/no_of_examples

#prediction for [3,5], x_2 = 1 always
pred = np.matmul(np.array([[3, 5,1]]),w)
print("Prediction for [3,5] should be 1*3 + 2*5 + 3 = 16")
print("Prediction:",pred)

#plot
plt.plot(cost)
plt.show()