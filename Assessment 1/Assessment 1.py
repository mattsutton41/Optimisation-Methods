import torch
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
matplotlib.use("TKAgg")

##### QUESTION 1 #####


# Function for Gradient Descent taking parameters
# step, alpha, beta, position x, acceptable error and function
def GradDescent(s, a, b, x, e, f):
	step = s
	iterations = 0
	value, gradient = f(x)
	while(torch.norm(gradient) > e):
		iterations += 1
		#print("iteration:", iterations)
		step = s
		while(f(x)[0]-f(x-(step*gradient))[0] < a * step * (gradient.norm().item()**2)):
			step = b * step
		#print("step:", step)
		x.data -= step*gradient
		value, gradient = f(x)
		# print("gradient norm:", torch.norm(gradient))
		# print("x:", x)
		# print()
	return iterations


#Function for computing the function in 1.1. Also returns Gradient.
def f1(x):
	A = torch.zeros((5,5))
	for i in range(5):
		for j in range(5):
			A[i,j] = 1/(i+j+1)
	xnew = x.clone().detach().requires_grad_(True)
	output = torch.matmul(torch.matmul(xnew.T, A), xnew)
	output.backward()
	return output.item(), xnew.grad


#Function for computing the first function given in 1.2. Also returns Gradient.
def f2(x):
	A = torch.tensor([[1,0],[0,2]], requires_grad = True, dtype = torch.float)
	xnew = x.clone().detach().requires_grad_(True)
	output = torch.matmul(torch.matmul(xnew.T, A), xnew)
	output.backward()
	return output.item(), xnew.grad

#Function for computing the second function give in 1.2. Also returns Gradient.
def f3(x):
	A = torch.tensor([[1,0],[0,0.01]], requires_grad = True, dtype = torch.float)
	xnew = x.clone().detach().requires_grad_(True)
	output = torch.matmul(torch.matmul(xnew.T, A), xnew)
	output.backward()
	return output.item(), xnew.grad


#Number of iterations for Gradient Descent to complete with given parameters
x = torch.tensor([1,2,3,4,5], requires_grad = True, dtype = torch.float)
print("Gradient Descent took {} iterations to fall within the given error".format(GradDescent(1, 0.5, 0.5, x, 0.1, f1)))
x = torch.tensor([1,2,3,4,5], requires_grad = True, dtype = torch.float)
print("Gradient Descent took {} iterations to fall within the given error".format(GradDescent(1, 0.1, 0.1, x, 0.1, f1)))
x = torch.tensor([2,1], requires_grad = True, dtype = torch.float)
print("Gradient Descent took {} iterations to fall within the given error".format(GradDescent(1, 0.25, 0.5, x, 10**-5, f2)))
x = torch.tensor([1/100,1], requires_grad = True, dtype = torch.float)
print("Gradient Descent took {} iterations to fall within the given error".format(GradDescent(2, 0.25, 0.5, x, 10**-5, f3)))
print()

##### QUESTION 2 #####

#Computes cost function using torch functions. Also returns gradient.
def cost_function(X, label, w):
	sig = torch.nn.Sigmoid()
	Xw = torch.mul(X,w)
	ones = torch.ones(label.shape)
	sum_Xw = torch.sum(Xw, dim = 1)
	left_matrix = torch.matmul((-1*label).T, torch.log(sig(sum_Xw)))
	right_matrix = torch.matmul((ones.sub(label)),torch.log(ones.sub(sig(sum_Xw))))
	sum_matrix = torch.sub(left_matrix, right_matrix)
	f = sum_matrix/label.numel()
	return f

def gradient(cost, w): # Your code goes here
	cost.backward()
	grad = w.grad.clone()
	w.grad.detach_()
	w.grad.zero_()
	return grad

#Computes a simpler Gradient Descent for a given number of steps and fixed step size
def gradient_descent(X, label, w, num_epoch):
	trace = torch.zeros([num_epoch, 3])
	step = 0.1
	iterations = 0
	while(iterations < num_epoch):
		trace[iterations] = w
		iterations += 1
		w = (w - step*gradient(cost_function(X,label,w),w)).clone().detach().requires_grad_(True)
		
	return w, trace

#Plots the line resulting from the trace
def plot(w):
	w1 = w[0].item()
	w2 = w[1].item()
	w3 = w[2].item()
	if (w1 == 0 and w2 == 0):
		x_values = []
		y_values = []
	else:
		points = np.linspace(3,7,1000)
		x_values = points
		y_values = (1/w2)*(-1*x_values*w1 - w3)
	line.set_data(x_values, y_values)
	return line,

data = np.loadtxt("newiris.csv", delimiter=',')
X = data[:, 2:-1]
X = np.c_[X, np.ones(X.shape[0])]
X = torch.Tensor(X)
label = torch.Tensor(data[:, -1])
w = torch.zeros(3, requires_grad=True)
w, trace = gradient_descent(X, label, w, 20000)
print(w)

fig = plt.figure()
ax = plt.axes(ylim=(-1,3))
ax.scatter(X[:49,0], X[:49,1], c='g')
ax.scatter(X[50:,0], X[50:,1], c='b')
line, = ax.plot([], [])
anim = animation.FuncAnimation(fig, lambda i: plot(trace[i]), frames=len(trace), interval=10, blit=True)
plt.show()