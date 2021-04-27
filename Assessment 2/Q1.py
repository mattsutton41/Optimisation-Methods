import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib

##CS416 Assessment 2 Question 1

def pure_Newton(f, g, H, x0, eps, M):
	T = 0
	xT = x0
	while(T < M)&(np.linalg.norm(g(xT))>eps):
		d = np.linalg.solve(H(xT), -g(xT))
		xT = xT + d
		T += 1
	return T, xT

def f(x):
	x1 = x[0]
	x2 = x[1]
	output = (x1**2+1)**0.5 + (x2**2+1)**0.5
	return output

def g(x):
	x1 = x[0]
	x2 = x[1]
	output = np.zeros(2)
	output[0] = x1/(((x1**2.0)+1.0)**(0.5))
	output[1] = x2/(((x2**2.0)+1.0)**(0.5))
	return output

def h(x):
	x1 = x[0]
	x2 = x[1]
	output = np.zeros((2,2))
	output[0,0] = ((x1**2.0)+1)**-(1.5)
	output[1,1] = ((x2**2.0)+1)**-(1.5)
	return output


##Outputs the solutions to Task 1.3
print("Task 1.3a: ", pure_Newton(f,g,h,[1,1],10**-8,10**3))
print("Task 1.3b: ", pure_Newton(f,g,h,[10,10],10**-8,10**3))