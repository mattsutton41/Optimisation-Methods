import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib

def damped_Newton(f, g, H, x0, eps, M, a, b):
	T = 0
	xT = x0
	while(T < M)&(np.linalg.norm(g(xT))>eps):
		d = np.linalg.solve(H(xT), -g(xT))
		tk = 1.0
		while((f(xT) - f(xT+(tk*d)))<((-a*tk*d)@g(xT))):
			tk = b*tk
		
		xT = xT + tk*d
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

print("Task 2.2a: ", damped_Newton(f,g,h,[1,1],10**-8,10**3,1/2,1/2))
print("Task 2.2b: ", damped_Newton(f,g,h,[10,10],10**-8,10**3,1/2,1/2))