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

def hybrid_Newton(f, g, H, x0, eps, M, a, b):
	T = 0
	xT = x0
	while(T<M)&(np.linalg.norm(g(xT))>eps):
		if (pos_def(H(xT))):
			d = np.linalg.solve(H(xT), -g(xT))
		else:
			d = -g(xT)

		tk = 1.0
		while((f(xT) - f(xT+(tk*d)))<((-a*tk*d)@g(xT))):
			tk = b*tk

		xT = xT + tk*d
		T += 1
	return T, xT

#Checks if a given matrix is Positive Definite by converting it to Cholesky Form.
def pos_def(H):
	if(np.array_equal(H,H.T)):
		try:
			np.linalg.cholesky(H)
			return True
		except np.linalg.LinAlgError:
			return False
	else:
		return False

def h1(x):
	x1 = x[0]
	x2 = x[1]
	return -13.0 + x1 + (((5.0-x2)*x2)-2.0)*x2

def h2(x):
	x1, x2 = x[0], x[1]
	return -29.0 + x1 + (((x2+1.0)*x2)-14.0)*x2

def fr(x):
	return ((h1(x))**2.0) + ((h2(x))**2.0)

#Computes the gradient of the FR Test function at point x
def gfr(x):
	x1, x2 = x[0], x[1]
	output = np.zeros(2)
	output[0] = 4.0*(x1+3.0*x2**2.0-8.0*x2-21.0)
	output[1] = 4.0*(x1*(6.0*x2-8.0)+3.0*x2**5.0-10.0*x2**4.0+2.0*x2**3.0-60.0*x2**2.0+6.0*x2+216.0)
	return output

#Computes the Hessian of the FR Test function at point x
def hfr(x):
	x1, x2 = x[0], x[1]
	output = np.zeros((2,2))
	output[0,0] = 4.0
	output[0,1] = 24.0*x2-32.0
	output[1,0] = 24.0*x2-32.0
	output[1,1] = 24.0*x1+60.0*x2**4.0-160.0*x2**3.0+24.0*x2**2.0-480.0*x2+24.0
	return output

#Outputs the solutions to Task 3.2
print("Task 3.2a Damped; Start (-50, 7): ", damped_Newton(fr,gfr,hfr,[-50,7],10**-5,10**3,1/2,1/2))
print("Task 3.2b Hybrid; Start (-50, 7): ", hybrid_Newton(fr,gfr,hfr,[-50,7],10**-5,10**3,1/2,1/2))

print("Task 3.2a Damped; Start (20, 7): ", damped_Newton(fr,gfr,hfr,[20,7],10**-5,10**3,1/2,1/2))
print("Task 3.2b Hybrid; Start (20, 7): ", hybrid_Newton(fr,gfr,hfr,[20,7],10**-5,10**3,1/2,1/2))

print("Task 3.2a Damped; Start (20, -18): ", damped_Newton(fr,gfr,hfr,[20,-18],10**-5,10**3,1/2,1/2))
print("Task 3.2b Hybrid; Start (20, -18): ", hybrid_Newton(fr,gfr,hfr,[20,-18],10**-5,10**3,1/2,1/2))

print("Task 3.2a Damped; Start (5, -10): ", damped_Newton(fr,gfr,hfr,[5,-10],10**-5,10**3,1/2,1/2))
print("Task 3.2a Hybrid; Start (5, -10): ", hybrid_Newton(fr,gfr,hfr,[5,-10],10**-5,10**3,1/2,1/2))