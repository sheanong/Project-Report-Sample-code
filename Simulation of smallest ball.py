import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

g = 9.81 # T decreases, frequency increases at g takes a higher value.
m = 2.09*10**-3 
d = 7.98*10**-3
rho = 1.2754
mu = 1.789*10**-5
C = 0.47
b = (math.pi/8)*rho*C*d**2 # damping parameter




# function that returns dv/dt
def model(v,t):
    
    dvdt = -(b/m) * v**2 + g
    return dvdt

# initial condition. velocity of the body about to be dropped starts with initial velocity = 0
y0 = 0 

# time values
t = np.linspace(0,15)


# solve ODE
v = odeint(model,y0,t)


# plot 
plt.plot(t,v)
plt.xlabel('time /s ')
plt.ylabel('velocity / ms^-1')
plt.tick_params(direction='in',      
                length=7,           
               )
plt.tick_params(bottom=True, top=True, left=True, right=True)
plt.margins(x=0, y=0)
plt.ylim([0, 40])

plt.show()

u = 71.8*10**-2/0.393
R = rho*u*d/mu
print(R)