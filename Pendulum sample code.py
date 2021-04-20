import matplotlib.pyplot as plt ### plotting things
import numpy as np ## one of python's main maths packages
import pandas as pd ## for reading in our data
from scipy.optimize import curve_fit ## for fitting a line to our data
import math
from numpy import arange
from matplotlib import pyplot

 

var = 2

def objective(x, a, b, c):
	return a * x + b * x**2 + c # b is the x**2 coefficient , a is the x coefficient, c is constant
 
dataframe = pd.read_excel('C:/Users/shean/Documents/PH20105/Y2 S2 Project Report/Reversible pendulum final data.xlsx', 
                     sheet_name = ('Small data'), # Using the first run and settings data.
                     usecols = (0,1),
                     names=('x','t'),
                     header = 0,
                     
                     nrows=22)

print('\n')
print(dataframe)
data = dataframe.values
# choose the input and output variables
x, y = data[:, 0], data[:, 1]
# curve fit
popt, pcov = curve_fit(objective, x, y)
a = popt[0]
b = popt[1]
c = popt[2]
err_a1 = np.sqrt(float(pcov[0][0]))
err_b1 = np.sqrt(float(pcov[1][1]))
err_c1 = np.sqrt(float(pcov[2][2]))
print('\n')
print(popt[0], popt[1], popt[2],err_a1,err_b1,err_c1) # These correspond to a,b,c and their respective errors(square root of their covariance)
# summarize the parameter values
a1, b1, c1 = popt

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
print('\n')
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a1, b1, c1)) # b is the x**2 coefficient , a is the x coefficient, c is constant
ax1 = pyplot.scatter(x, y, marker=6, label = '50*T_1 data')
x_line = arange(min(x), max(x), 0.01) 

y_line = objective(x_line, a1, b1, c1)

ax1= pyplot.plot(x_line, y_line, '--', color='blue', label = '50*T_1 curve fit')







dataframe = pd.read_excel('C:/Users/shean/Documents/PH20105/Y2 S2 Project Report/Reversible pendulum final data.xlsx', 
                     sheet_name = ('Small data'), # Using the first run and settings data.
                     usecols = (0,2),
                     names=('x','t'),
                     header = 0,
                     
                     nrows=22)
print('\n')
print(dataframe)
data = dataframe.values
# choose the input and output variables
x, y = data[:, 0], data[:, 1]
# curve fit
popt, pcov = curve_fit(objective, x, y)
a = popt[0]
b = popt[1]
c = popt[2]
err_a2 = np.sqrt(float(pcov[0][0]))
err_b2 = np.sqrt(float(pcov[1][1]))
err_c2 = np.sqrt(float(pcov[2][2]))
print('\n')
print(popt[0], popt[1], popt[2],err_a2,err_b2,err_c2) # These correspond to a,b,c and their respective errors(square root of their covariance)
# summarize the parameter values
a2, b2, c2 = popt
print('\n')
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a2, b2, c2)) # b is the x**2 coefficient , a is the x coefficient, c is constant
# plot input vs output

ax1 = pyplot.scatter(x, y, label = '50*T_2 data')
x_line = arange(min(x), max(x), 0.01) 
# calculate the output for the range
y_line = objective(x_line, a2, b2, c2)
ax1 = pyplot.plot(x_line, y_line, '--', color='orange', label = '50*T_2 curve fit')
plt.tick_params(direction='in',      
                length=7,) 
plt.tick_params(bottom=True, top=True, left=True, right=True)  
plt.xlabel('Position / m')
plt.ylabel('Time for 50 cycles / s')
plt.legend()

print('\n')
print(a1, b1, c1, a2, b2, c2) # Debugger

def quadsolve(a,b,c,aa,bb,cc):              # Solve for intersection points and calculates two g values.
    L = 993.9 * 10**-3
    aaa=a-aa
    bbb=b-bb
    ccc=c-cc
    coef = [bbb, aaa, ccc]
    x = np.array(np.roots(coef))
    y = np.array(a*x + b*x**2 + c)
    g = np.array(L*4*math.pi**2 / (y/50)**2) # divided by 50 cycles
    return g
    
#print(a1, b1, c1, a2, b2, c2) # Debugger
print('\n')
print(quadsolve(a1, b1, c1, a2, b2, c2))


def root(a,b,c,aa,bb,cc):              # Solve for intersection points and calculates two g values.
    aaa=a-aa
    bbb=b-bb
    ccc=c-cc
    coef = [bbb, aaa, ccc]
    x = np.array(np.roots(coef))
    return x

def T50(a,b,c,aa,bb,cc):              # Solve for intersection points and calculates two g values.
    aaa=a-aa
    bbb=b-bb
    ccc=c-cc
    coef = [bbb, aaa, ccc]
    x = np.array(np.roots(coef))
    y = np.array(a*x + b*x**2 + c)
    return y

err_x = 2*10**-3
L_eff = 993.9 * 10**-3
err_Leff = 0.1*10**-3
x_root = np.array(root(a1, b1, c1, a2, b2, c2))
# first = (a*x_root**2)**2 * ((err_a/a)**2 + (2*err_x/x_root)**2)
# second = (b*x_root)**2 * ((err_b/b)**2 + (err_x/x_root)**2)
# third = err_c**2
err_y1 = np.sqrt((b1*x_root[0]**2)**2 * ((err_b1/b1)**2 + (2*err_x/x_root[0])**2) + (a1*x_root[0])**2 * ((err_a1/a1)**2 + (err_x/x_root[0])**2) + err_c1**2)
err_y2 = np.sqrt((b2*x_root[1]**2)**2 * ((err_b2/b2)**2 + (2*err_x/x_root[1])**2) + (a2*x_root[1])**2 * ((err_a2/a2)**2 + (err_x/x_root[1])**2) + err_c2**2)
print(x_root)
print(err_y1, err_y2) # uncertainty in T plus and T minus respectively

g = np.array(quadsolve(a1, b1, c1, a2, b2, c2))
y12 = np.array(T50(a1, b1, c1, a2, b2, c2))
err_g1 = g[0]*np.sqrt((err_Leff/L_eff)**2 + 2*((err_y1/50)/y12[0])**2)
err_g2 = g[1]*np.sqrt((err_Leff/L_eff)**2 + 2*((err_y1/50)/y12[1])**2)
print(T50(a1, b1, c1, a2, b2, c2)/50)
print(err_g1, err_g2)
print('\n')
print('g+ = %f +- %f AND g- = %f +- %f' %(g[0], err_g1, g[1], err_g2))



dataframe = pd.read_excel('C:/Users/shean/Documents/PH20105/Y2 S2 Project Report/Reversible pendulum final data.xlsx', 
                     sheet_name = ('Large data'), # Using the first run and settings data.
                     usecols = (0,1),
                     names=('x','t'),
                     header = 0,
                     
                     nrows=10)

print('\n')
print(dataframe)
data = dataframe.values
# choose the input and output variables
x, y = data[:, 0], data[:, 1]
# curve fit
popt, pcov = curve_fit(objective, x, y)
a = popt[0]
b = popt[1]
c = popt[2]
err_a1 = np.sqrt(float(pcov[0][0]))
err_b1 = np.sqrt(float(pcov[1][1]))
err_c1 = np.sqrt(float(pcov[2][2]))
print('\n')
print(popt[0], popt[1], popt[2],err_a1,err_b1,err_c1) # These correspond to a,b,c and their respective errors(square root of their covariance)
# summarize the parameter values
a1, b1, c1 = popt
ax2 = fig.add_subplot(122)
print('\n')
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a1, b1, c1)) # b is the x**2 coefficient , a is the x coefficient, c is constant
# plot input vs output
ax2 = pyplot.scatter(x, y, marker=6, label = '50*T_1 data')
x_line = arange(min(x), max(x), 0.001)
# calculate the output for the range
y_line = objective(x_line, a1, b1, c1)
# create a line plot for the mapping function
ax2 = pyplot.plot(x_line, y_line, '--', color='blue', label = '50*T_1 curve fit')




dataframe = pd.read_excel('C:/Users/shean/Documents/PH20105/Y2 S2 Project Report/Reversible pendulum final data.xlsx', 
                     sheet_name = ('Large data'), # Using the first run and settings data.
                     usecols = (0,2),
                     names=('x','t'),
                     header = 0,
                     
                     nrows=10)
print('\n')
print(dataframe)
data = dataframe.values
# choose the input and output variables
x, y = data[:, 0], data[:, 1]
# curve fit
popt, pcov = curve_fit(objective, x, y)
a = popt[0]
b = popt[1]
c = popt[2]
err_a2 = np.sqrt(float(pcov[0][0]))
err_b2 = np.sqrt(float(pcov[1][1]))
err_c2 = np.sqrt(float(pcov[2][2]))
print('\n')
print(popt[0], popt[1], popt[2],err_a2,err_b2,err_c2) # These correspond to a,b,c and their respective errors(square root of their covariance)
# summarize the parameter values
a2, b2, c2 = popt
print('\n')
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a2, b2, c2)) # b is the x**2 coefficient , a is the x coefficient, c is constant
# plot input vs output
ax2 = pyplot.scatter(x, y, label = '50*T_2 data')
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 0.001) # Changed from 1 to 0.01 to have the curve fit
# calculate the output for the range
y_line = objective(x_line, a2, b2, c2)
# create a line plot for the mapping function
ax2 = pyplot.plot(x_line, y_line, '--', color='orange', label = '50*T_2 curve fit')
plt.tick_params(direction='in',      
                length=7,) 
plt.tick_params(bottom=True, top=True, left=True, right=True)  
#plt.title('Large data XXXXXXXXXXX')
plt.xlabel('Position / m')
#plt.ylabel('Time for 50 cycles / s')
plt.legend()
pyplot.show()


print('\n')
print(a1, b1, c1, a2, b2, c2) # Debugger


    
#print(a1, b1, c1, a2, b2, c2) # Debugger
print('\n')
print(quadsolve(a1, b1, c1, a2, b2, c2))            # g = 9.67896572 is the one we obtain. g = 10.17546141 is the interpolated one.




err_x = 2*10**-3
L_eff = 993.9 * 10**-3
err_Leff = 0.1*10**-3
x_root = np.array(root(a1, b1, c1, a2, b2, c2))

err_y1 = np.sqrt((b1*x_root[0]**2)**2 * ((err_b1/b1)**2 + (2*err_x/x_root[0])**2) + (a1*x_root[0])**2 * ((err_a1/a1)**2 + (err_x/x_root[0])**2) + err_c1**2)
err_y2 = np.sqrt((b2*x_root[1]**2)**2 * ((err_b2/b2)**2 + (2*err_x/x_root[1])**2) + (a2*x_root[1])**2 * ((err_a2/a2)**2 + (err_x/x_root[1])**2) + err_c2**2)
print(x_root)
print(err_y1, err_y2) # uncertainty in T plus and T minus respectively

g = np.array(quadsolve(a1, b1, c1, a2, b2, c2))
y12 = np.array(T50(a1, b1, c1, a2, b2, c2))
err_g1 = g[0]*np.sqrt((err_Leff/L_eff)**2 + 2*((err_y1/50)/y12[0])**2)
err_g2 = g[1]*np.sqrt((err_Leff/L_eff)**2 + 2*((err_y1/50)/y12[1])**2)
print(T50(a1, b1, c1, a2, b2, c2)/50)
print(err_g1, err_g2)
print('\n')
print('g+ = %f +- %f AND g- = %f +- %f' %(g[0], err_g1, g[1], err_g2))

fig.savefig('Reversible final combined.png', dpi=1000)