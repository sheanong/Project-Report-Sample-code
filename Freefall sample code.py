import matplotlib.pyplot as plt ### plotting things
import numpy as np ## one of python's main maths packages
import pandas as pd ## for reading in our data
from scipy.optimize import curve_fit ## for fitting a line to our data
import math


h_unc = 0.4*10**-2

d1 = 19.97 * 10 ** -3
d2 = 21.97 * 10 ** -3
d3 = 24.96 * 10 ** -3
d_unc = 0.01*10**-3

m1 = 32.6 * 10 ** -3
m2 = 43.4 * 10 ** -3
m3 = 63.6 * 10 ** -3
m_unc = 0.05*10**-3

t_unc = 1*10**-4


density = 1.2754
C = 0.47 # Drag coefficient of a sphere
coef = math.pi*density*C/8 #approximately 0.2261

data = pd.read_excel('C:/Users/shean/Downloads/Gravity balls.xlsx', 
                      sheet_name = ('Data 3'), # Using the first run and settings data.
                      usecols = (0,1,2,3),
                      names=('h','t1','t2', 't3'),
                      header = 0,
                      
                      nrows=10)


print(data.t3)

def withdrag(grad):
    g = grad**2
    return g

def ploty(h, d, m):
    # With drag y
    b = coef*d**2
    y = np.arccosh(np.exp(h*b/m))
    return y
    


def plotx(t, d, m):
    #t_avg = np.mean(t)    
    b = coef*d**2
    # With drag x
    x = np.sqrt(b/m)*t
    return x
    
def ploty_unc(h, d, m, h_unc, d_unc, m_unc): # Uncertainty of y
    q = np.exp(h*coef*d**2/m)
    p = coef*q*(np.sqrt((h_unc/h)**2+(d_unc/d)**2+(m_unc/m)**2))
    y_unc = ((1+q*((q**2-1)**-0.5)/(q + np.sqrt((q**2)-1))))*p
    return y_unc

def plotx_unc(t, d, m, t_unc, d_unc, m_unc, N): # Uncertainty of x
    t_avg = np.mean(t)
    t_unc_avg = t_unc/np.sqrt(N-1) # This is the Standard error of mean from Stats and LaTeX
    x_unc = np.sqrt(coef)*np.sqrt(0.5*(m_unc/m)**2 + (d_unc/d)**2 + (t_unc_avg/t_avg)**2)
    return x_unc


fig = plt.figure(figsize=(7,7))



ax1 = fig.add_subplot(111)

ax1.errorbar(plotx(data.t1, d1, m1)*1000,           
              ploty(data.h*10**-2,d1, m1)*1000,              
              xerr = plotx_unc(data.t1, d1, m1, t_unc, d_unc,m_unc, 9)*1000,
              yerr = ploty_unc(data.h*10**-2, d1, m1, t_unc, d_unc, m_unc)*1000,                 
              marker='^',             
              markersize = 5,        
              color='blue',         
              ecolor='blue',        
              markerfacecolor='blue',
              linestyle='none',       
              capsize=3,  
              label = "Small ball data"
              )
ax1.set_xlabel('1000*sqrt(b/m)*t / (ms^-2)^0.5')
ax1.set_ylabel('1000*arccosh(e^(h*b/m))')
ax1.set_ylim([40, 110])
ax1.set_xlim([12, 35])
ax1.tick_params(direction='in',      
                length=7,)           
                
ax1.tick_params(bottom=True, top=True, left=True, right=True)     
def line(x, slope, intercept):          
    return slope*x + intercept          


popt, pcov = curve_fit(line, plotx(data.t1, d1, m1), ploty(data.h*10**-2,d1, m1)) #,sigma = yset1_unc)   Doing a weighted linear fit makes our gradient lower for some reason     
slope = popt[0]
intercept = popt[1]
err_slope = np.sqrt(float(pcov[0][0]))
err_intercept = np.sqrt(float(pcov[1][1]))

print('\n')
print('Slope: {0:.3f} +- {1:.3f}'.format(slope, err_slope))
print('Intercept: {0:.3f} +- {1:.3f}'.format(intercept, err_intercept))
print('g1 = {0:f} +- {1:f}' .format(withdrag(slope) , 2*slope*err_slope))
print('\n')
bestfit_x = np.linspace(0, 0.05 , 1000)

ax1.plot(bestfit_x*1000, (bestfit_x*slope+intercept)*1000, 
          linestyle='-',
          color='blue',
          linewidth='1',
          label='Best fit for small ball')

ax1.margins(x=0)

ax2 = fig.add_subplot(111)


ax2.errorbar(plotx(data.t2, d2, m2)*1000,           
             ploty(data.h*10**-2,d2, m2)*1000,              
              xerr = plotx_unc(data.t2, d2, m2, t_unc, d_unc, m_unc, 9)*1000,
              yerr = ploty_unc(data.h*10**-2, d2, m2, t_unc, d_unc, m_unc)*1000,                 
              marker='o',             
              markersize = 5,        
              color='orange',         
              ecolor='orange',        
              markerfacecolor='orange',
              linestyle='none',       
              capsize=3,  
              label = "Medium ball data"
              )

ax2.set_ylim([40, 110])
ax2.set_xlim([12, 35])
ax2.tick_params(direction='in',      
                length=7,)           
ax2.tick_params(bottom=True, top=True, left=True, right=True)    

def line(x, slope, intercept):          
    return slope*x + intercept          


popt, pcov = curve_fit(line, plotx(data.t2, d2, m2), ploty(data.h*10**-2,d2, m2)) #,sigma = yset1_unc)   Doing a weighted linear fit makes our gradient lower for some reason     
slope = popt[0]
intercept = popt[1]
err_slope = np.sqrt(float(pcov[0][0]))
err_intercept = np.sqrt(float(pcov[1][1]))

print('\n')
print('Slope: {0:.3f} +- {1:.3f}'.format(slope, err_slope))
print('Intercept: {0:.3f} +- {1:.3f}'.format(intercept, err_intercept))
print('g2 = {0:f} +- {1:f}' .format(withdrag(slope) , 2*slope*err_slope))
print('\n')

bestfit_x = np.linspace(0, 0.05 , 1000)

ax2.plot(bestfit_x*1000, (bestfit_x*slope+intercept)*1000, 
          linestyle='-',
          color='orange',
          linewidth='1',
          label='Best fit for medium ball')

ax2.margins(x=0)

ax3 = fig.add_subplot(111)

ax3.errorbar(plotx(data.t3, d3, m3)*1000,           
              ploty(data.h*10**-2,d3, m3)*1000,              
              xerr = plotx_unc(data.t3, d3, m3, t_unc, d_unc,m_unc, 9)*1000,
              yerr = ploty_unc(data.h*10**-2, d3, m3, t_unc, d_unc, m_unc)*1000,                 
              marker='p',             
              markersize = 5,        
              color='black',         
              ecolor='black',        
              markerfacecolor='black',
              linestyle='none',       
              capsize=3,  
              label = "Large ball data"
              )

ax3.set_ylim([40, 110])
ax3.set_xlim([15, 35])
ax3.tick_params(direction='in',      
                length=7,)           
ax3.tick_params(bottom=True, top=True, left=True, right=True)    

def line(x, slope, intercept):          
    return slope*x + intercept          


popt, pcov = curve_fit(line, plotx(data.t3, d3, m3), ploty(data.h*10**-2,d3, m3)) 
slope = popt[0]
intercept = popt[1]
err_slope = np.sqrt(float(pcov[0][0]))
err_intercept = np.sqrt(float(pcov[1][1]))



print('\n')
print('Slope: {0:.3f} +- {1:.3f}'.format(slope, err_slope))
print('Intercept: {0:.3f} +- {1:.3f}'.format(intercept, err_intercept))
print('g3 = {0:f} +- {1:f}' .format(withdrag(slope) , 2*slope*err_slope))
print('\n')

bestfit_x = np.linspace(0, 0.05 , 1000)

ax3.plot(bestfit_x*1000, (bestfit_x*slope+intercept)*1000, 
          linestyle='-',
          color='black',
          linewidth='1',
          label='Best fit for large ball')

ax3.margins(x=0)

plt.legend()
plt.show()




              
               
               