#!/usr/bin/python




import numpy as np
import scipy as sc
import pylab as plt


data=sc.genfromtxt('Host.dat')

r_host=((data[0,8]/0.7)/1000.)
x_host=data[0,0]/0.7
y_host=data[0,1]/0.7

r_sub=((data[1,8]/0.7)/1000.)
x_sub=data[1,0]/0.7
y_sub=data[1,1]/0.7

an = np.linspace(0,2*np.pi,100)
plt.plot(r_sub*np.cos(an)+(x_sub),r_sub*np.sin(an)+(y_sub),color='black',label=r'$\nu_{circ,Sub}=$'+str(data[1,10])+' kms$^{-1}$')
plt.plot(r_host*np.cos(an)+(x_host),r_host*np.sin(an)+(y_host),ls='--',color='gray',label=r'$\nu_{circ,Halo}=$'+str(data[0,10])+' kms$^{-1}$')
plt.plot((x_host,x_sub),(y_host,y_sub),lw=3,color='black')
#plt.plot((0,x_sub),(0,y_sub),color='black')
#plt.plot((0,x_host),(0,y_host),color='black')
plt.title('Fernandez-Trincado et al. (2014)')
plt.text(333.701,148.78,'Bullet Group',fontsize='xx-large')
plt.text(333.701,146.27,r'X$_{off,new}=d_{real,(X,Y)}/R_{virial,Halo}$',fontsize='xx-large')
plt.text(334.89,148.,'Halo',fontsize='xx-large')
plt.text(335.729,146.626,'Sub',fontsize='xx-large')
plt.text(335.264,147.394,r'$d_{real,(X,Y)}$',fontsize='xx-large')
plt.xlabel('X (Mpc)')
plt.ylabel('Y (Mpc)')
plt.ylim(146,149)
plt.xlim(333.5,336.5)

plt.legend(loc=1,numpoints=1,fontsize='large')
plt.show()
