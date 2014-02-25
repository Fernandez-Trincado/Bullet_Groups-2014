#!/usr/bin/python

#Bullet Cluster and Bullet Groups with Jaime Forero at Universidad de Los Andes, Bogota - Colombia.

import numpy as np
import scipy as sc 
import pylab as plt

data=sc.genfromtxt('MassiveCatshortV.0416_mass_1013.DAT')

mask1013=data[:,6]<1E14
mask1014=data[:,6]>1E14

f=plt.figure(1)

ax1=f.add_subplot(1,2,1)
ax2=f.add_subplot(1,2,2)

N, N13=float(len(data[mask1013,15])), float(len(data[mask1014,15]))

Y_axis=1.-(np.arange(N))/N
Y_axis13=1.-(np.arange(N13))/N13

#figure 1

data_ord1, data_ord13=np.argsort(data[mask1013,15]), np.argsort(data[mask1014,15])


Xoff,Xoff13=np.array([]), np.array([])

for i in np.arange(len(data_ord1)): Xoff=np.append(Xoff,data[mask1013,15][data_ord1[i]])
for k in np.arange(len(data_ord13)): Xoff13=np.append(Xoff13,data[mask1014,15][data_ord13[k]])

ax1.plot(Xoff,Y_axis,color='gray',label=r'$M>10^{13} M_{\odot}$')
ax1.plot(Xoff13,Y_axis13,color='red',label=r'$M>10^{14} M_{\odot}$')
ax1.set_xlabel(r'X$_{off}$')
ax1.set_ylabel(r'P(>X$_{off}$)')
ax1.set_yscale('log')

#figure2

data_ord2, data_ord13_2=np.argsort(data[mask1013,8]), np.argsort(data[mask1014,8])

Rvirial, Rvirial13=np.array([]), np.array([])

for i in np.arange(len(data_ord2)): Rvirial=np.append(Rvirial,data[mask1013,8][data_ord2[i]])
for k in np.arange(len(data_ord13_2)): Rvirial13=np.append(Rvirial13,data[mask1014,8][data_ord13_2[k]])

ax2.plot(Xoff*Rvirial/0.6777,Y_axis,color='gray',label=r'$10^{13} M_{\odot}<M<10^{14} M_{\odot}$')
ax2.plot(Xoff13*Rvirial13/0.6777,Y_axis13,color='red',label=r'$M>10^{14} M_{\odot}$')
ax2.set_xlabel(r'd [kpc]')
ax2.set_ylabel(r'P(>d)')
ax2.set_yscale('log')
plt.legend(loc=1,numpoints=1,fontsize='large')

plt.show()
