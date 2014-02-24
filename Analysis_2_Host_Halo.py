#!/usr/bin/python

#Bullet Cluster and Bullet Groups with Jaime Forero at Universidad de Los Andes, Bogota - Colombia.

import numpy as np
import scipy as sc 
import pylab as plt

data_host=sc.genfromtxt('Host_1000kms.dat')
data_sub=sc.genfromtxt('Host_350kms_1000kms.dat')

f=plt.figure(1)

ax1=f.add_subplot(1,2,1)
ax2=f.add_subplot(1,2,2)
#ax3=f.add_subplot(2,2,3)
#ax4=f.add_subplot(2,2,4)

N_host, N_sub=float(len(data_host[:,15])), float(len(data_sub[:,15]))

Y_host=1.-(np.arange(N_host))/N_host
Y_sub=1.-(np.arange(N_sub))/N_sub


#figure 1

data_host_n, data_sub_n=np.argsort(data_host[:,15]), np.argsort(data_sub[:,15])

Xoff_host, Xoff_sub=np.array([]), np.array([])

for i in np.arange(len(data_host_n)): Xoff_host=np.append(Xoff_host,data_host[data_host_n[i],15])
for k in np.arange(len(data_sub_n)):  Xoff_sub=np.append(Xoff_sub,data_sub[data_sub_n[k],15])

ax1.plot(Xoff_host,Y_host,color='gray')
ax1.plot(Xoff_sub,Y_sub,color='red')
ax1.set_xlabel(r'X$_{off}$')
ax1.set_ylabel(r'P(>X$_{off}$)')
ax1.set_yscale('log')

#figure2

data_host_n, data_sub_n=np.argsort(data_host[:,8]), np.argsort(data_sub[:,8])

Rvirial_host, Rvirial_sub=np.array([]), np.array([])

for i in np.arange(len(data_host_n)): Rvirial_host=np.append(Rvirial_host,data_host[data_host_n[i],8])
for k in np.arange(len(data_sub_n)):  Rvirial_sub=np.append(Rvirial_sub,data_sub[data_sub_n[k],8])

ax2.plot(Xoff_host*Rvirial_host/0.6777,Y_host,color='gray',label=r'$V_{max}>1000 km s^{-1}$')
ax2.plot(Xoff_sub*Rvirial_sub/0.6777,Y_sub,color='red',label=r'$350 km s^{-1} < V_{max} < 1000 km s^{-1}$')
ax2.set_xlabel(r'd [kpc]')
ax2.set_ylabel(r'P(>d)')
ax2.set_yscale('log')
plt.legend(loc=1,numpoints=1,fontsize='large')

plt.show()
