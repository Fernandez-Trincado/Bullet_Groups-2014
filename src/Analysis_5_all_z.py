#!/usr/bin/python


#Bullet Cluster and Bullet Groups with Jaime Forero at Universidad de Los Andes, Bogota - Colombia.

import numpy as np
import scipy as sc
import pylab as plt
import sys


def dreal(file1,file2):

	halo=sc.genfromtxt(file1,dtype=float)
	sub=sc.genfromtxt(file2,dtype=float)
	d_real, Vcirc, Vreal=np.array([]),np.array([]), np.array([])
	Xoff_halo, Xoff_sub=np.array([]), np.array([])

	for i in np.arange(len(halo[:,0])):
	
		mask=sub[:,14]==halo[i,11]
		N=np.size(sub[mask,14])
	
		if N >= 1.:
		
			d_real=np.append(d_real,np.sqrt(((float(sub[mask,0][0])-float(halo[i,0]))**2)+((float(sub[mask,1][0])-float(halo[i,1]))**2)+((float(sub[mask,2][0])-float(halo[i,2]))**2))*1000.)
			Vcirc=np.append(Vcirc,float(sub[mask,10][0])/float(halo[i,10]))
			Vreal=np.append(Vreal,(np.sqrt((float(sub[mask,3][0])-float(halo[i,3]))**2+(float(sub[mask,4][0])-float(halo[i,4]))**2+(float(sub[mask,5][0])-float(halo[i,5]))**2)/float(halo[i,10])))
			Xoff_halo=np.append(Xoff_halo,float(halo[i,15]))
			Xoff_sub=np.append(Xoff_sub,float(sub[mask,15][0]))
			
			N=float(len(d_real))
			Y_=1.-(np.arange(N)/N)
			d_real=np.sort(d_real)
	
	return d_real, Y_, Vcirc, Vreal, Xoff_halo, Xoff_sub

d_real1_z0, Y_1_z0, V_circ1_z0, Vreal1_z0, Xoff_halo1_z0, Xoff_sub1_z0=dreal('Host_700kms_z0.dat','Host_700kms_z0.dat_substructure.dat')
d_real2_z0, Y_2_z0, V_circ2_z0, Vreal2_z0, Xoff_halo2_z0, Xoff_sub2_z0=dreal('Host_300kms_700kms_z0.dat','Host_300kms_700kms_z0.dat_substructure.dat')
d_real1_z1, Y_1_z1, V_circ1_z1, Vreal1_z1, Xoff_halo1_z1, Xoff_sub1_z1=dreal('Host_700kms_z1.dat','Host_700kms_z1.dat_substructure.dat')
d_real2_z1, Y_2_z1, V_circ2_z1, Vreal2_z1, Xoff_halo2_z1, Xoff_sub2_z1=dreal('Host_300kms_700kms_z1.dat','Host_300kms_700kms_z1.dat_substructure.dat')
d_real1_z2, Y_1_z2, V_circ1_z2, Vreal1_z2, Xoff_halo1_z2, Xoff_sub1_z2=dreal('Host_700kms_z2.dat','Host_700kms_z2.dat_substructure.dat')
d_real2_z2, Y_2_z2, V_circ2_z2, Vreal2_z2, Xoff_halo2_z2, Xoff_sub2_z2=dreal('Host_300kms_700kms_z2.dat','Host_300kms_700kms_z2.dat_substructure.dat')
d_real1_z3, Y_1_z3, V_circ1_z3, Vreal1_z3, Xoff_halo1_z3, Xoff_sub1_z3=dreal('Host_700kms_z3.dat','Host_700kms_z3.dat_substructure.dat')
d_real2_z3, Y_2_z3, V_circ2_z3, Vreal2_z3, Xoff_halo2_z3, Xoff_sub2_z3=dreal('Host_300kms_700kms_z3.dat','Host_300kms_700kms_z3.dat_substructure.dat')


f1=plt.figure(1)
f2=plt.figure(2)
f3=plt.figure(3)
f4=plt.figure(4)

af1_m1=f1.add_subplot(1,2,1)
af1_m2=f1.add_subplot(1,2,2)
af2_m1=f2.add_subplot(1,2,1)
af2_m2=f2.add_subplot(1,2,2)
af3_m1=f3.add_subplot(1,2,1)
af3_m2=f3.add_subplot(1,2,2)

af4_m1=f4.add_subplot(2,4,1)
af4_m2=f4.add_subplot(2,4,2)
af4_m3=f4.add_subplot(2,4,3)
af4_m4=f4.add_subplot(2,4,4)
af4_m5=f4.add_subplot(2,4,5)
af4_m6=f4.add_subplot(2,4,6)
af4_m7=f4.add_subplot(2,4,7)
af4_m8=f4.add_subplot(2,4,8)


#Figure 1

af1_m1.plot(d_real1_z0,Y_1_z0,color='gray',label='z=0')
af1_m1.plot(d_real1_z3,Y_1_z3,color='blue',label='z=0.25')
af1_m1.plot(d_real1_z2,Y_1_z2,color='red',label='z=0.5')
af1_m1.plot(d_real1_z1,Y_1_z1,color='green',label='z=1')

af1_m2.plot(d_real2_z0,Y_2_z0,color='gray',ls='--',label='z=0')
af1_m2.plot(d_real2_z3,Y_2_z3,color='blue',ls='--',label='z=0.25')
af1_m2.plot(d_real2_z2,Y_2_z2,color='red',ls='--',label='z=0.5')
af1_m2.plot(d_real2_z1,Y_2_z1,color='green',ls='--',label='z=1')

af1_m1.set_xlabel(r'$d_{real}$ (kpch$^{-1}$)',fontsize='x-large')
af1_m1.set_ylabel(r'P(>$d$)',fontsize='x-large')
af1_m1.legend(loc=1,numpoints=1,fontsize='medium')
af1_m2.set_xlabel(r'$d_{real}$ (kpch$^{-1}$)',fontsize='x-large')
af1_m2.set_ylabel(r'P(>$d$)',fontsize='x-large')
af1_m2.legend(loc=1,numpoints=1,fontsize='medium')

#Figure 2

af2_m1.hist(V_circ1_z0,edgecolor='gray',fill=False,label='z=0',histtype='stepfilled')
af2_m1.hist(V_circ1_z3,edgecolor='blue',fill=False,label='z=0.25',histtype='stepfilled')
af2_m1.hist(V_circ1_z2,edgecolor='red',fill=False,label='z=0.5',histtype='stepfilled')
af2_m1.hist(V_circ1_z1,edgecolor='green',fill=False,label='z=1',histtype='stepfilled')

af2_m2.hist(V_circ2_z0,edgecolor='gray',ls='dashed',fill=False,label='z=0',histtype='stepfilled')
af2_m2.hist(V_circ2_z3,edgecolor='blue',ls='dashed',fill=False,label='z=0.25',histtype='stepfilled')
af2_m2.hist(V_circ2_z2,edgecolor='red',ls='dashed',fill=False,label='z=0.5',histtype='stepfilled')
af2_m2.hist(V_circ2_z1,edgecolor='green',ls='dashed',fill=False,label='z=1',histtype='stepfilled')

af2_m1.legend(loc=1,numpoints=1,fontsize='medium')
af2_m1.set_xlabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
af2_m1.set_ylabel('N',fontsize='x-large')

af2_m2.legend(loc=1,numpoints=1,fontsize='medium')
af2_m2.set_xlabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
af2_m2.set_ylabel('N',fontsize='x-large')

#Figure 3

af3_m1.hist(Vreal1_z0,edgecolor='gray',fill=False,label='z=0',histtype='stepfilled')
af3_m1.hist(Vreal1_z3,edgecolor='blue',fill=False,label='z=0.25',histtype='stepfilled')
af3_m1.hist(Vreal1_z2,edgecolor='red',fill=False,label='z=0.5',histtype='stepfilled')
af3_m1.hist(Vreal1_z1,edgecolor='green',fill=False,label='z=1',histtype='stepfilled')

af3_m2.hist(Vreal2_z0,ls='dashed',edgecolor='gray',fill=False,label='z=0',histtype='stepfilled')
af3_m2.hist(Vreal2_z3,ls='dashed',edgecolor='blue',fill=False,label='z=0.25',histtype='stepfilled')
af3_m2.hist(Vreal2_z2,ls='dashed',edgecolor='red',fill=False,label='z=0.5',histtype='stepfilled')
af3_m2.hist(Vreal2_z1,ls='dashed',edgecolor='green',fill=False,label='z=1',histtype='stepfilled')

af3_m1.legend(loc=1,numpoints=1,fontsize='medium')
af3_m1.set_xlabel(r'$| \nu |/\nu_{circ,Halo}$',fontsize='x-large')
af3_m1.set_ylabel('N',fontsize='x-large')
af3_m2.legend(loc=1,numpoints=1,fontsize='medium')
af3_m2.set_xlabel(r'$| \nu |/\nu_{circ,Halo}$',fontsize='x-large')
af3_m2.set_ylabel('N',fontsize='x-large')

#Figure 4

af4_m1.plot(Xoff_halo1_z0,V_circ1_z0,'.',color='gray',label='z=0')
af4_m2.plot(Xoff_halo1_z3,V_circ1_z3,'.',color='gray',label='z=0.25')
af4_m3.plot(Xoff_halo1_z2,V_circ1_z2,'.',color='gray',label='z=0.5')
af4_m4.plot(Xoff_halo1_z1,V_circ1_z1,'.',color='gray',label='z=1')

af4_m5.plot(Xoff_halo2_z0,V_circ2_z0,'.',color='red',label='z=0')
af4_m6.plot(Xoff_halo2_z3,V_circ2_z3,'.',color='red',label='z=0.25')
af4_m7.plot(Xoff_halo2_z2,V_circ2_z2,'.',color='red',label='z=0.5')
af4_m8.plot(Xoff_halo2_z1,V_circ2_z1,'.',color='red',label='z=1')

af4_m1.legend(loc=1,numpoints=1,fontsize='medium')
af4_m2.legend(loc=1,numpoints=1,fontsize='medium')
af4_m3.legend(loc=1,numpoints=1,fontsize='medium')
af4_m4.legend(loc=1,numpoints=1,fontsize='medium')
af4_m5.legend(loc=1,numpoints=1,fontsize='medium')
af4_m6.legend(loc=1,numpoints=1,fontsize='medium')
af4_m7.legend(loc=1,numpoints=1,fontsize='medium')
af4_m8.legend(loc=1,numpoints=1,fontsize='medium')

af4_m1.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
af4_m5.set_xlabel(r'X$_{off}$',fontsize='x-large')
af4_m5.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
af4_m6.set_xlabel(r'X$_{off}$',fontsize='x-large')
af4_m7.set_xlabel(r'X$_{off}$',fontsize='x-large')
af4_m8.set_xlabel(r'X$_{off}$',fontsize='x-large')

af1_m1.tick_params(labelsize='medium')
af2_m1.tick_params(labelsize='medium')
af3_m1.tick_params(labelsize='medium')
af1_m2.tick_params(labelsize='medium')
af2_m2.tick_params(labelsize='medium')
af3_m2.tick_params(labelsize='medium')
af4_m1.tick_params(labelsize='small')
af4_m2.tick_params(labelsize='small')
af4_m3.tick_params(labelsize='small')
af4_m4.tick_params(labelsize='small')
af4_m5.tick_params(labelsize='small')
af4_m6.tick_params(labelsize='small')
af4_m7.tick_params(labelsize='small')
af4_m8.tick_params(labelsize='small')


plt.show()
