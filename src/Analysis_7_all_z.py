#!/usr/bin/python


import numpy as np
import scipy as sc
import pylab as plt

def fun_(var1,var2):

	halo_=sc.genfromtxt(var1)
	sub_=sc.genfromtxt(var2)

	#Variables
	
	Xoff_new_, Xoff_old_, V_circ_, dist_3d_, dist_2d_XY_, dist_2d_YZ_, dist_2d_XZ_, Rvirial_host_=np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	
	#h=1
	Parameter=0.5
	
	for i in np.arange(len(halo_[:,0])):
	
		mask=sub_[:,14]==halo_[i,11]
		N_=np.size(sub_[mask,14])
	
	#	print '['+str(i)+'/'+str(len(halo_[:,0]))+']'
	
		if N_==1.:
	
	#Parameters
	
			V_circ=float(sub_[mask,10][0])/float(halo_[i,10])
			dist_3d=np.sqrt(((float(sub_[mask,0][0])-float(halo_[i,0]))**2)+((float(sub_[mask,1][0])-float(halo_[i,1]))**2)+((float(sub_[mask,2][0])-float(halo_[i,2]))**2))*1000.
			dist_2d_XY=np.sqrt(((float(sub_[mask,0][0])-float(halo_[i,0]))**2)+((float(sub_[mask,1][0])-float(halo_[i,1]))**2))*1000.
			dist_2d_YZ=np.sqrt(((float(sub_[mask,1][0])-float(halo_[i,1]))**2)+((float(sub_[mask,2][0])-float(halo_[i,2]))**2))*1000.
			dist_2d_XZ=np.sqrt(((float(sub_[mask,0][0])-float(halo_[i,0]))**2)+((float(sub_[mask,2][0])-float(halo_[i,2]))**2))*1000.
			Rvirial_host=float(halo_[i,8])
			Xoff_new=(dist_3d/Rvirial_host)
			Xoff_old=float(halo_[i,15])
	
	#Vector
	
			Xoff_new_=np.append(Xoff_new_,Xoff_new)
			Xoff_old_=np.append(Xoff_old_,Xoff_old)
			V_circ_=np.append(V_circ_,V_circ)
			dist_3d_=np.append(dist_3d_,dist_3d)
			dist_2d_XY_=np.append(dist_2d_XY_,dist_2d_XY)
			dist_2d_YZ_=np.append(dist_2d_YZ_,dist_2d_YZ)
			dist_2d_XZ_=np.append(dist_2d_XZ_,dist_2d_XZ)
			Rvirial_host_=np.append(Rvirial_host_,Rvirial_host)
	
	mask_v=(V_circ_>=Parameter)
	N=float(len(V_circ_[mask_v]))
	Y_=1.-(np.arange(N)/N)
	XYZ=np.sort(dist_3d_[mask_v])
	XY=np.sort(dist_2d_XY_[mask_v])	

	return XYZ, XY, Y_, dist_2d_XY_, V_circ_


XYZ_h_z0,XY_h_z0,Y_h_z0,d2d_h_z0,V_h_z0=fun_('Host_700kms_z0.dat','Host_700kms_z0.dat_substructure.dat')
XYZ_h_z3,XY_h_z3,Y_h_z3,d2d_h_z3,V_h_z3=fun_('Host_700kms_z3.dat','Host_700kms_z3.dat_substructure.dat')
XYZ_h_z2,XY_h_z2,Y_h_z2,d2d_h_z2,V_h_z2=fun_('Host_700kms_z2.dat','Host_700kms_z2.dat_substructure.dat')
XYZ_h_z1,XY_h_z1,Y_h_z1,d2d_h_z1,V_h_z1=fun_('Host_700kms_z1.dat','Host_700kms_z1.dat_substructure.dat')

XYZ_s_z0,XY_s_z0,Y_s_z0,d2d_s_z0,V_s_z0=fun_('Host_300kms_700kms_z0.dat','Host_300kms_700kms_z0.dat_substructure.dat')
XYZ_s_z3,XY_s_z3,Y_s_z3,d2d_s_z3,V_s_z3=fun_('Host_300kms_700kms_z3.dat','Host_300kms_700kms_z3.dat_substructure.dat')
XYZ_s_z2,XY_s_z2,Y_s_z2,d2d_s_z2,V_s_z2=fun_('Host_300kms_700kms_z2.dat','Host_300kms_700kms_z2.dat_substructure.dat')
XYZ_s_z1,XY_s_z1,Y_s_z1,d2d_s_z1,V_s_z1=fun_('Host_300kms_700kms_z1.dat','Host_300kms_700kms_z1.dat_substructure.dat')


f1=plt.figure(1)
a1=f1.add_subplot(1,2,1)
a2=f1.add_subplot(1,2,2)

a1.plot(XY_h_z0,Y_h_z0,color='gray',label='z=0')
a1.plot(XY_h_z3,Y_h_z3,color='blue',label='z=0.25')
a1.plot(XY_h_z2,Y_h_z2,color='red',label='z=0.5')
a1.plot(XY_h_z1,Y_h_z1,color='green',label='z=1')
a1.set_yscale('log')
a1.set_ylabel(r'P(>$d_{2d,(X,Y)}$)',fontsize='x-large')
a1.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a1.legend(loc=1,numpoints=1,fontsize='medium')
#a1.text(55,4E-2,r'V$_{max}$ < 700 km s$^{-1}$',fontsize='x-large')
#a1.text(55,2E-2,r'$(V_{circ,sub}/V_{circ,Halo}) > 0.5$',fontsize='x-large')
a1.axvline(x=(124./0.6777)*2.,color='black',linestyle='--')
a1.tick_params(labelsize='x-large')


a2.plot(XY_s_z0,Y_s_z0,ls='--',color='gray',label='z=0')
a2.plot(XY_s_z3,Y_s_z3,ls='--',color='blue',label='z=0.25')
a2.plot(XY_s_z2,Y_s_z2,ls='--',color='red',label='z=0.5')
a2.plot(XY_s_z1,Y_s_z1,ls='--',color='green',label='z=1')
a2.set_yscale('log')
a2.set_ylabel(r'P(>$d_{2d,(X,Y)}$)',fontsize='x-large')
a2.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a2.legend(loc=1,numpoints=1,fontsize='medium')
#a2.text(55,2E-4,r'$(V_{circ,sub}/V_{circ,Halo}) > 0.5$',fontsize='x-large')
#a2.text(55,4E-4,r'300 km s$^{-1}$ < V$_{max}$ < 700 km s$^{-1}$',fontsize='x-large')
a2.tick_params(labelsize='x-large')


f2=plt.figure(2)
a2_1=f2.add_subplot(2,2,1)
a2_2=f2.add_subplot(2,2,2)
a2_3=f2.add_subplot(2,2,3)
a2_4=f2.add_subplot(2,2,4)

n_=20
col1='red'
col2='red'

a2_1.plot(d2d_h_z0,V_h_z0,'.',color='gray',label='z=0')
a2_1.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a2_1.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a2_1.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a2_1.legend(loc=1,numpoints=1,fontsize='medium')
a2_2.plot(d2d_h_z3,V_h_z3,'.',color='gray',label='z=0.25')
a2_2.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a2_2.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a2_2.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a2_2.legend(loc=1,numpoints=1,fontsize='medium')
a2_3.plot(d2d_h_z2,V_h_z2,'.',color='gray',label='z=0.5')
a2_3.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a2_3.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a2_3.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a2_3.legend(loc=1,numpoints=1,fontsize='medium')
a2_4.plot(d2d_h_z1,V_h_z1,'.',color='gray',label='z=1')
a2_4.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a2_4.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a2_4.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a2_4.legend(loc=1,numpoints=1,fontsize='medium')
a2_1.axhline(y=0.5,color='black',linestyle='--')
a2_2.axhline(y=0.5,color='black',linestyle='--')
a2_3.axhline(y=0.5,color='black',linestyle='--')
a2_4.axhline(y=0.5,color='black',linestyle='--')


f3=plt.figure(3)
a3_1=f3.add_subplot(2,2,1)
a3_2=f3.add_subplot(2,2,2)
a3_3=f3.add_subplot(2,2,3)
a3_4=f3.add_subplot(2,2,4)
a3_1.plot(d2d_s_z0,V_s_z0,'.',color='gray',label='z=0')
a3_1.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a3_1.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a3_1.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a3_1.legend(loc=1,numpoints=1,fontsize='medium')
a3_2.plot(d2d_s_z3,V_s_z3,'.',color='gray',label='z=0.25')
a3_2.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a3_2.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a3_2.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a3_2.legend(loc=1,numpoints=1,fontsize='medium')
a3_3.plot(d2d_s_z2,V_s_z2,'.',color='gray',label='z=0.5')
a3_3.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a3_3.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a3_3.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a3_3.legend(loc=1,numpoints=1,fontsize='medium')
a3_4.plot(d2d_s_z1,V_s_z1,'.',color='gray',label='z=1')
a3_4.plot((124./0.6777)*2.,0.54,'*',mec=col1,ms=n_,color=col2)
a3_4.set_xlabel(r'$d_{2d,(X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
a3_4.set_ylabel(r'$(\nu_{circ,sub}/\nu_{circ,Halo})$',fontsize='x-large')
a3_4.legend(loc=1,numpoints=1,fontsize='medium')
a3_1.axhline(y=0.5,color='black',linestyle='--')
a3_2.axhline(y=0.5,color='black',linestyle='--')
a3_3.axhline(y=0.5,color='black',linestyle='--')
a3_4.axhline(y=0.5,color='black',linestyle='--')

a1.tick_params(labelsize='x-large')
a2.tick_params(labelsize='x-large')
a2_1.tick_params(labelsize='x-large')
a2_2.tick_params(labelsize='x-large')
a2_3.tick_params(labelsize='x-large')
a2_4.tick_params(labelsize='x-large')
a3_1.tick_params(labelsize='x-large')
a3_2.tick_params(labelsize='x-large')
a3_3.tick_params(labelsize='x-large')
a3_4.tick_params(labelsize='x-large')

plt.show()
