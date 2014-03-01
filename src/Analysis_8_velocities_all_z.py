#!/usr/bin/python


import numpy as np
import scipy as sc
import pylab as plt

def fun_(var1,var2):

	halo_=sc.genfromtxt(var1)
	sub_=sc.genfromtxt(var2)

	#Variables
	
	Xoff_new_, V_circ_,dist_3d_,dist_2d_XY_, dist_2d_YZ_, dist_2d_XZ_=np.array([]), np.array([]), np.array([]),np.array([]), np.array([]), np.array([])#, Xoff_old_, V_circ_, dist_3d_, dist_2d_XY_, dist_2d_YZ_, dist_2d_XZ_, Rvirial_host_=np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	angulo_=np.array([])	
	#h=1
	Parameter=0.
	
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
			Vx, Vy, Vz=float(sub_[mask,3][0])-float(halo_[i,3]), float(sub_[mask,4][0])-float(halo_[i,4]), float(sub_[mask,5][0])-float(halo_[i,5])
			X_sh, Y_sh, Z_sh=(float(sub_[mask,0][0])-float(halo_[i,0]))*1000., (float(sub_[mask,1][0])-float(halo_[i,1]))*1000., (float(sub_[mask,2][0])-float(halo_[i,2]))*1000.
			V_r=(Vx*X_sh)+(Vy*Y_sh)+(Vz*Z_sh) # Escalar Product
			d_norm=np.sqrt(X_sh**2+Y_sh**2+Z_sh**2)
			V_norm=np.sqrt(Vx**2+Vy**2+Vz**2)
			angulo=V_r/(d_norm*V_norm)
			Rvirial_host=float(halo_[i,8])
			Xoff_new=(dist_3d/Rvirial_host)
			Xoff_old=float(halo_[i,15])
	
	#Vector
	
			angulo_=np.append(angulo_,angulo)
			Xoff_new_=np.append(Xoff_new_,Xoff_new)
#			Xoff_old_=np.append(Xoff_old_,Xoff_old)
			V_circ_=np.append(V_circ_,V_circ)
			dist_3d_=np.append(dist_3d_,dist_3d)
			dist_2d_XY_=np.append(dist_2d_XY_,dist_2d_XY)
			dist_2d_YZ_=np.append(dist_2d_YZ_,dist_2d_YZ)
			dist_2d_XZ_=np.append(dist_2d_XZ_,dist_2d_XZ)
#			Rvirial_host_=np.append(Rvirial_host_,Rvirial_host)
	
	mask_v=(V_circ_>=Parameter)
#	N=float(len(V_circ_[mask_v]))
#	Y_=1.-(np.arange(N)/N)
#	XYZ=np.sort(dist_3d_[mask_v])
#	XY=np.sort(dist_2d_XY_[mask_v])	

	return angulo_[mask_v], Xoff_new_[mask_v], dist_3d_[mask_v], dist_2d_XY_[mask_v], dist_2d_YZ_[mask_v], dist_2d_XZ_[mask_v]


#angulo_s_s_z0,Xoffnew_s_z0,dist_3d_s_z0,dist_2d_XY_s_z0, dist_2d_YZ_s_z0, dist_2d_XZ_s_z0=fun_('Host_700kms_z0.dat','Host_700kms_z0.dat_substructure.dat')
#angulo_s_s_z3,Xoffnew_s_z3,dist_3d_s_z3,dist_2d_XY_s_z3, dist_2d_YZ_s_z3, dist_2d_XZ_s_z3=fun_('Host_700kms_z3.dat','Host_700kms_z3.dat_substructure.dat')
#angulo_s_s_z2,Xoffnew_s_z2,dist_3d_s_z2,dist_2d_XY_s_z2, dist_2d_YZ_s_z2, dist_2d_XZ_s_z2=fun_('Host_700kms_z2.dat','Host_700kms_z2.dat_substructure.dat')
#angulo_s_s_z1,Xoffnew_s_z1,dist_3d_s_z1,dist_2d_XY_s_z1, dist_2d_YZ_s_z1, dist_2d_XZ_s_z1=fun_('Host_700kms_z1.dat','Host_700kms_z1.dat_substructure.dat')

angulo_s_s_z0,Xoffnew_s_z0,dist_3d_s_z0,dist_2d_XY_s_z0, dist_2d_YZ_s_z0, dist_2d_XZ_s_z0=fun_('Host_300kms_700kms_z0.dat','Host_300kms_700kms_z0.dat_substructure.dat')
angulo_s_s_z3,Xoffnew_s_z3,dist_3d_s_z3,dist_2d_XY_s_z3, dist_2d_YZ_s_z3, dist_2d_XZ_s_z3=fun_('Host_300kms_700kms_z3.dat','Host_300kms_700kms_z3.dat_substructure.dat')
angulo_s_s_z2,Xoffnew_s_z2,dist_3d_s_z2,dist_2d_XY_s_z2, dist_2d_YZ_s_z2, dist_2d_XZ_s_z2=fun_('Host_300kms_700kms_z2.dat','Host_300kms_700kms_z2.dat_substructure.dat')
angulo_s_s_z1,Xoffnew_s_z1,dist_3d_s_z1,dist_2d_XY_s_z1, dist_2d_YZ_s_z1, dist_2d_XZ_s_z1=fun_('Host_300kms_700kms_z1.dat','Host_300kms_700kms_z1.dat_substructure.dat')

f=plt.figure(1)

a1=f.add_subplot(2,2,1)
a2=f.add_subplot(2,2,2)
a3=f.add_subplot(2,2,3)
a4=f.add_subplot(2,2,4)


a1.plot(Xoffnew_s_z0,angulo_s_s_z0,'.',color='gray',label='z=0')
a1.legend(loc=1,numpoints=1,fontsize='medium')
#a1.set_xlabel(r'$X_{off,new}$',fontsize='x-large')
a1.set_ylabel(r'$cos(\theta)$',fontsize='x-large')
a1.tick_params(labelsize='x-large')
a1.axhline(y=0.5,color='red',lw=3.,linestyle='--')

a2.plot(Xoffnew_s_z3,angulo_s_s_z3,'.',color='gray',label='z=0.25')
a2.legend(loc=1,numpoints=1,fontsize='medium')
#a2.set_xlabel(r'$X_{off,new}$',fontsize='x-large')
#a2.set_ylabel(r'$cos(\theta)$',fontsize='x-large')
a2.tick_params(labelsize='x-large')
a2.axhline(y=0.5,color='red',lw=3.,linestyle='--')

a3.plot(Xoffnew_s_z2,angulo_s_s_z2,'.',color='gray',label='z=0.5')
a3.legend(loc=1,numpoints=1,fontsize='medium')
a3.set_xlabel(r'$X_{off,new}$',fontsize='x-large')
a3.set_ylabel(r'$cos(\theta)$',fontsize='x-large')
a3.tick_params(labelsize='x-large')
a3.axhline(y=0.5,color='red',lw=3.,linestyle='--')

a4.plot(Xoffnew_s_z1,angulo_s_s_z1,'.',color='gray',label='z=1')
a4.legend(loc=1,numpoints=1,fontsize='medium')
a4.set_xlabel(r'$X_{off,new}$',fontsize='x-large')
#a4.set_ylabel(r'$cos(\theta)$',fontsize='x-large')
a4.tick_params(labelsize='x-large')
a4.axhline(y=0.5,color='red',lw=3.,linestyle='--')

a1.set_xlim(0,1)
a2.set_xlim(0,1)
a3.set_xlim(0,1)
a4.set_xlim(0,1)

a1.tick_params(labelsize='x-large')
a2.tick_params(labelsize='x-large')
a3.tick_params(labelsize='x-large')
a4.tick_params(labelsize='x-large')

#Plot for figure(2)

f1=plt.figure(2)
af1=f1.add_subplot(1,1,1)

cos_theta=0.5
mask_ang0=(angulo_s_s_z0>=cos_theta)
N=float(len(Xoffnew_s_z0[mask_ang0]))
Y_z0=1-(np.arange(N)/N)
mask_ang3=(angulo_s_s_z3>=cos_theta)
N=float(len(Xoffnew_s_z3[mask_ang3]))
Y_z3=1-(np.arange(N)/N)
mask_ang2=(angulo_s_s_z2>=cos_theta)
N=float(len(Xoffnew_s_z2[mask_ang2]))
Y_z2=1-(np.arange(N)/N)
mask_ang1=(angulo_s_s_z1>=cos_theta)
N=float(len(Xoffnew_s_z1[mask_ang1]))
Y_z1=1-(np.arange(N)/N)

af1.plot(np.sort(Xoffnew_s_z0[mask_ang0]),Y_z0,lw=3,ls='--',color='black',label='z=0')
af1.plot(np.sort(Xoffnew_s_z3[mask_ang3]),Y_z3,lw=3,ls='-',color='gray',label='z=0.25')
af1.plot(np.sort(Xoffnew_s_z2[mask_ang2]),Y_z2,lw=3,ls=':',color='gray',label='z=0.5')
af1.plot(np.sort(Xoffnew_s_z1[mask_ang1]),Y_z1,lw=3,ls='-.',color='black',label='z=1')
af1.set_ylabel(r'P(>X$_{off,new}$)',fontsize='x-large')
af1.set_xlabel(r'X$_{off,new}$',fontsize='x-large')
af1.tick_params(labelsize='x-large')
af1.legend(loc=1,numpoints=1,fontsize='medium')
af1.set_yscale('log')

f2=plt.figure(3)
#af21=f2.add_subplot(2,2,1)
af22=f2.add_subplot(1,1,1)
#af23=f2.add_subplot(2,2,3)
#af24=f2.add_subplot(2,2,4)

#af21.plot(np.sort(dist_3d_s_z0[mask_ang0]),Y_z0,lw=3,ls='--',color='black',label='z=0')
#af21.plot(np.sort(dist_3d_s_z3[mask_ang3]),Y_z3,lw=3,ls='-',color='gray',label='z=0.25')
#af21.plot(np.sort(dist_3d_s_z2[mask_ang2]),Y_z2,lw=3,ls=':',color='gray',label='z=0.5')
#af21.plot(np.sort(dist_3d_s_z1[mask_ang1]),Y_z1,lw=3,ls='-.',color='black',label='z=1')
#af21.set_ylabel(r'P(>$d_{real}$)',fontsize='x-large')
#af21.set_xlabel(r'$d_{real}$ (kpch$^{-1}$)',fontsize='x-large')
#af21.tick_params(labelsize='x-large')
#af21.legend(loc=1,numpoints=1,fontsize='medium')

af22.plot(np.sort(dist_2d_XY_s_z0[mask_ang0]),Y_z0,lw=3,ls='--',color='black',label='z=0')
af22.plot(np.sort(dist_2d_XY_s_z3[mask_ang3]),Y_z3,lw=3,ls='-',color='gray',label='z=0.25')
af22.plot(np.sort(dist_2d_XY_s_z2[mask_ang2]),Y_z2,lw=3,ls=':',color='gray',label='z=0.5')
af22.plot(np.sort(dist_2d_XY_s_z1[mask_ang1]),Y_z1,lw=3,ls='-.',color='black',label='z=1')
af22.set_ylabel(r'P(>$d_{real, (X,Y)}$)',fontsize='x-large')
af22.set_xlabel(r'$d_{real, (X,Y)}$ (kpch$^{-1}$)',fontsize='x-large')
af22.tick_params(labelsize='x-large')
af22.legend(loc=1,numpoints=1,fontsize='medium')
af22.set_yscale('log')
af22.axvline(x=(124./0.6777)*2.,color='gray',linestyle='-')


#af23.plot(np.sort(dist_2d_YZ_s_z0[mask_ang0]),Y_z0,lw=3,ls='--',color='black',label='z=0')
#af23.plot(np.sort(dist_2d_YZ_s_z3[mask_ang3]),Y_z3,lw=3,ls='-',color='gray',label='z=0.25')
#af23.plot(np.sort(dist_2d_YZ_s_z2[mask_ang2]),Y_z2,lw=3,ls=':',color='gray',label='z=0.5')
#af23.plot(np.sort(dist_2d_YZ_s_z1[mask_ang1]),Y_z1,lw=3,ls='-.',color='black',label='z=1')
#af23.set_ylabel(r'P(>$d_{real,(Y,Z)}$)',fontsize='x-large')
#af23.set_xlabel(r'$d_{real,(Y,Z)}$ (kpch$^{-1}$)',fontsize='x-large')
#af23.tick_params(labelsize='x-large')
#af23.legend(loc=1,numpoints=1,fontsize='medium')
#af23.axvline(x=(124./0.6777)*2.,color='gray',linestyle='-')

#af24.plot(np.sort(dist_2d_XZ_s_z0[mask_ang0]),Y_z0,lw=3,ls='--',color='black',label='z=0')
#af24.plot(np.sort(dist_2d_XZ_s_z3[mask_ang3]),Y_z3,lw=3,ls='-',color='gray',label='z=0.25')
#af24.plot(np.sort(dist_2d_XZ_s_z2[mask_ang2]),Y_z2,lw=3,ls=':',color='gray',label='z=0.5')
#af24.plot(np.sort(dist_2d_XZ_s_z1[mask_ang1]),Y_z1,lw=3,ls='-.',color='black',label='z=1')
#af24.set_ylabel(r'P(>$d_{real,(X,Z)}$)',fontsize='x-large')
#af24.set_xlabel(r'$d_{real,(X,Z)}$ (kpch$^{-1}$)',fontsize='x-large')
#af24.tick_params(labelsize='x-large')
#af24.legend(loc=1,numpoints=1,fontsize='medium')
#af24.axvline(x=(124./0.6777)*2.,color='gray',linestyle='-')


plt.show()
