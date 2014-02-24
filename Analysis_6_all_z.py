#!/usr/bin/python


import numpy as np
import scipy as sc
import pylab as plt

z_=[0,1,0.25,0.5]

for i in np.arange(4):

	halo_=sc.genfromtxt('Host_700kms_z'+str(int(i))+'.dat')
	sub_=sc.genfromtxt('Host_700kms_z'+str(int(i))+'.dat_substructure.dat')
	label_=r'V$_{max}$ > 700 km s$^{-1}$ & z='+str(z_[i]) 
	siz1, siz2=2E-2, 4E-2

#	halo_=sc.genfromtxt('Host_300kms_700kms_z'+str(int(i))+'.dat')
#	sub_=sc.genfromtxt('Host_300kms_700kms_z'+str(int(i))+'.dat_substructure.dat')
#	label_=r'300 km s$^{-1}$ < $\nu_{max}$ < 700 km s$^{-1}$ & z='+str(z_[i])
#	siz1, siz2=2E-4, 4E-4
		
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
	
	XY=np.sort(dist_2d_XY_[mask_v])
	YZ=np.sort(dist_2d_YZ_[mask_v])
	XZ=np.sort(dist_2d_XZ_[mask_v])
	XYZ=np.sort(dist_3d_[mask_v])
	
	f=plt.figure(1)
	f2=plt.figure(2)
	f3=plt.figure(3)
	f4=plt.figure(4)
	f5=plt.figure(5)
	
	a1=f.add_subplot(2,2,1)
	a2=f.add_subplot(2,2,2)
	a3=f.add_subplot(2,2,3)
	a4=f.add_subplot(2,2,4)
	a5=f2.add_subplot(1,1,1)
	a6=f3.add_subplot(1,1,1)
	a7=f4.add_subplot(1,1,1)
	a8=f5.add_subplot(1,1,1)
	
	a1.plot(dist_3d_[mask_v],V_circ_[mask_v],'.',color='red',label=label_)
	a2.plot(dist_2d_XY_[mask_v],V_circ_[mask_v],'.',color='red',label=label_)
	a3.plot(dist_2d_YZ_[mask_v],V_circ_[mask_v],'.',color='red',label=label_)
	a4.plot(dist_2d_XZ_[mask_v],V_circ_[mask_v],'.',color='red',label=label_)
	a1.legend(loc=1,numpoints=1,fontsize='medium')
	a2.legend(loc=1,numpoints=1,fontsize='medium')
	a3.legend(loc=1,numpoints=1,fontsize='medium')
	a4.legend(loc=1,numpoints=1,fontsize='medium')
	
	
	a1.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
	a2.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
	a3.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
	a4.set_ylabel(r'$\nu_{circ,sub}/\nu_{circ,Halo}$',fontsize='x-large')
	
	a1.set_xlabel(r'$d_{real, (X,Y,Z)}$ (kpch$^{-1}$) ',fontsize='x-large')
	a2.set_xlabel(r'$d_{real, (X,Y)}  $ (kpch$^{-1}$) ',fontsize='x-large')
	a3.set_xlabel(r'$d_{real, (Y,Z)}  $ (kpch$^{-1}$) ',fontsize='x-large')
	a4.set_xlabel(r'$d_{real, (X,Z)}  $ (kpch$^{-1}$) ',fontsize='x-large')
	
	
	a5.plot(XYZ,Y_,color='gray',label='(X,Y,Z)')
	a5.plot(XY,Y_,color='red',label='(X,Y)')
	a5.plot(YZ,Y_,color='green',label='(Y,Z)')
	a5.plot(XZ,Y_,color='blue',label='(X,Z)')
	a5.set_yscale('log')
	a5.set_ylabel('P(>$d$)',fontsize='x-large')
	a5.set_xlabel('$d$(kpch$^{-1}$)',fontsize='x-large')
	a5.legend(loc=1,numpoints=1,fontsize='medium')
	a5.text(55,siz1,r'$(\nu_{circ,sub}/\nu_{circ,Halo}) > 0.5$',fontsize='x-large')
	a5.text(55,siz2,label_,fontsize='x-large')
	
	
	a8.plot(XY,Y_,color='red',label='(X,Y)')
	a8.plot(YZ,Y_,color='green',label='(Y,Z)')
	a8.plot(XZ,Y_,color='blue',label='(X,Z)')
	a8.set_yscale('log')
	a8.set_ylabel('P(>$d$)',fontsize='x-large')
	a8.set_xlabel('$d_{2d}$(kpch$^{-1}$)',fontsize='x-large')
	a8.legend(loc=1,numpoints=1,fontsize='medium')
	a8.text(55,siz1,r'$(\nu_{circ,sub}/\nu_{circ,Halo}) > 0.5$',fontsize='x-large')
	a8.text(55,siz2,label_,fontsize='x-large')
	a8.axvline(x=(124./0.6777)*2.,color='black',linestyle='--')
	
	
	
	a6.plot(Xoff_old_[mask_v],Xoff_new_[mask_v],'.',color='black',label=label_)
	a6.set_xlabel(r'X$_{off,old}$',fontsize='x-large')
	a6.set_ylabel(r'X$_{off,new}$',fontsize='x-large')
	a6.legend(loc=4,numpoints=1,fontsize='medium')
	a6.set_ylim(0,1)
	
	a7.plot(dist_3d_[mask_v],Xoff_new_[mask_v],'.',color='black',label=label_)
	a7.set_ylim(0,1)
	a7.set_xlabel(r'$d_{3d}$ (kpch$^{-1}$)',fontsize='x-large')
	a7.set_ylabel(r'X$_{off,new}$',fontsize='x-large')
	a7.legend(loc=4,numpoints=1,fontsize='medium')
	
	
	a1.tick_params(labelsize='x-large')
	a2.tick_params(labelsize='x-large')
	a3.tick_params(labelsize='x-large')
	a4.tick_params(labelsize='x-large')
	a5.tick_params(labelsize='x-large')
	a6.tick_params(labelsize='x-large')
	a7.tick_params(labelsize='x-large')
	a8.tick_params(labelsize='x-large')
	
	plt.show()

#	f.savefig('figure_6_700kms_v1_z.png')
#	f2.savefig('figure6_700kms_v2_z'+str(z_[i])+'.png')
#	f3.savefig('Host_700kms_v3_z'+str(z_[i])+'.png')
#	f4.savefig('Host_700kms_v4_z'+str(z_[i])+'.png')
#	f5.savefig('Host_700kms_v5_z'+str(z_[i])+'.png')

