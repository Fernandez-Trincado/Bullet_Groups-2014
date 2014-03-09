#!/usr/bin/python


import numpy as np
import scipy as sc
import pylab as plt
from matplotlib.colors import LogNorm

def fun_(var1,var2):

	halo_=sc.genfromtxt(var1)
	sub_=sc.genfromtxt(var2)

	#Variables
	
	Xoff_new_, V_circ_,dist_3d_,dist_2d_XY_, dist_2d_YZ_, dist_2d_XZ_=np.array([]), np.array([]), np.array([]),np.array([]), np.array([]), np.array([])#, Xoff_old_, V_circ_, dist_3d_, dist_2d_XY_, dist_2d_YZ_, dist_2d_XZ_, Rvirial_host_=np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	angulo_,dist_bar_,dist_barion,Doff_=np.array([]),np.array([]),np.array([]),np.array([])	
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
			Xoff_new=(dist_2d_XY/Rvirial_host)
			Xoff_old=float(halo_[i,15])
			dist_bar=Xoff_old*Rvirial_host	
			Doff=dist_3d/Rvirial_host	

	#Vector
	
			dist_bar_=np.append(dist_bar_,dist_bar)
			angulo_=np.append(angulo_,angulo)
			Xoff_new_=np.append(Xoff_new_,Xoff_new)
#			Xoff_old_=np.append(Xoff_old_,Xoff_old)
			V_circ_=np.append(V_circ_,V_circ)
			dist_3d_=np.append(dist_3d_,dist_3d)
			dist_2d_XY_=np.append(dist_2d_XY_,dist_2d_XY)
			dist_2d_YZ_=np.append(dist_2d_YZ_,dist_2d_YZ)
			dist_2d_XZ_=np.append(dist_2d_XZ_,dist_2d_XZ)
#			Rvirial_host_=np.append(Rvirial_host_,Rvirial_host)
			Doff_=np.append(Doff_,Doff)
	
	mask_v=(V_circ_>=Parameter)

	mask_angul=angulo_<=-0.9
	mask_Doff1=Doff_[mask_angul]>=0.6
	mask_Doff2=Doff_[mask_angul]<0.6
	mask_angul2=(angulo_>-0.9)&(angulo_<0.9)
	mask_angul3=angulo_>=0.9

	dist_barion=np.append(dist_barion,dist_bar_[mask_angul][mask_Doff2])
	dist_barion=np.append(dist_barion,dist_bar_[mask_angul3])

	for i in np.arange(len(Doff_[mask_angul][mask_Doff1])): dist_barion=np.append(dist_barion,0.)
	for j in np.arange(len(Doff_[mask_angul2])): dist_barion=np.append(dist_barion,0.) 


#	N=float(len(V_circ_[mask_v]))
#	Y_=1.-(np.arange(N)/N)
#	XYZ=np.sort(dist_3d_[mask_v])
#	XY=np.sort(dist_2d_XY_[mask_v])	

	return angulo_[mask_v], Xoff_new_[mask_v], dist_3d_[mask_v], dist_2d_XY_[mask_v]/1000., dist_2d_YZ_[mask_v], dist_2d_XZ_[mask_v], V_circ_[mask_v], dist_barion

angulo_h_z0,Xoffnew_h_z0,dist_3d_h_z0,dist_2d_XY_h_z0, dist_2d_YZ_h_z0, dist_2d_XZ_h_z0,V_circ_h_z0,distbarion_h_z0=fun_('Host_700kms_z0.dat','Host_700kms_z0.dat_substructure.dat')
angulo_h_z3,Xoffnew_h_z3,dist_3d_h_z3,dist_2d_XY_h_z3, dist_2d_YZ_h_z3, dist_2d_XZ_h_z3,V_circ_h_z3,distbarion_h_z3=fun_('Host_700kms_z3.dat','Host_700kms_z3.dat_substructure.dat')
angulo_h_z2,Xoffnew_h_z2,dist_3d_h_z2,dist_2d_XY_h_z2, dist_2d_YZ_h_z2, dist_2d_XZ_h_z2,V_circ_h_z2,distbarion_h_z2=fun_('Host_700kms_z2.dat','Host_700kms_z2.dat_substructure.dat')
angulo_h_z1,Xoffnew_h_z1,dist_3d_h_z1,dist_2d_XY_h_z1, dist_2d_YZ_h_z1, dist_2d_XZ_h_z1,V_circ_h_z1,distbarion_h_z1=fun_('Host_700kms_z1.dat','Host_700kms_z1.dat_substructure.dat')
                                                                                                                   
angulo_s_z0,Xoffnew_s_z0,dist_3d_s_z0,dist_2d_XY_s_z0, dist_2d_YZ_s_z0, dist_2d_XZ_s_z0,V_circ_s_z0,distbarion_s_z0=fun_('Host_300kms_700kms_z0.dat','Host_300kms_700kms_z0.dat_substructure.dat')
angulo_s_z3,Xoffnew_s_z3,dist_3d_s_z3,dist_2d_XY_s_z3, dist_2d_YZ_s_z3, dist_2d_XZ_s_z3,V_circ_s_z3,distbarion_s_z3=fun_('Host_300kms_700kms_z3.dat','Host_300kms_700kms_z3.dat_substructure.dat')
angulo_s_z2,Xoffnew_s_z2,dist_3d_s_z2,dist_2d_XY_s_z2, dist_2d_YZ_s_z2, dist_2d_XZ_s_z2,V_circ_s_z2,distbarion_s_z2=fun_('Host_300kms_700kms_z2.dat','Host_300kms_700kms_z2.dat_substructure.dat')
angulo_s_z1,Xoffnew_s_z1,dist_3d_s_z1,dist_2d_XY_s_z1, dist_2d_YZ_s_z1, dist_2d_XZ_s_z1,V_circ_s_z1,distbarion_s_z1=fun_('Host_300kms_700kms_z1.dat','Host_300kms_700kms_z1.dat_substructure.dat')

#f=plt.figure(1,(25,8))
#---------------------------------------------
size_=18

f=plt.figure(1,(18,6))

#------------------------------------------------------------------------------------------------------------------------------------
a1=f.add_subplot(1,2,1)
#------------------------------------------------------------------------------------------------------------------------------------

X_d2d=np.append(dist_2d_XY_h_z0,dist_2d_XY_h_z3)
X_d2d=np.append(X_d2d,dist_2d_XY_h_z2)
X_d2d=np.append(X_d2d,dist_2d_XY_h_z1)
#X_d2d=np.append(X_d2d,np.linspace(0,1.77,5))
#X_d2d=np.append(X_d2d,np.linspace(0,1.77,5))
V_c=np.append(V_circ_h_z0,V_circ_h_z3)
V_c=np.append(V_c,V_circ_h_z2)
V_c=np.append(V_c,V_circ_h_z1)
#V_c=np.append(V_c,np.linspace(0,0.1,5))
#V_c=np.append(V_c,np.linspace(np.max(V_c),1,5))

hist,xedges,yedges = np.histogram2d(X_d2d,V_c,bins=(20,20))#,range=[[ymin,ymax],[xmin,xmax]])
aspectratio = 1.0*(np.max(X_d2d) - np.min(X_d2d))/(1.0*np.max(V_c) - np.min(V_c))
a=a1.imshow(hist.transpose(),extent=[np.min(X_d2d),np.max(X_d2d),np.min(V_c),np.max(V_c)],interpolation='nearest',origin='lower', aspect=aspectratio,cmap=plt.cm.gray_r)
a_=plt.colorbar(a,shrink=1.)
a_.ax.tick_params(labelsize = size_)
_min=np.array([0.51,0.51])
_max=np.array([0.23,0.23])
a1.errorbar((133./1000.,133./1000.),(0.54,0.54),xerr=21./1000.,yerr=[_min,_max],fmt='o',color='red',elinewidth=2)
a1.plot(133./1000.,0.54,'.',ms=10,color='red')
a1.set_xlabel(r'$d_{2d}$(h$^{-1}$Mpc)',fontsize='x-large')
a1.set_ylabel(r'$V_{c,sub}$/$V_{c,host}$',fontsize='x-large')
a1.set_title('$V_{c,host}>700$ kms$^{-1}$',fontsize='xx-large')
#a1.text(0.15,0.95,'$V_{c,host}>700$ kms$^{-1}$')
a1.tick_params(labelsize='x-large')
a1.set_xlim(0.1,1.77)
a1.set_ylim(0,1.)

#------------------------------------------------------------------------------------------------------------------------------------
a2=f.add_subplot(122,sharey=a1)
#------------------------------------------------------------------------------------------------------------------------------------

X_sd2d=np.append(dist_2d_XY_s_z0,dist_2d_XY_s_z3)
X_sd2d=np.append(X_sd2d,dist_2d_XY_s_z2)
X_sd2d=np.append(X_sd2d,dist_2d_XY_s_z1)
#X_sd2d=np.append(X_sd2d,np.linspace(0,0.91,15))
V_cs=np.append(V_circ_s_z0,V_circ_s_z3)
V_cs=np.append(V_cs,V_circ_s_z2)
V_cs=np.append(V_cs,V_circ_s_z1)
#V_cs=np.append(V_cs,np.linspace(0,0.1,15))

hist,xedges,yedges = np.histogram2d(X_sd2d,V_cs,bins=(20,20))#,range=[[ymin,ymax],[xmin,xmax]])
aspectratio = 1.0*(np.max(X_sd2d) - np.min(X_sd2d))/(1.0*np.max(V_cs) - np.min(V_cs))
b=a2.imshow(hist.transpose(),extent=[np.min(X_sd2d),np.max(X_sd2d),np.min(V_cs),np.max(V_cs)],interpolation='nearest',origin='lower', aspect=aspectratio,cmap=plt.cm.gray_r)
as_=plt.colorbar(b,shrink=1.)
as_.ax.tick_params(labelsize = size_)
a2.errorbar((133./1000.,133./1000.),(0.54,0.54),xerr=21./1000.,yerr=[_min,_max],fmt='o',color='red',elinewidth=2)
a2.plot(133./1000.,0.54,'o',ms=10,color='red')
a2.set_xlabel(r'$d_{2d}$(h$^{-1}$Mpc)',fontsize='x-large')
a2.set_ylabel(r'V$_{c,sub}$/$V_{c,host}$',fontsize='x-large')
a2.set_title('$300$ km s$^{-1} < V_{c,host} < 700 $ km s$^{-1}$',fontsize='xx-large')
#a2.text(0.15,0.95,'$300$ km s$^{-1} < V_{c,host} < 700 $ km s$^{-1}$')
a2.tick_params(labelsize='x-large')
a2.set_xlim(0.,np.max(X_sd2d))
a2.set_ylim(0,1.)

#------------------------------------------------------------------------------------------------------------------------------------
f=plt.figure(3)
#------------------------------------------------------------------------------------------------------------------------------------

Xoff_d2d=np.append(Xoffnew_h_z0,Xoffnew_h_z3)
Xoff_d2d=np.append(Xoff_d2d,Xoffnew_h_z2)
Xoff_d2d=np.append(Xoff_d2d,Xoffnew_h_z1)
V_ci=np.append(angulo_h_z0,angulo_h_z3)
V_ci=np.append(V_ci,angulo_h_z2)
V_ci=np.append(V_ci,angulo_h_z1)

a1=f.add_subplot(1,2,1)
hist,xedges,yedges = np.histogram2d(Xoff_d2d,V_ci,bins=(20,20))#,range=[[ymin,ymax],[xmin,xmax]])
aspectratio = 1.0*(np.max(Xoff_d2d) - np.min(Xoff_d2d))/(1.0*np.max(V_ci) - np.min(V_ci))
d_s=a1.imshow(hist.transpose(),extent=[np.min(Xoff_d2d),np.max(Xoff_d2d),np.min(V_ci),np.max(V_ci)],interpolation='nearest',origin='lower', aspect=aspectratio,cmap=plt.cm.gray_r)
ad_=plt.colorbar(d_s,shrink=0.85)
ad_.ax.tick_params(labelsize = size_)
a1.set_xlabel(r'X$_{off}$',fontsize='xx-large')
a1.set_ylabel(r'$\mu$',fontsize='xx-large')
a1.set_title('V$_{c,host}>700$ kms$^{-1}$',fontsize='xx-large')
a1.tick_params(labelsize='x-large')

Xoff_sd2d=np.append(Xoffnew_s_z0,Xoffnew_s_z3)
Xoff_sd2d=np.append(Xoff_sd2d,Xoffnew_s_z2)
Xoff_sd2d=np.append(Xoff_sd2d,Xoffnew_s_z1)

V_sci=np.append(angulo_s_z0,angulo_s_z3)
V_sci=np.append(V_sci,angulo_s_z2)
V_sci=np.append(V_sci,angulo_s_z1)


a2=f.add_subplot(1,2,2)
hist,xedges,yedges = np.histogram2d(Xoff_sd2d,V_sci,bins=(20,20))#,range=[[ymin,ymax],[xmin,xmax]])
aspectratio = 1.0*(np.max(Xoff_sd2d) - np.min(Xoff_sd2d))/(1.0*np.max(V_sci) - np.min(V_sci))
r_=a2.imshow(hist.transpose(),extent=[np.min(Xoff_sd2d),np.max(Xoff_sd2d),np.min(V_sci),np.max(V_sci)],interpolation='nearest',origin='lower', aspect=aspectratio,cmap=plt.cm.gray_r)
ae_=plt.colorbar(r_,shrink=0.85)
ae_.ax.tick_params(labelsize = size_)
a2.set_xlabel(r'X$_{off}$',fontsize='xx-large')
a2.set_ylabel(r'$\mu$',fontsize='xx-large')
a2.set_title('$300$ km s$^{-1} < $V$_{c,host} < 700 $ km s$^{-1}$',fontsize='xx-large')
a2.tick_params(labelsize='x-large')

#---------------------------------------------------------------------------

#f1=plt.figure(2,(18,8))
#af11=f1.add_subplot(1,2,1)
#af12=f1.add_subplot(1,2,2)
#
#cos_theta=0.5
#mask_s_ang0=(angulo_s_z0>=cos_theta)
#mask_s_ang3=(angulo_s_z3>=cos_theta)
#mask_s_ang2=(angulo_s_z2>=cos_theta)
#mask_s_ang1=(angulo_s_z1>=cos_theta)
#
#mask_h_ang0=(angulo_h_z0>=cos_theta)
#mask_h_ang3=(angulo_h_z3>=cos_theta)
#mask_h_ang2=(angulo_h_z2>=cos_theta)
#mask_h_ang1=(angulo_h_z1>=cos_theta)
#
#
#af11.plot(np.sort(dist_2d_XY_h_z0[mask_h_ang0])/1000.,1.-(np.arange(float(len(dist_2d_XY_h_z0[mask_h_ang0])))/float(len(dist_2d_XY_h_z0[mask_h_ang0]))),lw=3,color='gray',label='z=0')
#af11.plot(np.sort(dist_2d_XY_h_z3[mask_h_ang3])/1000.,1.-(np.arange(float(len(dist_2d_XY_h_z3[mask_h_ang3])))/float(len(dist_2d_XY_h_z3[mask_h_ang3]))),lw=3,color='blue',label='z=0.25')
#af11.plot(np.sort(dist_2d_XY_h_z2[mask_h_ang2])/1000.,1.-(np.arange(float(len(dist_2d_XY_h_z2[mask_h_ang2])))/float(len(dist_2d_XY_h_z2[mask_h_ang2]))),lw=3,color='red',label='z=0.5')
#af11.plot(np.sort(dist_2d_XY_h_z1[mask_h_ang1])/1000.,1.-(np.arange(float(len(dist_2d_XY_h_z1[mask_h_ang1])))/float(len(dist_2d_XY_h_z1[mask_h_ang1]))),lw=3,color='green',label='z=1')
#
#af11.plot(np.sort(dist_2d_XY_s_z0[mask_s_ang0])/1000.,1.-(np.arange(float(len(dist_2d_XY_s_z0[mask_s_ang0])))/float(len(dist_2d_XY_s_z0[mask_s_ang0]))),ls='--',color='gray',label='z=0')
#af11.plot(np.sort(dist_2d_XY_s_z3[mask_s_ang3])/1000.,1.-(np.arange(float(len(dist_2d_XY_s_z3[mask_s_ang3])))/float(len(dist_2d_XY_s_z3[mask_s_ang3]))),ls='--',color='blue',label='z=0.25')
#af11.plot(np.sort(dist_2d_XY_s_z2[mask_s_ang2])/1000.,1.-(np.arange(float(len(dist_2d_XY_s_z2[mask_s_ang2])))/float(len(dist_2d_XY_s_z2[mask_s_ang2]))),ls='--',color='red',label='z=0.5')
#af11.plot(np.sort(dist_2d_XY_s_z1[mask_s_ang1])/1000.,1.-(np.arange(float(len(dist_2d_XY_s_z1[mask_s_ang1])))/float(len(dist_2d_XY_s_z1[mask_s_ang1]))),ls='--',color='green',label='z=1')
#af11.axvline(x=133./1000.,color='black',linestyle='-')
##af11.axvspan((124./0.6777)*2./1000.-(20./0.6777)*2./1000.,(124./0.6777)*2./1000.+(20./0.6777)*2./1000., facecolor='gray', alpha=0.02)
#af11.axvline(x=(133./1000.)+(21./1000.),color='black',linestyle='--')
#af11.axvline(x=(133./1000.)-(21./1000.),color='black',linestyle='--')
#
#af11.set_ylabel(r'P(>$d_{2d}$)',fontsize=22)
#af11.set_xlabel(r'$d_{2d}$(h$^{-1}$Mpc)',fontsize=22)
#af11.tick_params(labelsize=22)
##af11.legend(loc=1,numpoints=1,fontsize='medium')
#af11.set_yscale('log')
#
#
#af12.plot(np.sort(Xoffnew_h_z0[mask_h_ang0]),1.-(np.arange(float(len(Xoffnew_h_z0[mask_h_ang0])))/float(len(Xoffnew_h_z0[mask_h_ang0]))),lw=3,color='gray',label='z=0')
#af12.plot(np.sort(Xoffnew_h_z3[mask_h_ang3]),1.-(np.arange(float(len(Xoffnew_h_z3[mask_h_ang3])))/float(len(Xoffnew_h_z3[mask_h_ang3]))),lw=3,color='blue',label='z=0.25')
#af12.plot(np.sort(Xoffnew_h_z2[mask_h_ang2]),1.-(np.arange(float(len(Xoffnew_h_z2[mask_h_ang2])))/float(len(Xoffnew_h_z2[mask_h_ang2]))),lw=3,color='red',label='z=0.5')
#af12.plot(np.sort(Xoffnew_h_z1[mask_h_ang1]),1.-(np.arange(float(len(Xoffnew_h_z1[mask_h_ang1])))/float(len(Xoffnew_h_z1[mask_h_ang1]))),lw=3,color='green',label='z=1')
#
#af12.plot(np.sort(Xoffnew_s_z0[mask_s_ang0]),1.-(np.arange(float(len(Xoffnew_s_z0[mask_s_ang0])))/float(len(Xoffnew_s_z0[mask_s_ang0]))),ls='--',color='gray',label='z=0')
#af12.plot(np.sort(Xoffnew_s_z3[mask_s_ang3]),1.-(np.arange(float(len(Xoffnew_s_z3[mask_s_ang3])))/float(len(Xoffnew_s_z3[mask_s_ang3]))),ls='--',color='blue',label='z=0.25')
#af12.plot(np.sort(Xoffnew_s_z2[mask_s_ang2]),1.-(np.arange(float(len(Xoffnew_s_z2[mask_s_ang2])))/float(len(Xoffnew_s_z2[mask_s_ang2]))),ls='--',color='red',label='z=0.5')
#af12.plot(np.sort(Xoffnew_s_z1[mask_s_ang1]),1.-(np.arange(float(len(Xoffnew_s_z1[mask_s_ang1])))/float(len(Xoffnew_s_z1[mask_s_ang1]))),ls='--',color='green',label='z=1')
#
#af12.set_ylabel(r'P(>X$_{off}$)',fontsize=22)
#af12.set_xlabel(r'X$_{off}$',fontsize=22)
#af12.tick_params(labelsize=22)
#af12.legend(loc=3,numpoints=1,fontsize=22)
#af12.set_yscale('log')
#
#fig=plt.figure(4)
#ax=fig.add_subplot(1,1,1)
#
##____________________________________________________________________________________________________________________________________
##Forero et al. (2010)
##Parameters
#
#h=0.7
#z_=0.
#Po=0.04
#d_star=200.*h
#alpha=-1.
#dz_=np.sort(distbarion_s_z0[distbarion_s_z0>0.])/np.sqrt(1.+z_)
#
#P2D_=Po*((dz_/d_star)**(alpha))*np.exp(-dz_/d_star)
#
##____________________________________________________________________________________________________________________________________
#
#ax.plot(np.sort(distbarion_h_z0)/1000.,1.-(np.arange(float(len(distbarion_h_z0)))/float(len(distbarion_h_z0))),'-',color='gray',label='z=0')
#ax.plot(np.sort(distbarion_h_z3)/1000.,1.-(np.arange(float(len(distbarion_h_z3)))/float(len(distbarion_h_z3))),'-',color='blue',label='z=0.25')
#ax.plot(np.sort(distbarion_h_z2)/1000.,1.-(np.arange(float(len(distbarion_h_z2)))/float(len(distbarion_h_z2))),'-',color='red',label='z=0.5')
#ax.plot(np.sort(distbarion_h_z1)/1000.,1.-(np.arange(float(len(distbarion_h_z1)))/float(len(distbarion_h_z1))),'-',color='green',label='z=1')
#
#a0,b0=np.sort(distbarion_s_z0)/1000., 1.-(np.arange(float(len(distbarion_s_z0)))/float(len(distbarion_s_z0)))
#a3,b3=np.sort(distbarion_s_z3)/1000., 1.-(np.arange(float(len(distbarion_s_z3)))/float(len(distbarion_s_z3)))
#a2,b2=np.sort(distbarion_s_z2)/1000., 1.-(np.arange(float(len(distbarion_s_z2)))/float(len(distbarion_s_z2)))
#a1,b1=np.sort(distbarion_s_z1)/1000., 1.-(np.arange(float(len(distbarion_s_z1)))/float(len(distbarion_s_z1)))
#
#ax.plot(a0[a0==0.],b0[a0==0.],'.',color='gray')
#ax.plot(a3[a3==0.],b3[a3==0.],'.',color='blue')
#ax.plot(a2[a2==0.],b2[a2==0.],'.',color='red')
#ax.plot(a1[a1==0.],b1[a1==0.],'.',color='green')
#
#ax.plot(np.sort(distbarion_s_z0)/1000.,1.-(np.arange(float(len(distbarion_s_z0)))/float(len(distbarion_s_z0))),ls='--',ms=0.001,color='gray',label='z=0')
#ax.plot(np.sort(distbarion_s_z3)/1000.,1.-(np.arange(float(len(distbarion_s_z3)))/float(len(distbarion_s_z3))),ls='--',ms=0.001,color='blue',label='z=0.25')
#ax.plot(np.sort(distbarion_s_z2)/1000.,1.-(np.arange(float(len(distbarion_s_z2)))/float(len(distbarion_s_z2))),ls='--',ms=0.001,color='red',label='z=0.5')
#ax.plot(np.sort(distbarion_s_z1)/1000.,1.-(np.arange(float(len(distbarion_s_z1)))/float(len(distbarion_s_z1))),ls='--',ms=0.001,color='green',label='z=1')
#ax.axvline(x=86.8/1000.,color='black',linestyle='-')
#ax.axvline(x=86.8/1000.+14./1000.,color='black',linestyle='--')
#ax.axvline(x=86.8/1000.-14./1000.,color='black',linestyle='--')
#
#ax.plot(np.sort(distbarion_s_z0[distbarion_s_z0>0.])/1000.,P2D_,lw=3,color='black',label='$P_{2D}$')
#
#ax.set_ylabel(r'P(>$d_{2d}^{bar}$)',fontsize=22)
#ax.set_xlabel(r'$d_{2d}^{bar}$ h$^{-1}$Mpc',fontsize=22)
#ax.tick_params(labelsize=22)
#ax.legend(loc=1,numpoints=1,fontsize=18)
#ax.set_yscale('log')
#ax.set_ylim(10**-4,10**0)

plt.show()

