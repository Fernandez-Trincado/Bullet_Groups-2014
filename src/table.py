#!/usr/bin/python




import numpy as np
import scipy as sc
import pylab as plt


host_z0=sc.genfromtxt('Host_700kms_z0.dat')
sub_z0=sc.genfromtxt('Host_700kms_z0.dat_substructure.dat')

host_z3=sc.genfromtxt('Host_700kms_z3.dat')
sub_z3=sc.genfromtxt('Host_700kms_z3.dat_substructure.dat')

host_z2=sc.genfromtxt('Host_700kms_z2.dat')
sub_z2=sc.genfromtxt('Host_700kms_z2.dat_substructure.dat')

host_z1=sc.genfromtxt('Host_700kms_z1.dat')
sub_z1=sc.genfromtxt('Host_700kms_z1.dat_substructure.dat')


print "z=0 :", "Host Halo="+str(len(host_z0[:,0])), "Sub="+str(len(sub_z0[:,0]))
print "z=0 :", "Host Halo="+str(np.max(host_z0[:,6])),str(np.min(host_z0[:,6])), "Sub="+str(np.max(sub_z0[:,6])), str(np.min(sub_z0[:,6]))

print "z=0.25 :", "Host Halo="+str(len(host_z3[:,0])), "Sub="+str(len(sub_z3[:,0]))
print "z=0.25 :", "Host Halo="+str(np.max(host_z3[:,6])),str(np.min(host_z3[:,6])), "Sub="+str(np.max(sub_z3[:,6])), str(np.min(sub_z3[:,6]))


print "z=0.5 :", "Host Halo="+str(len(host_z2[:,0])), "Sub="+str(len(sub_z2[:,0]))
print "z=0.5 :", "Host Halo="+str(np.max(host_z2[:,6])),str(np.min(host_z2[:,6])), "Sub="+str(np.max(sub_z2[:,6])), str(np.min(sub_z2[:,6]))


print "z=1 :", "Host Halo="+str(len(host_z1[:,0])), "Sub="+str(len(sub_z1[:,0]))
print "z=1 :", "Host Halo="+str(np.max(host_z1[:,6])),str(np.min(host_z1[:,6])), "Sub="+str(np.max(sub_z1[:,6])), str(np.min(sub_z1[:,6]))




