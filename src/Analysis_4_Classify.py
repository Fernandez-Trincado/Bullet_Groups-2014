#!/usr/bin/python

#Bullet Cluster and Bullet Groups with Jaime Forero at Universidad de Los Andes, Bogota - Colombia.

import numpy as np
import scipy as sc 
import pylab as plt
import sys

data_in=sc.genfromtxt(sys.argv[1]) #Input file, example: Host_1000kms.dat
Catg=sc.genfromtxt(sys.argv[2])    #Comparative catalog
#Catg=sc.genfromtxt('MassiveV150CatshortV.0416.DAT')
#Catg=sc.genfromtxt('MassiveV75CatshortV.0416.DAT')

files=open(sys.argv[1]+'_substructure.dat','a')

for i in np.arange(len(data_in[:,0])):

	mask=Catg[:,14]==data_in[i,11]

	N=np.size(Catg[mask,14])

	if N>=1.:

		print '[ID='+str(int(i)+1)+'] = Is greater than 1'

		init=np.vstack(Catg[mask,:])
		index=np.argsort(init[:,10]) # Circular velocity

		sc.savetxt(files,np.column_stack(init[index[-1],:]),fmt='%s')

	else: 

		print '[ID='+str(int(i)+1)+'] = Is less than two'
		print "NO DATA"

files.close()

