import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

#subdir = sys.argv[1]

try:
	os.mkdir("mimages")
#	os.mkdir("%s/mimages" % subdir)
except:
	pass
	
err1 = np.loadtxt("dense-dense/error.txt")
err2 = np.loadtxt("conv-conv/error.txt")
err3 = np.loadtxt("lstm-lstm/error.txt")

for i in range(0,10001,20):
	gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
	
	igs = gridspec.GridSpecFromSubplotSpec(1,3, gs[0])
	
	data = np.loadtxt("dense-dense/messages/%.6d.txt" % (i))
	plt.subplot(igs[0])
	plt.imshow(data,vmin=-1,vmax=1,interpolation='nearest',aspect=0.13, cmap=plt.cm.BuGn)
	plt.xticks([])
	plt.yticks([])
	plt.title("Dense")	
	
	data = np.loadtxt("conv-conv/messages/%.6d.txt" % (i))
	plt.subplot(igs[1])
	plt.imshow(data,vmin=-1,vmax=1,interpolation='nearest',aspect=0.13, cmap=plt.cm.BuGn)
	plt.xticks([])
	plt.yticks([])
	plt.title("Conv")
	
	data = np.loadtxt("lstm-lstm/messages/%.6d.txt" % (i))
	plt.subplot(igs[2])
	plt.imshow(data,vmin=-1,vmax=1,interpolation='nearest',aspect=0.13, cmap=plt.cm.BuGn)
	plt.xticks([])
	plt.yticks([])
	plt.title("LSTM")

	ax = plt.subplot(gs[1])
	plt.plot(err1[:,1],label="Dense")
	plt.plot(err2[:,1],label="Conv")
	plt.plot(err3[:,1],label="LSTM")
	#box = ax.get_position()
	#ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
	plt.legend(loc='lower left')#, bbox_to_anchor = (1,0.5))
	plt.axvline(i,0,1)
	plt.ylim(0.4,0.7)
	plt.xlabel("Epoch")
	plt.ylabel("Log-Loss")
	plt.gcf().set_size_inches((11,8))
	#plt.gcf().subplots_adjust(hspace=0.5)
	plt.savefig("mimages/%.6d.png" % (i), bbox_inches='tight')
	plt.clf()
	
