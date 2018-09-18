import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

data_path=''
perfff=np.zeros((8,13))*np.nan
beststrg=np.zeros((8,13))

perfffb=np.zeros((8,13))*np.nan
beststrgb=np.zeros((8,13))
#plt.plot([0, 1])
for ci in range(8):
	#plt.subplot(4,5,ci+1)
	for li in range(13):
		savstrb='attnGRAD1Ntraindetect2_40b_o'+str((ci+1)*20)+'_L'+str(li)+'TFP.npz' #these need to be renamed according to the format in the Ori_Performance.py if you've run the simulation yourself rather than downloaded performance data from dryad
		savstr='attnTCtraindetect2_40b_o'+str((ci+1)*20)+'_L'+str(li)+'TFP.npz';

		F=np.load(data_path+savstr); TP=F['arr_0']; TN=F['arr_1']
                print(TP)
		BLperf=((TP+TN)/2)[0]
		perfff[ci,li]=np.nanmax((TP+TN)/2-BLperf)*100; beststrg[ci,li]=np.nanargmax((TP+TN)/2-BLperf)
		F=np.load(data_path+savstrb); TP=F['arr_0']; TN=F['arr_1']
		BLperfb=((TP+TN)/2)[0]
		perfffb[ci,li]=np.nanmax((TP+TN)/2-BLperfb)*100; beststrgb[ci,li]=np.nanargmax((TP+TN)/2-BLperfb)


ax1=plt.subplot(1,1,1); #ax2 = ax1.twinx()
for label in (ax1.get_xticklabels() + ax1.get_yticklabels() ):
    			#label.set_fontname('Arial')
    			label.set_fontsize(20)

ax1.errorbar(x=np.arange(1,14),y=np.mean(perfff,axis=0),yerr=np.std(perfff,axis=0)/np.sqrt(20),linewidth=3,color='k',ls='-')
ax1.errorbar(x=np.arange(1,14),y=np.mean(perfffb,axis=0),yerr=np.std(perfffb,axis=0)/np.sqrt(20),linewidth=3,color='k',ls=':')
ax1.set_xticks([1,3,5,7,9,11,13])



plt.show()
