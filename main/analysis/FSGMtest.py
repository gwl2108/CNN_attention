import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

'''This script calculates and plots the slope and intercept of lines defined by activity ratios (attention/no-attention), for attention applied with TCs or Grads'''

def later_effect(lattn,TCs):

 allcoefslist=[]
 layinds=[0,0,1,1,2,2,2,3,3,3,4,4,4]; ilayi=[0,1,0,1,0,1,2,0,1,2,0,1,2]

 with open("featvecs_ORIgrats40Astd.txt", "rb") as fp: #ori tuning curves file
 	  oriTCs = pickle.load(fp)  


 oris=np.arange(0,180,20); totO=len(oris)
 allcoefs=np.zeros((13,2))

 for li in range(0,13):
	loriTCs=oriTCs[li] # ori x fm

	rankindz=np.argsort(loriTCs,axis=0)
	FMs=np.shape(loriTCs)[1]
	prefz=np.argmax(loriTCs,axis=0) #preferred ori
	antiprefz=np.argmin(loriTCs,axis=0)
	AllCombos=np.zeros((FMs,totO,4,totO))

	
	for oi in range(len(oris)):
                #these files are in the ori_activity zip file on github
                if TCs:
		    hldr=np.load('FFrec_attnTCtrain'+str(oi*20)+'_L'+str(lattn)+'.npz') 
                else:
		    hldr=np.load('FFrec_attnGRAD1Ntrain'+str(oi*20)+'_L'+str(lattn)+'.npz') 
		allAll=hldr['arr_0']; allresps=allAll[layinds[li]]; resps=allresps[ilayi[li],:,:,:]
		AllCombos[:,:,:,oi]=resps

        strg_ind = 2 #beta value index
	ModInds=np.diagonal(AllCombos[:,:,strg_ind,:],axis1=1, axis2=2)/np.diagonal(AllCombos[:,:,0,:],axis1=1, axis2=2)# fm x ori
	
	modlins=np.zeros((FMs,5));
	modlins[:,0]=np.diag(ModInds[:,prefz])

	#line-to-fit is made by "folding over" the tuning curve (averaging activity values for symmetric values around preferred ori): 
	piii=0
	for pii in range(7,-1,-2):
		piii+=1
		modlins[:,piii]=(np.diag(ModInds[:,rankindz[pii,:]])+np.diag(ModInds[:,rankindz[pii-1,:]]))/2.


        modlins=modlins.T; RegCoefs=np.zeros((2,modlins.shape[1]))
        sii=-1
        for si in range(modlins.shape[1]):
         idx=np.isfinite(modlins[:,si]) #need to exclude nans etc
         if sum(idx)>4:
           sii+=1
           #print idx
	   RegCoefs[:,sii]=np.polyfit(np.arange(5)[idx],modlins[idx,si],1) #get slope and inter
        RegCoefs=RegCoefs[:,0:sii+1]

        useindz=np.arange(RegCoefs.shape[1]) #can choose to exclude certain FMs here if needed
	allcoefs[li,1]=np.nanmedian(RegCoefs[1,useindz])-1; #median adjusted inter
        allcoefs[li,0]=np.nanmedian(RegCoefs[0,useindz]); #median slope
	allcoefslist.append(RegCoefs[:,useindz]) 



 return allcoefs, allcoefslist
	
mpl.rcParams['font.size'] = 30
cols=['r','g','b','y','k','.1']; lss=['-','--',':']
lii=-1; lays=[1,5,7,9,11] #layers attention applied at
alp=1 #.4
Allcoefs =[]
for li in lays:
	if li==13:
		alp=1 
	vals,coefs=later_effect(li,True) #True = TCs, False = Grads
        Allcoefs.append(coefs)
	lii+=1
	for vi in range(2):
		plt.plot(np.arange(13)+1,vals[:,vi],ls=lss[vi],color=cols[lii],alpha=alp,linewidth=3);

plt.xlabel('Layer Recorded From'); plt.ylabel('Fit Values')
plt.show()

