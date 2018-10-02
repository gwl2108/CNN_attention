import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

'''this script plots the criteria, sensitivity, and performance for different attention strengths when attention is applied according to features and/or spatial location. Run with python 2.7. Contact: gracewlindsay@gmail.com'''


mpl.rcParams['font.size']=22
lc=8 #number of categories
betas = np.arange(0,1,.5)
ns = len(betas) #number of strengths used
lays=[12] #layers attention applited at
def critsens(li):
 quad=1 
#load them and get tps etc
 cii=0; 
 c=np.zeros((lc,ns)); dprime=np.zeros((lc,ns)); c1=np.zeros((lc,ns)); dprime1=np.zeros((lc,ns)); c2=np.zeros((lc,ns)); dprime2=np.zeros((lc,ns))
 c3=np.zeros((lc,ns)); dprime3=np.zeros((lc,ns))
 tp=np.zeros((lc,ns))*np.nan; tp1=np.zeros((lc,ns))*np.nan; tp2=np.zeros((lc,ns))*np.nan; tp3=np.zeros((lc,ns))*np.nan; 
 fp=np.zeros((lc,ns))*np.nan; fp1=np.zeros((lc,ns))*np.nan; fp2=np.zeros((lc,ns))*np.nan; fp3=np.zeros((lc,ns))*np.nan; 
 for ci in [20, 40, 60, 80, 100, 120, 140, 160]:
        #run simulation code provided on github to get these files:
	savstr='Ori'+'Spat'+'Attn'+'_aTCs'+'_'+str(ci)+'o'+str(li)+'l'+'_'+'SFperf.npz' 
	savstr1='Ori'+'Spat'+'Attn'+'_aTCs'+'_'+str(ci)+'o'+str(li)+'l'+'_'+'Sperf.npz' 
	savstr2='Ori'+'Spat'+'Attn'+'_aTCs'+'_'+str(ci)+'o'+str(li)+'l'+'_'+'Fperf.npz' 
	savstr3='Ori'+'Spat'+'Attn'+'_aGRADs'+'_'+str(ci)+'o'+str(li)+'l'+'_'+'Fperf.npz' 

	F=np.load(savstr); tp[cii,:]=F['arr_0']; TN=F['arr_1']; fp[cii,:]=1-TN
	F=np.load(savstr1); tp1[cii,:]=F['arr_0']; TN=F['arr_1']; fp1[cii,:]=1-TN
	F=np.load(savstr2); tp2[cii,:]=F['arr_0']; TN=F['arr_1']; fp2[cii,:]=1-TN 
	F=np.load(savstr3); tp3[cii,:]=F['arr_0']; TN=F['arr_1']; fp3[cii,:]=1-TN 

        #correction:
	tp[tp==1]=.99; fp[fp==0]=.01
	tp1[tp1==1]=.99; fp1[fp1==0]=.01
	tp2[tp2==1]=.99; fp2[fp2==0]=.01
	tp3[tp3==1]=.99; fp3[fp3==0]=.01

	fp[fp==1]=.99; tp[tp==0]=.01
	fp1[fp1==1]=.99; tp1[tp1==0]=.01
	fp2[fp2==1]=.99; tp2[tp2==0]=.01
	fp3[fp3==1]=.99; tp3[tp3==0]=.01


	cii+=1
 return tp,tp1,tp2,tp3,fp,fp1,fp2,fp3
#st=1; 
cAlay=[]; cA2lay=[]; cA3lay=[]; cA1lay=[]; dprimeAlay=[]; dprimeA2lay=[]; dprimeA3lay=[]; dprimeA1lay=[]
perflay=[]; perf1lay=[]; perf2lay=[]; perf3lay=[]
lii=-1 
lss=['-','-']
for li in lays:
 plt.figure()
 lii+=1
 tp,tp1,tp2,tp3,fp,fp1,fp2,fp3=critsens(li) 
 print tp, fp
 perf=(tp+(1-fp))/2.
 perf1=(tp1+(1-fp1))/2.
 perf2=(tp2+(1-fp2))/2.
 perf3=(tp3+(1-fp3))/2.

 cA=-.5*(norm.ppf(np.mean(tp,axis=0))+norm.ppf(np.mean(fp,axis=0)))
 dprimeA=norm.ppf(np.mean(tp,axis=0))-norm.ppf(np.mean(fp,axis=0))
 cA1=-.5*(norm.ppf(np.mean(tp1,axis=0))+norm.ppf(np.mean(fp1,axis=0)))
 dprimeA1=norm.ppf(np.mean(tp1,axis=0))-norm.ppf(np.mean(fp1,axis=0))
 cA2=-.5*(norm.ppf(np.mean(tp2,axis=0))+norm.ppf(np.mean(fp2,axis=0)))
 dprimeA2=norm.ppf(np.mean(tp2,axis=0))-norm.ppf(np.mean(fp2,axis=0))
 cA3=-.5*(norm.ppf(np.mean(tp3,axis=0))+norm.ppf(np.mean(fp3,axis=0)))
 dprimeA3=norm.ppf(np.mean(tp3,axis=0))-norm.ppf(np.mean(fp3,axis=0))
 print cA, dprimeA

#plot all over strg
 plt.subplot(131)

 plt.plot(betas,cA,color='k',linewidth=3,ls=lss[lii]); 
 plt.plot(betas,cA1,color='r',linewidth=3,ls=lss[lii]); 
 plt.plot(betas,cA2,color='b',linewidth=3,ls=lss[lii]);
 plt.plot(betas,cA3,color='b',linewidth=3,ls=':');
 plt.ylabel('Criteria')
 plt.legend(['Both','Spat','Feat-TC','Feat-G'], fontsize='small',loc=2) 
 #plt.title('Criteria')
 plt.subplot(132)

 plt.plot(betas,dprimeA,color='k',linewidth=3,ls=lss[lii]); 
 plt.plot(betas,dprimeA1,color='r',linewidth=3,ls=lss[lii]); 
 plt.plot(betas,dprimeA2,color='b',linewidth=3,ls=lss[lii]); 
 plt.plot(betas,dprimeA3,color='b',linewidth=3,ls=':');
 plt.ylabel('Sensitivity')
 #plt.title('Sensitivity')
 plt.subplot(133)
 plt.errorbar(betas,np.nanmean(perf,axis=0),np.nanstd(perf,axis=0)/np.sqrt(lc)*0,color='k',linewidth=3,ls=lss[lii]); 
 plt.errorbar(betas,np.nanmean(perf1,axis=0),np.nanstd(perf1,axis=0)/np.sqrt(lc)*0,color='r',linewidth=3,ls=lss[lii]); 
 plt.errorbar(betas,np.nanmean(perf2,axis=0),np.nanstd(perf2,axis=0)/np.sqrt(lc)*0,color='b',linewidth=3,ls=lss[lii]); 
 plt.errorbar(betas,np.nanmean(perf3,axis=0),np.nanstd(perf3,axis=0)/np.sqrt(lc)*0,color='b',linewidth=3,ls=':');
 plt.ylabel('Performance')

plt.show()
